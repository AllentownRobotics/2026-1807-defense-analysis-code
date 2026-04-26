#!/usr/bin/env python3
"""
Analyze FRC match data to estimate defensive strength.

Inputs:
  - TBA-derived matches CSV from the earlier pull script
    Required columns:
      event_key, comp_level,
      red_score, blue_score,
      red_team_keys, blue_team_keys

Outputs:
  - <stem>_team_ratings.csv
  - <stem>_team_ratings_ranked_defense.csv
  - optional per-event ratings if --by-event is given

Metrics:
  - OPR / DPR / CCWM            : OLS team ratings
  - ridge_offense / ridge_defense : ridge-adjusted joint offense/defense
  - suppression_rating_abs/pct  : ridge regression of expected-vs-actual gap
  - defensive_specialist_index  : composite, higher is better
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


VALID_COMP_LEVELS = {"qm", "ef", "qf", "sf", "f"}

# Suppression cleaning parameters
MIN_EXPECTED_SCORE = 50       # ignore weak opponent expectations
MAX_SUPPRESSION_PCT = 0.75    # cap extreme positive % suppression
MIN_SUPPRESSION_PCT = -0.50   # cap extreme negative % suppression
MIN_TOTAL_MATCH_SCORE = 50    # drop broken / very low-info matches
MIN_SUPPRESSION_SAMPLES = 5   # drop teams with too few suppression samples


# ---------------------------------------------------------------------------
# Data loading / normalization
# ---------------------------------------------------------------------------

def parse_team_list(value: str) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def normalize_matches(df: pd.DataFrame, quals_only: bool = False) -> pd.DataFrame:
    required = {
        "event_key", "comp_level",
        "red_score", "blue_score",
        "red_team_keys", "blue_team_keys",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["comp_level"] = df["comp_level"].astype(str).str.strip().str.lower()
    df = df[df["comp_level"].isin(VALID_COMP_LEVELS)]

    if quals_only:
        df = df[df["comp_level"] == "qm"]

    df["red_score"] = pd.to_numeric(df["red_score"], errors="coerce")
    df["blue_score"] = pd.to_numeric(df["blue_score"], errors="coerce")
    df = df[
        (df["red_score"] >= 0)
        & (df["blue_score"] >= 0)
        & ((df["red_score"] + df["blue_score"]) >= MIN_TOTAL_MATCH_SCORE)
    ]

    df["red_teams"] = df["red_team_keys"].apply(parse_team_list)
    df["blue_teams"] = df["blue_team_keys"].apply(parse_team_list)
    df = df[(df["red_teams"].map(len) > 0) & (df["blue_teams"].map(len) > 0)]

    return df.reset_index(drop=True)


def rescale_scores_per_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rescale red_score and blue_score so each event has the same mean alliance score.
    Equalizes high-scoring vs low-scoring events when fitting cross-event models.
    """
    df = df.copy()
    alliance_scores = pd.concat([df["red_score"], df["blue_score"]])
    global_mean = alliance_scores.mean()
    event_means = (
        pd.concat([df[["event_key", "red_score"]].rename(columns={"red_score": "s"}),
                   df[["event_key", "blue_score"]].rename(columns={"blue_score": "s"})])
        .groupby("event_key")["s"].mean()
    )
    scale = (global_mean / event_means).rename("_scale")
    df = df.merge(scale, left_on="event_key", right_index=True)
    df["red_score"] = df["red_score"] * df["_scale"]
    df["blue_score"] = df["blue_score"] * df["_scale"]
    return df.drop(columns=["_scale"]).reset_index(drop=True)


def get_all_teams(df: pd.DataFrame) -> list[str]:
    teams: set[str] = set()
    for row in df["red_teams"]:
        teams.update(row)
    for row in df["blue_teams"]:
        teams.update(row)
    return sorted(teams)


# ---------------------------------------------------------------------------
# Shared alliance design matrix
# ---------------------------------------------------------------------------

def build_alliance_design(
    df: pd.DataFrame, teams: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-alliance-appearance arrays. Each match contributes 2 rows
    (red appearance, then blue appearance).

    Returns:
        own_X    : (2*n_matches, n_teams) 1.0 for own-alliance teams
        opp_X    : (2*n_matches, n_teams) 1.0 for opposing teams
        own_score: own alliance score
        opp_score: opposing alliance score
    """
    team_index = {t: i for i, t in enumerate(teams)}
    n_alliances = len(df) * 2
    n_teams = len(teams)

    own_X = np.zeros((n_alliances, n_teams), dtype=float)
    opp_X = np.zeros((n_alliances, n_teams), dtype=float)
    own_score = np.zeros(n_alliances, dtype=float)
    opp_score = np.zeros(n_alliances, dtype=float)

    red_lists = df["red_teams"].tolist()
    blue_lists = df["blue_teams"].tolist()
    red_scores = df["red_score"].to_numpy(dtype=float)
    blue_scores = df["blue_score"].to_numpy(dtype=float)

    for i, (red, blue) in enumerate(zip(red_lists, blue_lists)):
        r_row, b_row = 2 * i, 2 * i + 1
        red_idx = [team_index[t] for t in red]
        blue_idx = [team_index[t] for t in blue]

        own_X[r_row, red_idx] = 1.0
        opp_X[r_row, blue_idx] = 1.0
        own_X[b_row, blue_idx] = 1.0
        opp_X[b_row, red_idx] = 1.0

        own_score[r_row] = red_scores[i]
        opp_score[r_row] = blue_scores[i]
        own_score[b_row] = blue_scores[i]
        opp_score[b_row] = red_scores[i]

    return own_X, opp_X, own_score, opp_score


def solve_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge: beta = (X'X + alpha*I)^-1 X'y."""
    reg = alpha * np.eye(X.shape[1])
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------

def compute_suppression(
    own_X: np.ndarray,
    opp_X: np.ndarray,
    own_score: np.ndarray,
    ridge_offense: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    For each alliance appearance, compute expected score from ridge_offense
    and the gap to the actual score. Drop rows with weak expectations.

    The defending teams for each kept row are the opposing alliance.
    """
    expected_all = own_X @ ridge_offense
    keep = expected_all > MIN_EXPECTED_SCORE

    expected = expected_all[keep]
    actual = own_score[keep]
    suppression = expected - actual
    suppression_pct = np.clip(
        suppression / expected, MIN_SUPPRESSION_PCT, MAX_SUPPRESSION_PCT
    )

    return {
        "defense_X": opp_X[keep],
        "expected": expected,
        "actual": actual,
        "suppression": suppression,
        "suppression_pct": suppression_pct,
    }


def compute_team_suppression_summary(
    teams: list[str], sup: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Per-defending-team descriptive suppression metrics."""
    defense_X = sup["defense_X"]
    samples = defense_X.sum(axis=0)
    expected_total = defense_X.T @ sup["expected"]
    actual_total = defense_X.T @ sup["actual"]
    suppression_total = defense_X.T @ sup["suppression"]

    suppression_pct = sup["suppression_pct"]
    avg_match_pct = np.full(len(teams), np.nan)
    for j in range(len(teams)):
        mask = defense_X[:, j] > 0
        if mask.any():
            avg_match_pct[j] = float(np.median(suppression_pct[mask]))

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_expected = np.where(samples > 0, expected_total / samples, np.nan)
        avg_actual = np.where(samples > 0, actual_total / samples, np.nan)
        avg_abs = np.where(samples > 0, suppression_total / samples, np.nan)
        agg_pct = np.where(expected_total > 0, suppression_total / expected_total, np.nan)

    return pd.DataFrame(
        {
            "suppression_samples": samples.astype(int),
            "expected_score_against_total": expected_total,
            "actual_score_against_total": actual_total,
            "avg_expected_score_against": avg_expected,
            "avg_actual_score_against": avg_actual,
            "avg_absolute_suppression": avg_abs,
            "aggregate_suppression_pct": agg_pct,
            "avg_match_suppression_pct": avg_match_pct,
        },
        index=teams,
    )


# ---------------------------------------------------------------------------
# Misc per-team stats
# ---------------------------------------------------------------------------

def compute_match_counts(own_X: np.ndarray, teams: list[str]) -> pd.Series:
    return pd.Series(own_X.sum(axis=0).astype(int), index=teams, name="match_count")


def compute_strength_of_schedule(
    opp_X: np.ndarray,
    ridge_offense: np.ndarray,
    own_X: np.ndarray,
    teams: list[str],
) -> pd.Series:
    """Average opposing-alliance offensive strength faced by each team."""
    opp_strength = opp_X @ ridge_offense
    samples = own_X.sum(axis=0)
    totals = own_X.T @ opp_strength
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(samples > 0, totals / samples, np.nan)
    return pd.Series(avg, index=teams, name="avg_opponent_offense_seen")


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_frame(df: pd.DataFrame, ridge_alpha: float) -> pd.DataFrame:
    teams = get_all_teams(df)
    if not teams:
        raise ValueError("No teams found in filtered match set.")

    own_X, opp_X, own_score, opp_score = build_alliance_design(df, teams)
    n_teams = len(teams)

    # Informational OLS offense
    opr, *_ = np.linalg.lstsq(own_X, own_score, rcond=None)

    # Joint ridge offense/defense:
    #   score ~= sum(offense[own]) - sum(defense[opp])
    X_rd = np.hstack([own_X, -opp_X])
    beta_rd = solve_ridge(X_rd, own_score, alpha=ridge_alpha)
    ridge_off = beta_rd[:n_teams]
    ridge_def = beta_rd[n_teams:]

    # Suppression, derived from a single pass
    sup = compute_suppression(own_X, opp_X, own_score, ridge_off)
    suppression_summary = compute_team_suppression_summary(teams, sup)
    suppression_rating_abs = solve_ridge(sup["defense_X"], sup["suppression"], alpha=ridge_alpha)
    suppression_rating_pct = solve_ridge(sup["defense_X"], sup["suppression_pct"], alpha=ridge_alpha)

    ratings = pd.concat(
        [
            compute_match_counts(own_X, teams),
            pd.Series(opr, index=teams, name="opr"),
            pd.Series(ridge_off, index=teams, name="ridge_offense"),
            pd.Series(ridge_def, index=teams, name="ridge_defense"),
            pd.Series(suppression_rating_abs, index=teams, name="suppression_rating_abs"),
            pd.Series(suppression_rating_pct, index=teams, name="suppression_rating_pct"),
            suppression_summary,
            compute_strength_of_schedule(opp_X, ridge_off, own_X, teams),
        ],
        axis=1,
    ).reset_index().rename(columns={"index": "team_key"})

    # Composite defensive specialist index, with reliability shrinkage on the final score.
    max_matches = max(int(ratings["match_count"].max()), 1)
    ratings["sample_reliability"] = np.sqrt(ratings["match_count"] / max_matches)

    ratings["defensive_specialist_index"] = (
        0.50 * zscore(ratings["ridge_defense"])
        + 0.30 * zscore(ratings["suppression_rating_pct"])
        + 0.20 * zscore(ratings["avg_match_suppression_pct"])
    )
    ratings["defensive_specialist_index_shrunk"] = (
        ratings["defensive_specialist_index"] * ratings["sample_reliability"]
    )

    # Ranks
    ratings["rank_ridge_defense_best"] = ratings["ridge_defense"].rank(method="min", ascending=False)
    ratings["rank_suppression_abs_best"] = ratings["suppression_rating_abs"].rank(method="min", ascending=False)
    ratings["rank_suppression_pct_best"] = ratings["suppression_rating_pct"].rank(method="min", ascending=False)
    ratings["rank_avg_match_suppression_pct_best"] = ratings["avg_match_suppression_pct"].rank(method="min", ascending=False)
    ratings["rank_defensive_specialist_best"] = ratings["defensive_specialist_index_shrunk"].rank(method="min", ascending=False)

    return ratings.sort_values(
        ["rank_defensive_specialist_best", "match_count", "team_key"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

RANKED_COLUMNS = [
    "team_key",
    "match_count",
    "opr",
    "ridge_offense", "ridge_defense",
    "defensive_specialist_index_shrunk",
    "rank_ridge_defense_best",
    "rank_defensive_specialist_best",
    "suppression_rating_abs",
    "suppression_rating_pct",
    "avg_absolute_suppression",
    "aggregate_suppression_pct",
    "avg_match_suppression_pct",
    "rank_suppression_abs_best",
    "rank_suppression_pct_best",
    "rank_avg_match_suppression_pct_best",
    "suppression_samples",
]


def write_outputs(ratings: pd.DataFrame, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ratings_path = outdir / f"{stem}_team_ratings.csv"
    rankings_path = outdir / f"{stem}_team_ratings_ranked_defense.csv"

    ratings.to_csv(ratings_path, index=False)
    ratings[RANKED_COLUMNS].sort_values(
        ["rank_defensive_specialist_best", "match_count"],
        ascending=[True, False],
    ).to_csv(rankings_path, index=False)

    print(f"Wrote: {ratings_path}")
    print(f"Wrote: {rankings_path}")


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate defensive strength from FRC match data.")
    parser.add_argument("--matches-csv", help="Matches CSV. Optional if --refresh is given.")
    parser.add_argument("--outdir", default="frc_defense_output", help="Directory for output CSV files.")
    parser.add_argument("--ridge-alpha", type=float, default=10.0, help="Ridge regularization strength. Default: 10.0")
    parser.add_argument("--quals-only", action="store_true", help="Use qualification matches only.")
    parser.add_argument("--by-event", action="store_true", help="Also compute separate ratings per event.")
    parser.add_argument("--min-matches", type=int, default=1, help="Drop teams with fewer alliance appearances.")
    parser.add_argument(
        "--normalize-events",
        action="store_true",
        help="Rescale scores per event so all events have equal mean alliance score.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Pull fresh data from TBA before analyzing (uses --scope/--district-key/--year).",
    )
    parser.add_argument("--scope", choices=["fma", "all"], default="fma", help="TBA scope for --refresh.")
    parser.add_argument("--district-key", default="2026fma", help="District key for --refresh.")
    parser.add_argument("--year", type=int, default=2026, help="Season year for --refresh.")
    parser.add_argument("--tba-outdir", default="tba_output", help="Where --refresh writes its CSVs.")
    args = parser.parse_args()

    if args.refresh:
        import pull_frc_data
        try:
            matches_path = pull_frc_data.pull_to_csv(
                scope=args.scope, year=args.year, district_key=args.district_key,
                outdir=args.tba_outdir,
            )
        except RuntimeError as exc:
            print(f"Error refreshing TBA data: {exc}", file=sys.stderr)
            return 1
    elif args.matches_csv:
        matches_path = Path(args.matches_csv)
    else:
        parser.error("either --matches-csv or --refresh is required")

    outdir = Path(args.outdir)

    df = normalize_matches(pd.read_csv(matches_path), quals_only=args.quals_only)

    stem_parts = [safe_stem(matches_path)]
    if args.quals_only:
        stem_parts.append("quals")
    if args.normalize_events:
        df = rescale_scores_per_event(df)
        stem_parts.append("event_normalized")
    base_stem = "_".join(stem_parts)

    ratings = analyze_frame(df, ridge_alpha=args.ridge_alpha)
    ratings = ratings[
        (ratings["match_count"] >= args.min_matches)
        & (ratings["suppression_samples"] >= MIN_SUPPRESSION_SAMPLES)
    ]
    write_outputs(ratings, outdir, stem=base_stem)

    print()
    print("Top 25 likely defensive specialists:")
    preview = ratings[
        [
            "team_key",
            "suppression_rating_pct",
            "avg_match_suppression_pct",
            "aggregate_suppression_pct",
            "rank_suppression_pct_best",
            "rank_avg_match_suppression_pct_best",
            "rank_defensive_specialist_best",
        ]
    ].head(25)
    print(preview.to_string(index=False))

    if args.by_event:
        event_dir = outdir / f"{base_stem}_by_event"
        event_dir.mkdir(parents=True, exist_ok=True)
        for event_key, g in df.groupby("event_key"):
            try:
                event_ratings = analyze_frame(g, ridge_alpha=args.ridge_alpha)
                event_ratings = event_ratings[event_ratings["match_count"] >= args.min_matches]
                write_outputs(event_ratings, event_dir, stem=f"{base_stem}_{event_key}")
            except Exception as exc:
                print(f"Skipping event {event_key}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
