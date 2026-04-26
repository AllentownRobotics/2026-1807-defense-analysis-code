#!/usr/bin/env python3
"""
defender_impact_analysis.py

For each Newton-division team, find every 2026 match where they faced one of
the top-N defenders from any 2026 district, and compute how much their score
dropped vs. the team's expected score (from their home district's ridge model).

Outputs into frc_defense_output/:
  - newton_high_scorers_summary.csv  one row per Newton team
  - newton_defender_encounters.csv   one row per match against a top defender
  - newton_roster_defenders.csv      Newton-roster teams ranked as defenders

Usage:
  export TBA_AUTH_KEY="..."
  python defender_impact_analysis.py                        # uses cached district CSVs if present
  python defender_impact_analysis.py --refresh              # re-pull from TBA
  python defender_impact_analysis.py --top-defenders-per-district 15
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import analyze_frc_defense as afd
import pull_frc_data as pfd


YEAR = 2026
NEWTON_EVENT_KEY = "2026new"
TBA_BASE = "https://www.thebluealliance.com/api/v3"
TBA_OUTDIR = Path("tba_output")
DEFAULT_OUTDIR = Path("frc_defense_output")
RIDGE_ALPHA = 10.0
REGIONAL_KEY = "2026regionals"  # synthetic "district" for the regional event pool


# ---------------------------------------------------------------------------
# TBA helpers
# ---------------------------------------------------------------------------

def _tba_get(path: str):
    key = os.environ.get("TBA_AUTH_KEY", "").strip()
    if not key:
        raise RuntimeError('TBA_AUTH_KEY is not set. export TBA_AUTH_KEY="..."')
    resp = requests.get(
        f"{TBA_BASE}{path}",
        headers={"X-TBA-Auth-Key": key, "Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_district_keys(year: int) -> list[str]:
    return [d["key"] for d in _tba_get(f"/districts/{year}")]


def get_event_team_keys(event_key: str) -> list[str]:
    return _tba_get(f"/event/{event_key}/teams/keys")


# ---------------------------------------------------------------------------
# District pull + analysis (cached)
# ---------------------------------------------------------------------------

def pull_district_matches(district_key: str, refresh: bool) -> Path:
    """Return cached matches CSV path for a district, pulling from TBA if needed."""
    matches_csv = TBA_OUTDIR / f"matches_{district_key}.csv"
    if matches_csv.exists() and not refresh:
        return matches_csv
    return pfd.pull_to_csv(
        scope="fma",  # "fma" scope means "district mode" in pull_frc_data
        year=YEAR,
        district_key=district_key,
        outdir=TBA_OUTDIR,
        verbose=False,
    )


def pull_regional_matches(refresh: bool) -> Path:
    """Return cached matches CSV path for the all-regionals pool."""
    matches_csv = TBA_OUTDIR / f"matches_{REGIONAL_KEY}.csv"
    if matches_csv.exists() and not refresh:
        return matches_csv
    return pfd.pull_to_csv(
        scope="regionals",
        year=YEAR,
        outdir=TBA_OUTDIR,
        verbose=False,
    )


def _analyze_matches_csv(
    matches_csv: Path, label: str, top_n: int
) -> dict | None:
    """Run the standard quals-only event-normalized analysis on a matches CSV."""
    raw = pd.read_csv(matches_csv)

    quals = afd.normalize_matches(raw, quals_only=True)
    if quals.empty:
        return None
    quals = afd.rescale_scores_per_event(quals)

    ratings = afd.analyze_frame(quals, ridge_alpha=RIDGE_ALPHA)
    eligible = ratings[ratings["suppression_samples"] >= afd.MIN_SUPPRESSION_SAMPLES]
    if eligible.empty:
        return None
    top = (
        eligible.sort_values("rank_defensive_specialist_best")
        .head(top_n)
        .copy()
    )
    top["district"] = label

    ridge_off = ratings.set_index("team_key")["ridge_offense"]
    all_matches = afd.normalize_matches(raw, quals_only=False)

    return {
        "district": label,
        "top_defenders": top,
        "top_defender_set": set(top["team_key"]),
        "ridge_off": ridge_off,
        "ratings": ratings.set_index("team_key"),
        "all_matches": all_matches,
    }


def analyze_district(
    district_key: str, top_n: int, refresh: bool
) -> dict | None:
    matches_csv = pull_district_matches(district_key, refresh=refresh)
    return _analyze_matches_csv(matches_csv, district_key, top_n)


def analyze_regionals(top_n: int, refresh: bool) -> dict | None:
    matches_csv = pull_regional_matches(refresh=refresh)
    return _analyze_matches_csv(matches_csv, REGIONAL_KEY, top_n)


# ---------------------------------------------------------------------------
# Cross-district team mapping
# ---------------------------------------------------------------------------

def map_teams_to_home_district(
    district_data: dict[str, dict]
) -> dict[str, str]:
    """For every team appearing in any district's matches, pick the district they played in most."""
    counts: dict[tuple[str, str], int] = {}
    for dk, d in district_data.items():
        team_iter: list[str] = []
        for lst in d["all_matches"]["red_teams"]:
            team_iter.extend(lst)
        for lst in d["all_matches"]["blue_teams"]:
            team_iter.extend(lst)
        for team, n in Counter(team_iter).items():
            counts[(team, dk)] = n

    home: dict[str, tuple[str, int]] = {}
    for (team, dk), n in counts.items():
        if team not in home or n > home[team][1]:
            home[team] = (dk, n)
    return {t: dk for t, (dk, _) in home.items()}


# ---------------------------------------------------------------------------
# Encounter detection
# ---------------------------------------------------------------------------

def find_encounters_for_team(
    team_key: str,
    matches: pd.DataFrame,
    top_defender_set: set[str],
    ridge_off: pd.Series,
) -> list[dict]:
    """Find matches in `matches` where `team_key` faced any team in
    `top_defender_set`. Caller is expected to pass the team's home pool's
    matches and that pool's top defenders (we assume no cross-pool encounters)."""
    rows = []
    for _, m in matches.iterrows():
        red, blue = m["red_teams"], m["blue_teams"]
        if team_key in red:
            own, opp = red, blue
            own_score, opp_score = float(m["red_score"]), float(m["blue_score"])
        elif team_key in blue:
            own, opp = blue, red
            own_score, opp_score = float(m["blue_score"]), float(m["red_score"])
        else:
            continue

        defenders_present = [t for t in opp if t in top_defender_set]
        if not defenders_present:
            continue

        expected = float(ridge_off.reindex(own).fillna(0).sum())
        if expected <= 0:
            continue

        rows.append({
            "newton_team": team_key,
            "match_key": m.get("match_key", ""),
            "event_key": m["event_key"],
            "comp_level": m["comp_level"],
            "match_number": m.get("match_number", ""),
            "newton_alliance": ",".join(own),
            "opposing_alliance": ",".join(opp),
            "top_defenders_in_opposing": ",".join(defenders_present),
            "expected_score": round(expected, 1),
            "actual_score": own_score,
            "suppression": round(expected - own_score, 1),
            "suppression_pct": round((expected - own_score) / expected, 3),
            "tba_url": f"https://www.thebluealliance.com/match/{m.get('match_key', '')}",
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="Re-pull district data from TBA even if cached.")
    parser.add_argument("--top-defenders-per-district", type=int, default=10,
                        help="Number of top defenders to pull from each district. Default: 10")
    parser.add_argument("--newton-event-key", default=NEWTON_EVENT_KEY,
                        help=f"TBA event key for the division. Default: {NEWTON_EVENT_KEY}")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                        help="Output directory.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    TBA_OUTDIR.mkdir(parents=True, exist_ok=True)

    try:
        district_keys = get_district_keys(YEAR)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Found {len(district_keys)} districts in {YEAR}: {district_keys}")

    district_data: dict[str, dict] = {}
    for dk in district_keys:
        print(f"\n[{dk}]")
        try:
            d = analyze_district(dk, args.top_defenders_per_district, refresh=args.refresh)
        except Exception as exc:
            print(f"  skipped: {exc}")
            continue
        if d is None:
            print("  no usable data")
            continue
        district_data[dk] = d
        print(f"  top {len(d['top_defenders'])} defenders: "
              f"{d['top_defenders']['team_key'].tolist()}")

    print(f"\n[{REGIONAL_KEY}]  (all 2026 Regional events as one pool)")
    try:
        d = analyze_regionals(args.top_defenders_per_district, refresh=args.refresh)
    except Exception as exc:
        print(f"  skipped: {exc}")
        d = None
    if d is None:
        print("  no usable data")
    else:
        district_data[REGIONAL_KEY] = d
        print(f"  top {len(d['top_defenders'])} defenders: "
              f"{d['top_defenders']['team_key'].tolist()}")

    if not district_data:
        print("No district or regional data; aborting.", file=sys.stderr)
        return 1

    print(f"\nFetching Newton roster from {args.newton_event_key} ...")
    try:
        newton_teams = get_event_team_keys(args.newton_event_key)
    except requests.HTTPError as exc:
        print(f"  failed to fetch {args.newton_event_key}: {exc}", file=sys.stderr)
        return 1
    print(f"  {len(newton_teams)} teams in {args.newton_event_key}")

    home_district = map_teams_to_home_district(district_data)

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    for team in newton_teams:
        hd = home_district.get(team)
        if hd is None:
            summary_rows.append({
                "newton_team": team, "home_district": "",
                "match_count_in_district": 0,
                "ridge_offense": np.nan, "avg_score": np.nan,
                "encounters_with_top_defenders": 0,
                "avg_suppression": np.nan, "avg_suppression_pct": np.nan,
                "biggest_suppression": np.nan,
                "biggest_suppression_match_key": "",
                "biggest_suppression_defender": "",
            })
            continue

        d = district_data[hd]
        encounters = find_encounters_for_team(
            team, d["all_matches"], d["top_defender_set"], d["ridge_off"]
        )
        detail_rows.extend(encounters)

        all_m = d["all_matches"]
        played = all_m[
            all_m["red_teams"].apply(lambda lst: team in lst)
            | all_m["blue_teams"].apply(lambda lst: team in lst)
        ]
        if not played.empty:
            own_scores = played.apply(
                lambda m: float(m["red_score"]) if team in m["red_teams"]
                else float(m["blue_score"]),
                axis=1,
            )
            avg_score = float(own_scores.mean())
        else:
            avg_score = np.nan

        ridge_off_val = (
            float(d["ratings"].loc[team, "ridge_offense"])
            if team in d["ratings"].index else np.nan
        )

        if encounters:
            edf = pd.DataFrame(encounters)
            biggest = edf.loc[edf["suppression"].idxmax()]
            summary_rows.append({
                "newton_team": team, "home_district": hd,
                "match_count_in_district": int(len(played)),
                "ridge_offense": ridge_off_val, "avg_score": avg_score,
                "encounters_with_top_defenders": int(len(edf)),
                "avg_suppression": round(float(edf["suppression"].mean()), 1),
                "avg_suppression_pct": round(float(edf["suppression_pct"].mean()), 3),
                "biggest_suppression": float(biggest["suppression"]),
                "biggest_suppression_match_key": biggest["match_key"],
                "biggest_suppression_defender": biggest["top_defenders_in_opposing"],
            })
        else:
            summary_rows.append({
                "newton_team": team, "home_district": hd,
                "match_count_in_district": int(len(played)),
                "ridge_offense": ridge_off_val, "avg_score": avg_score,
                "encounters_with_top_defenders": 0,
                "avg_suppression": np.nan, "avg_suppression_pct": np.nan,
                "biggest_suppression": np.nan,
                "biggest_suppression_match_key": "",
                "biggest_suppression_defender": "",
            })

    summary = pd.DataFrame(summary_rows).sort_values(
        "ridge_offense", ascending=False, na_position="last",
    )
    detail = (
        pd.DataFrame(detail_rows).sort_values("suppression", ascending=False)
        if detail_rows else pd.DataFrame()
    )

    # Newton-roster defenders: rank Newton teams by their home-district defensive metric
    newton_def_rows = []
    for team in newton_teams:
        hd = home_district.get(team)
        if hd is None or team not in district_data[hd]["ratings"].index:
            continue
        r = district_data[hd]["ratings"].loc[team]
        newton_def_rows.append({
            "newton_team": team, "home_district": hd,
            "ridge_defense": float(r["ridge_defense"]),
            "defensive_specialist_index_shrunk": float(r["defensive_specialist_index_shrunk"]),
            "rank_in_district": int(r["rank_defensive_specialist_best"]),
            "suppression_rating_pct": float(r["suppression_rating_pct"]),
            "suppression_samples": int(r["suppression_samples"]),
            "is_top_defender_in_district": team in district_data[hd]["top_defender_set"],
        })
    newton_defenders = pd.DataFrame(newton_def_rows).sort_values(
        "defensive_specialist_index_shrunk", ascending=False,
    )

    summary_path = outdir / "newton_high_scorers_summary.csv"
    detail_path = outdir / "newton_defender_encounters.csv"
    roster_def_path = outdir / "newton_roster_defenders.csv"
    summary.to_csv(summary_path, index=False)
    detail.to_csv(detail_path, index=False)
    newton_defenders.to_csv(roster_def_path, index=False)

    print()
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {detail_path}  ({len(detail)} encounters)")
    print(f"Wrote: {roster_def_path}")
    print()
    print("Top 10 Newton high scorers and their suppression history:")
    preview_cols = [
        "newton_team", "home_district", "ridge_offense",
        "encounters_with_top_defenders", "avg_suppression",
        "biggest_suppression", "biggest_suppression_match_key",
        "biggest_suppression_defender",
    ]
    print(summary[preview_cols].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
