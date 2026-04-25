#!/usr/bin/env python3
"""
Summarize an FRC matches dataset and optionally a specific team's performance.

Usage:
  python summarize_frc_dataset.py --matches-csv tba_output/matches_2026fma.csv

Team-specific:
  python summarize_frc_dataset.py \
    --matches-csv tba_output/matches_2026fma.csv \
    --team 1807

With defensive ratings:
  python summarize_frc_dataset.py \
    --matches-csv tba_output/matches_2026fma.csv \
    --team 1807 \
    --ratings-csv frc_defense_output/matches_2026fma_quals_team_ratings.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def parse_team_list(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def normalize_team_key(team):
    team = str(team).strip().lower()
    if team.startswith("frc"):
        return team
    return "frc" + team


def summarize_dataset(df, path):
    num_matches = len(df)
    num_events = df["event_key"].nunique()

    all_teams = set()
    for teams in df["red_teams"]:
        all_teams.update(teams)
    for teams in df["blue_teams"]:
        all_teams.update(teams)

    print("\n=== Dataset Summary ===")
    print(f"File: {path}")
    print(f"Matches: {num_matches}")
    print(f"Events: {num_events}")
    print(f"Unique teams: {len(all_teams)}")

    print("\n--- Matches by comp_level ---")
    print(df["comp_level"].value_counts().sort_index())

    print("\n--- Matches per event (top 10) ---")
    print(df.groupby("event_key").size().sort_values(ascending=False).head(10))

    teams_per_event = []
    for event_key, g in df.groupby("event_key"):
        teams = set()
        for t in g["red_teams"]:
            teams.update(t)
        for t in g["blue_teams"]:
            teams.update(t)
        teams_per_event.append(len(teams))

    if teams_per_event:
        print("\n--- Teams per event ---")
        print(f"Average teams/event: {sum(teams_per_event)/len(teams_per_event):.1f}")
        print(f"Min teams/event: {min(teams_per_event)}")
        print(f"Max teams/event: {max(teams_per_event)}")


def get_team_matches(df, team_key):
    rows = []
    for _, m in df.iterrows():
        red = m["red_teams"]
        blue = m["blue_teams"]

        if team_key in red:
            alliance = "red"
            partners = [t for t in red if t != team_key]
            opponents = blue
            own_score = m["red_score"]
            opp_score = m["blue_score"]
        elif team_key in blue:
            alliance = "blue"
            partners = [t for t in blue if t != team_key]
            opponents = red
            own_score = m["blue_score"]
            opp_score = m["red_score"]
        else:
            continue

        margin = own_score - opp_score

        rows.append({
            "event_key": m["event_key"],
            "match_key": m.get("match_key", ""),
            "comp_level": m["comp_level"],
            "set_number": m.get("set_number", ""),
            "match_number": m.get("match_number", ""),
            "alliance": alliance,
            "partners": ",".join(partners),
            "opponents": ",".join(opponents),
            "own_score": own_score,
            "opp_score": opp_score,
            "margin": margin,
            "win": margin > 0,
            "tie": margin == 0,
        })

    return pd.DataFrame(rows)


def summarize_team_raw(df, team_key):
    team_df = get_team_matches(df, team_key)

    if team_df.empty:
        print(f"\nNo matches found for {team_key}.")
        return

    wins = int(team_df["win"].sum())
    ties = int(team_df["tie"].sum())
    losses = len(team_df) - wins - ties

    print(f"\n=== Team Summary: {team_key} ===")
    print(f"Matches: {len(team_df)}")
    print(f"Events: {team_df['event_key'].nunique()}")
    print(f"Record: {wins}-{losses}-{ties}")
    print(f"Average own alliance score: {team_df['own_score'].mean():.2f}")
    print(f"Average opponent score allowed: {team_df['opp_score'].mean():.2f}")
    print(f"Average margin: {team_df['margin'].mean():.2f}")
    print(f"Median margin: {team_df['margin'].median():.2f}")

    print("\n--- By event ---")
    event_summary = (
        team_df.groupby("event_key")
        .agg({
            "match_key": "count",
            "win": "sum",
            "own_score": "mean",
            "opp_score": "mean",
            "margin": "mean",
        })
        .rename(columns={
            "match_key": "matches",
            "win": "wins",
            "own_score": "avg_own_score",
            "opp_score": "avg_opp_score",
            "margin": "avg_margin",
        })
    )
    event_summary["losses_or_ties"] = event_summary["matches"] - event_summary["wins"]
    print(event_summary.sort_index())

    print("\n--- Opponent scores in this team's matches ---")
    print(team_df["opp_score"].describe())

    print("\n--- Best score-suppression games: lowest opponent scores ---")
    cols = [
        "event_key", "comp_level", "match_number",
        "alliance", "partners", "opponents",
        "own_score", "opp_score", "margin",
    ]
    print(team_df.sort_values(["opp_score", "margin"], ascending=[True, False])[cols].head(10))

    print("\n--- Worst opponent-scoring games: highest opponent scores ---")
    print(team_df.sort_values(["opp_score", "margin"], ascending=[False, True])[cols].head(10))


def summarize_team_ratings(ratings_csv, team_key):
    path = Path(ratings_csv)
    ratings = pd.read_csv(path)

    if "team_key" not in ratings.columns:
        raise ValueError("Ratings CSV must include a team_key column.")

    row = ratings[ratings["team_key"] == team_key]
    if row.empty:
        print(f"\nNo ratings row found for {team_key} in {path}.")
        return

    row = row.iloc[0]
    n = len(ratings)

    print(f"\n=== Defensive Ratings: {team_key} ===")
    print(f"Ratings file: {path}")

    fields = [
        "match_count",
        "opr",
        "ridge_offense",
        "ridge_defense",
        "suppression_rating_pct",
        "avg_match_suppression_pct",
        "defensive_specialist_index_shrunk",
        "rank_ridge_defense_best",
        "rank_suppression_pct_best",
        "rank_avg_match_suppression_pct_best",
        "rank_defensive_specialist_best",
    ]

    for f in fields:
        if f in ratings.columns:
            print(f"{f}: {row[f]}")

    def rank_line(metric, ascending, label):
        if metric not in ratings.columns:
            return
        rank = ratings[metric].rank(method="min", ascending=ascending)[row.name]
        pct = 100.0 * (1.0 - ((rank - 1.0) / max(n - 1.0, 1.0)))
        direction = "lower is better" if ascending else "higher is better"
        print(f"{label}: rank {int(rank)} / {n}; percentile ≈ {pct:.1f}; {direction}")

    print("\n--- Rank interpretation ---")
    rank_line("ridge_defense", False, "Ridge defense")
    rank_line("suppression_rating_pct", False, "Suppression rating (pct)")
    rank_line("avg_match_suppression_pct", False, "Avg match suppression pct")
    rank_line("defensive_specialist_index_shrunk", False, "Composite defensive specialist index")

    print("\nInterpretation note:")
    print(
        "A team can be widely regarded as a strong defender but still rank lower if "
        "the model sees limited suppression in raw opponent scores, if the team often "
        "faces unusually strong scorers, if defense is used selectively, or if its "
        "partners/opponents create schedule artifacts. Treat this as a candidate list "
        "for video/scouting validation, not a definitive scouting verdict."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches-csv", required=True)
    parser.add_argument("--team", default=None, help="Team number or key, e.g. 1807 or frc1807")
    parser.add_argument("--ratings-csv", default=None, help="Optional ratings CSV from analyze_frc_defense.py")
    args = parser.parse_args()

    path = Path(args.matches_csv)
    df = pd.read_csv(path)

    required_cols = {
        "event_key",
        "comp_level",
        "red_team_keys",
        "blue_team_keys",
        "red_score",
        "blue_score",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["red_score"] = pd.to_numeric(df["red_score"], errors="coerce")
    df["blue_score"] = pd.to_numeric(df["blue_score"], errors="coerce")
    df = df.dropna(subset=["red_score", "blue_score"]).copy()

    df["red_teams"] = df["red_team_keys"].apply(parse_team_list)
    df["blue_teams"] = df["blue_team_keys"].apply(parse_team_list)

    summarize_dataset(df, path)

    if args.team:
        team_key = normalize_team_key(args.team)
        summarize_team_raw(df, team_key)

        if args.ratings_csv:
            summarize_team_ratings(args.ratings_csv, team_key)

    print("\nDone.\n")


if __name__ == "__main__":
    main()