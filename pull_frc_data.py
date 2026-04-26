#!/usr/bin/env python3
"""
Pull 2026 FRC match data from The Blue Alliance (TBA) API.

Modes:
  1) Mid-Atlantic district only
  2) All 2026 events

Outputs:
  - events CSV
  - matches CSV
  - optional raw JSON files

Setup:
  export TBA_AUTH_KEY="your_key_here"
  python pull_tba_2026_matches.py --scope fma
  python pull_tba_2026_matches.py --scope all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://www.thebluealliance.com/api/v3"
DEFAULT_SLEEP_SECONDS = 0.10


class TBAClient:
    def __init__(self, auth_key: str, user_agent: str = "frc-data-pull/1.0") -> None:
        if not auth_key:
            raise ValueError("Missing TBA auth key. Set TBA_AUTH_KEY in your environment.")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-TBA-Auth-Key": auth_key,
                "User-Agent": user_agent,
                "Accept": "application/json",
            }
        )

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{BASE_URL}{path}"
        resp = self.session.get(url, params=params, timeout=30)
        if resp.status_code == 401:
            raise RuntimeError(
                "401 Unauthorized from TBA. Check your X-TBA-Auth-Key / TBA_AUTH_KEY."
            )
        resp.raise_for_status()
        return resp.json()

    def get_events_for_year(self, year: int) -> list[dict[str, Any]]:
        return self.get_json(f"/events/{year}")

    def get_event_keys_for_year(self, year: int) -> list[str]:
        return self.get_json(f"/events/{year}/keys")

    def get_district_events(self, district_key: str) -> list[dict[str, Any]]:
        return self.get_json(f"/district/{district_key}/events")

    def get_district_event_keys(self, district_key: str) -> list[str]:
        return self.get_json(f"/district/{district_key}/events/keys")

    def get_event_matches(self, event_key: str) -> list[dict[str, Any]]:
        return self.get_json(f"/event/{event_key}/matches")


def flatten_match(match: dict[str, Any]) -> dict[str, Any]:
    """Turn one TBA match object into a flatter row for CSV export."""
    alliances = match.get("alliances", {})
    red = alliances.get("red", {})
    blue = alliances.get("blue", {})
    score_breakdown = match.get("score_breakdown", {}) or {}

    row = {
        "match_key": match.get("key"),
        "event_key": match.get("event_key"),
        "comp_level": match.get("comp_level"),      # qm, qf, sf, f
        "set_number": match.get("set_number"),
        "match_number": match.get("match_number"),
        "winning_alliance": match.get("winning_alliance"),
        "time": match.get("time"),
        "predicted_time": match.get("predicted_time"),
        "actual_time": match.get("actual_time"),
        "post_result_time": match.get("post_result_time"),

        "red_score": red.get("score"),
        "red_team_keys": ",".join(red.get("team_keys", [])),
        "red_surrogates": ",".join(red.get("surrogate_team_keys", [])),
        "red_dqs": ",".join(red.get("dq_team_keys", [])),

        "blue_score": blue.get("score"),
        "blue_team_keys": ",".join(blue.get("team_keys", [])),
        "blue_surrogates": ",".join(blue.get("surrogate_team_keys", [])),
        "blue_dqs": ",".join(blue.get("dq_team_keys", [])),
    }

    # Common 2026-style breakdown handling:
    # keep raw nested JSON in case you want to inspect schema later
    row["score_breakdown_raw"] = json.dumps(score_breakdown, separators=(",", ":"))

    # Optional convenience: flatten top-level red/blue breakdown keys if present
    if isinstance(score_breakdown, dict):
        for alliance_name in ("red", "blue"):
            alliance_breakdown = score_breakdown.get(alliance_name)
            if isinstance(alliance_breakdown, dict):
                for k, v in alliance_breakdown.items():
                    row[f"{alliance_name}_breakdown__{k}"] = (
                        json.dumps(v, separators=(",", ":"))
                        if isinstance(v, (dict, list))
                        else v
                    )

    return row


def save_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pull_matches(
    client: TBAClient,
    event_keys: list[str],
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_matches_raw: list[dict[str, Any]] = []
    flattened_rows: list[dict[str, Any]] = []

    total = len(event_keys)
    for idx, event_key in enumerate(event_keys, start=1):
        if verbose:
            print(f"[{idx}/{total}] Pulling matches for {event_key} ...", flush=True)
        matches = client.get_event_matches(event_key)
        all_matches_raw.extend(matches)

        for match in matches:
            flattened_rows.append(flatten_match(match))

        time.sleep(sleep_seconds)

    return all_matches_raw, flattened_rows


def pull_to_csv(
    scope: str = "fma",
    year: int = 2026,
    district_key: str = "2026fma",
    outdir: Path | str = "tba_output",
    save_raw_json: bool = False,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    verbose: bool = True,
) -> Path:
    """Pull events+matches from TBA and write CSVs. Returns the matches CSV path."""
    auth_key = os.getenv("TBA_AUTH_KEY", "").strip()
    if not auth_key:
        raise RuntimeError(
            "TBA_AUTH_KEY is not set. Run: export TBA_AUTH_KEY=\"your_key_here\""
        )

    outdir = Path(outdir)
    ensure_dir(outdir)
    client = TBAClient(auth_key=auth_key)

    if scope == "fma":
        events = client.get_district_events(district_key)
        scope_slug = district_key
    elif scope == "regionals":
        all_events = client.get_events_for_year(year)
        events = [e for e in all_events if e.get("event_type") == 0]
        scope_slug = f"{year}regionals"
    else:
        events = client.get_events_for_year(year)
        scope_slug = f"{year}_all"
    event_keys = [e["key"] for e in events]

    if not event_keys:
        raise RuntimeError(f"No events found for scope={scope}")
    if verbose:
        print(f"Found {len(event_keys)} events for scope={scope}")

    events_df = pd.json_normalize(events)
    events_csv = outdir / f"events_{scope_slug}.csv"
    events_df.to_csv(events_csv, index=False)
    if save_raw_json:
        save_json(events, outdir / f"events_{scope_slug}.json")

    raw_matches, flat_rows = pull_matches(
        client=client, event_keys=event_keys,
        sleep_seconds=sleep_seconds, verbose=verbose,
    )

    matches_df = pd.DataFrame(flat_rows)
    matches_csv = outdir / f"matches_{scope_slug}.csv"
    matches_df.to_csv(matches_csv, index=False)
    if save_raw_json:
        save_json(raw_matches, outdir / f"matches_{scope_slug}.json")

    summary = (
        matches_df.groupby("event_key", dropna=False).size()
        .reset_index(name="match_count")
        .sort_values(["match_count", "event_key"], ascending=[False, True])
    )
    summary.to_csv(outdir / f"match_counts_by_event_{scope_slug}.csv", index=False)

    if verbose:
        print(f"Events CSV:  {events_csv}")
        print(f"Matches CSV: {matches_csv}")
        print(f"Event count: {len(events_df)}  Match count: {len(matches_df)}")
    return matches_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull 2026 FRC matches from TBA")
    parser.add_argument("--scope", choices=["fma", "all", "regionals"], default="fma")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--district-key", default="2026fma")
    parser.add_argument("--outdir", default="tba_output")
    parser.add_argument("--save-raw-json", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    args = parser.parse_args()

    try:
        pull_to_csv(
            scope=args.scope, year=args.year, district_key=args.district_key,
            outdir=args.outdir, save_raw_json=args.save_raw_json,
            sleep_seconds=args.sleep_seconds,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())