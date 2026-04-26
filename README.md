# FRC Defensive Ratings

Analyzes FRC match data and ranks teams by defensive impact. Built around 2026
Mid-Atlantic district data but works on any TBA-style match set.

## Scripts

- `pull_frc_data.py` — pulls events and matches from The Blue Alliance into
  `tba_output/`.
- `analyze_frc_defense.py` — reads a matches CSV and writes ratings/ranking
  CSVs into `frc_defense_output/`.
- `summarize_frc_dataset.py` — prints summary stats for a dataset and
  optionally for a single team.
- `defender_impact_analysis.py` — for each team in a target Worlds division,
  finds every 2026 match they played against a top-10 defender from any
  district and reports the score drop with the TBA match key for video
  review. See section below.

## Setup

```bash
export TBA_AUTH_KEY="your_tba_read_api_key"
pip install pandas numpy requests
```

## Command-line examples

Pull 2026 Mid-Atlantic district data (writes `tba_output/matches_2026fma.csv`):

```bash
python pull_frc_data.py --scope fma --district-key 2026fma
```

Pull every 2026 event (writes `tba_output/matches_2026_all.csv`):

```bash
python pull_frc_data.py --scope all --year 2026
```

Run the analysis on a previously pulled CSV, qualification matches only:

```bash
python analyze_frc_defense.py \
    --matches-csv tba_output/matches_2026fma.csv \
    --quals-only
```

Same, with per-event score normalization (recommended for cross-event runs):

```bash
python analyze_frc_defense.py \
    --matches-csv tba_output/matches_2026fma.csv \
    --quals-only \
    --normalize-events
```

One-shot pull-and-analyze (the typical workflow at an event):

```bash
python analyze_frc_defense.py \
    --refresh --scope fma --district-key 2026fma \
    --quals-only --normalize-events
```

Analyze every event separately into a per-event subdirectory:

```bash
python analyze_frc_defense.py \
    --matches-csv tba_output/matches_2026fma.csv \
    --quals-only --by-event
```

Summarize one team and cross-reference its ratings:

```bash
python summarize_frc_dataset.py \
    --matches-csv tba_output/matches_2026fma.csv \
    --team 1807 \
    --ratings-csv frc_defense_output/matches_2026fma_quals_event_normalized_team_ratings.csv
```

## Inputs

The analyzer requires these columns in the matches CSV (produced by
`pull_frc_data.py`):

- `event_key`
- `comp_level` (qm, ef, qf, sf, f)
- `red_score`, `blue_score`
- `red_team_keys`, `blue_team_keys` — comma-separated TBA keys, e.g.
  `frc1807,frc272,frc5895`

## Outputs

For input `tba_output/matches_2026fma.csv` with `--quals-only`, the analyzer
writes (filename grows when flags add suffixes like `_event_normalized`):

- `frc_defense_output/matches_2026fma_quals_team_ratings.csv` — every
  computed column for every team.
- `frc_defense_output/matches_2026fma_quals_team_ratings_ranked_defense.csv` —
  reduced column set sorted by `rank_defensive_specialist_best`.

## Models

### OPR (informational)

OLS estimate of per-team offensive contribution:

```
alliance_score ~= sum(team_offense)
```

Reported as `opr` for context. Not used in any defensive ranking.

### Ridge offense / defense

Joint ridge regression with separate offense and defense coefficients per team:

```
alliance_score ~= sum(offense[own_teams]) - sum(defense[opp_teams])
```

Solved as `beta = (X'X + alpha*I)^-1 X'y` with `alpha=10.0` by default.
Coefficients are reported as `ridge_offense` and `ridge_defense`. Higher
`ridge_defense` means the team's presence on the opposing alliance is
associated with lower scoring.

### Suppression

For each alliance score the code computes:

```
expected = sum(ridge_offense[scoring_alliance_teams])
actual   = alliance_score
suppression     = expected - actual
suppression_pct = clip(suppression / expected, MIN_SUPPRESSION_PCT, MAX_SUPPRESSION_PCT)
```

Rows with `expected <= MIN_EXPECTED_SCORE` are dropped to avoid division
instability. Each surviving row is attributed to the *opposing* alliance
(the defenders).

Two ridge regressions then attribute suppression to defenders:

```
suppression     ~= sum(defense_suppression_abs[defending_teams])  -> suppression_rating_abs
suppression_pct ~= sum(defense_suppression_pct[defending_teams])  -> suppression_rating_pct
```

The code also produces per-team descriptive stats by aggregating the same
suppression rows:

- `suppression_samples`
- `expected_score_against_total`, `actual_score_against_total`
- `avg_expected_score_against`, `avg_actual_score_against`
- `avg_absolute_suppression`, `aggregate_suppression_pct`
- `avg_match_suppression_pct` (median across the team's suppression rows)

### Cleaning constants

Defined at the top of `analyze_frc_defense.py`:

| Constant | Value | Effect |
|---|---|---|
| `MIN_EXPECTED_SCORE` | 50 | Drop suppression rows with weak expected score |
| `MAX_SUPPRESSION_PCT` | 0.75 | Clamp positive suppression percentage |
| `MIN_SUPPRESSION_PCT` | -0.50 | Clamp negative suppression percentage |
| `MIN_TOTAL_MATCH_SCORE` | 50 | Drop matches with combined score below this |
| `MIN_SUPPRESSION_SAMPLES` | 5 | Drop teams with fewer suppression rows from final outputs |

### Reliability shrinkage

One factor in `[0, 1]` applied to the final composite, to pull down ratings
from teams with few matches:

```
sample_reliability                = sqrt(match_count / max(match_count))
defensive_specialist_index_shrunk = defensive_specialist_index * sample_reliability
```

### Composite

```
defensive_specialist_index =
      0.50 * zscore(ridge_defense)
    + 0.30 * zscore(suppression_rating_pct)
    + 0.20 * zscore(avg_match_suppression_pct)
```

The shrunk version `defensive_specialist_index_shrunk` is what
`rank_defensive_specialist_best` is sorted by.

## Rank columns

| Column | Best direction |
|---|---|
| `rank_ridge_defense_best` | higher `ridge_defense` |
| `rank_suppression_abs_best` | higher `suppression_rating_abs` |
| `rank_suppression_pct_best` | higher `suppression_rating_pct` |
| `rank_avg_match_suppression_pct_best` | higher `avg_match_suppression_pct` |
| `rank_defensive_specialist_best` | higher `defensive_specialist_index_shrunk` |

## Optional analysis flags

- `--quals-only` — restricts to `comp_level == "qm"`.
- `--by-event` — also writes a per-event ratings folder.
- `--normalize-events` — multiplies each match's `red_score` and `blue_score`
  by `(global_mean_alliance_score / event_mean_alliance_score)` so all events
  carry equal weight in cross-event regressions.
- `--refresh [--scope fma|all] [--district-key KEY] [--year YEAR] [--tba-outdir DIR]` —
  calls `pull_frc_data.pull_to_csv(...)` first and uses the resulting CSV.
  Requires `TBA_AUTH_KEY`. With `--refresh`, `--matches-csv` is optional.
- `--ridge-alpha FLOAT` — ridge regularization strength, default `10.0`.
- `--min-matches INT` — drop teams with fewer alliance appearances, default `1`.

## Defender impact analysis (Worlds division prep)

`defender_impact_analysis.py` is purpose-built for "which top defenders shut
down our division's top scorers, and which specific matches should the kids
go watch?"

```bash
export TBA_AUTH_KEY="..."
python defender_impact_analysis.py                          # uses cached district CSVs if present
python defender_impact_analysis.py --refresh                # re-pull all districts
python defender_impact_analysis.py --newton-event-key 2026new --top-defenders-per-district 10
```

What it does:

1. Pulls all 2026 districts (one CSV per district under `tba_output/`).
2. Pulls all 2026 Regional events as a single pool (`matches_2026regionals.csv`)
   so teams from non-district areas (CA outside the CA district, FL, TX, MN,
   Brazil, etc.) also get analyzed.
3. Runs `analyze_frc_defense` per district and on the regional pool
   (quals-only, event-normalized) and takes the top N defenders by
   `defensive_specialist_index_shrunk` from each.
4. Pulls the team list for `--newton-event-key` (default `2026new`).
5. Maps each Newton team to whichever pool they played in most (a specific
   district, or the regional pool labeled `2026regionals`).
6. For every match a Newton team played in that pool (quals + elims), records
   any encounter where the opposing alliance contained a top-N defender from
   the same pool, including expected score (from the pool's `ridge_offense`),
   actual score, suppression, and TBA match key.

The script assumes there are no cross-pool encounters: each Newton team is only
checked against the top defenders from their own home pool. In 2026 this is
effectively always true — district teams almost never play across districts,
and the regional pool is its own world.

Outputs in `frc_defense_output/`:

- `newton_high_scorers_summary.csv` — one row per Newton team with
  `home_district` (district key, or `2026regionals` for regional teams),
  `ridge_offense`, `avg_score`, count of encounters with top defenders,
  average suppression, biggest single suppression match.
- `newton_defender_encounters.csv` — one row per (Newton team × match against
  a top defender), sorted by suppression descending, with `match_key`,
  `tba_url`, defender(s), expected vs actual.
- `newton_roster_defenders.csv` — Newton-roster teams ranked by their home-
  pool defensive metrics. Forward-looking: who you'll actually face on
  the Newton field, with their pre-Worlds defensive pedigree.

### Newton teams without district matches

Most Newton teams come from one of the 14 official 2026 districts. The
remainder are tagged in `home_district` as `2026regionals` (analyzed against
the all-regionals pool) or left blank (no 2026 match data on TBA at all).

Regional-only Newton teams (`home_district == 2026regionals`):

```
frc180   frc233   frc386   frc424   frc695   frc1108  frc1796  frc1884
frc1902  frc2052  frc2783  frc2996  frc3044  frc3276  frc3354  frc3966
frc4253  frc4400  frc5736  frc5948  frc6352  frc6436  frc6647  frc6988
frc9029  frc9067  frc10291 frc10903
```

Newton teams with no TBA match data yet (blank `home_district`):

```
frc1577  frc4590  frc5951  frc10935
```

These four are all from the Israel district (`2026isr`), whose match results
have not been posted to TBA at the time of writing. Re-run with `--refresh`
once Israeli matches appear and they will populate.


## Limitations

- Defense is inferred from aggregate scoring, not observed robot behavior.
- Partner effects can mask individual defensive impact.
- Selective defense (only deployed against certain opponents) is diluted
  across all that team's matches.
- Sparse FRC schedules make estimates noisy; trust ranks more after several
  rounds of quals.
- Weak opponents can inflate percentage suppression; the cleaning constants
  reduce but do not eliminate this.
- Video and human scouting are still required to validate any candidate.
