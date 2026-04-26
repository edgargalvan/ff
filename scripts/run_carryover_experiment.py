"""Experiment 2: Multi-season carryover.

After fitting season N's full schedule, extract per-team posterior means for
attack and defense. Pass those means as informed priors for season N+1's
rolling backtest. Compare against `base` (cold-start each season).

Chain:
  2022 cold-start -> extract priors -> 2023 with priors -> extract -> 2024 with priors -> ...

Comparison: 2023, 2024, 2025 metrics with vs without carryover (2022 is
identical for both since it's the chain origin).

Outputs to results/raw/carryover_{variant}_{...} and results/carryover_summary.csv.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import backtest, summarize_backtest
from src.data import load_game_data
from src.model import bhm

SEASONS = [2022, 2023, 2024, 2025]
TRAIN_WINDOW = 8
SAMPLES = 500
NSIMS = 500
CARRYOVER_SD = 0.1   # how much teams change year-to-year, in log-rate units

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"


def fit_season_full(season: int, team_priors: dict | None = None) -> tuple:
    """Fit the model on full-season data and return (idata, teams, df)."""
    df, teams = load_game_data(season)
    idata = bhm(df, metric="score", samples=SAMPLES, team_priors=team_priors)
    return idata, teams, df


def extract_posterior_means(idata, teams: pd.DataFrame) -> dict:
    """Extract per-team posterior means in i-aligned order, ready to pass as
    team_priors to the next season's bhm() call."""
    atts_mean = idata.posterior["atts"].mean(dim=["chain", "draw"]).values
    defs_mean = idata.posterior["defs"].mean(dim=["chain", "draw"]).values
    return {
        "atts_mean": np.asarray(atts_mean, dtype=float),
        "defs_mean": np.asarray(defs_mean, dtype=float),
        "carryover_sd": CARRYOVER_SD,
        "teams_index": teams.copy(),  # for cross-checking team alignment
    }


def align_priors_to_season(priors: dict, season_teams: pd.DataFrame) -> dict | None:
    """Given priors from season N (with their team index) and season N+1's
    team index, return priors aligned to N+1's order. Returns None if
    alignment is impossible (no overlap)."""
    src_teams = priors["teams_index"]
    # Map src team -> src index, then build new arrays in season_teams order
    src_to_i = dict(zip(src_teams["team"], src_teams["i"]))
    n_dst = len(season_teams)
    new_atts = np.zeros(n_dst)
    new_defs = np.zeros(n_dst)
    matched = 0
    for _, row in season_teams.iterrows():
        team = row["team"]
        i_dst = int(row["i"])
        if team in src_to_i:
            i_src = src_to_i[team]
            new_atts[i_dst] = priors["atts_mean"][i_src]
            new_defs[i_dst] = priors["defs_mean"][i_src]
            matched += 1
        # else: team didn't exist last year; leave at 0 (cold prior)
    if matched == 0:
        return None
    return {
        "atts_mean": new_atts,
        "defs_mean": new_defs,
        "carryover_sd": priors["carryover_sd"],
    }


def run_chain() -> pd.DataFrame:
    """Run the full carryover chain and the cold-start baseline in parallel."""
    summaries = []
    carryover_priors = None  # cold start for the first season

    for i, season in enumerate(SEASONS):
        # ---- BASE (cold start every season) ----
        t0 = time.time()
        print(f"[base / {season}] backtest starting...", flush=True)
        try:
            preds_base = backtest(season=season, train_window=TRAIN_WINDOW,
                                  nsims=NSIMS, samples=SAMPLES)
        except Exception as e:
            print(f"[base / {season}] FAILED: {e}", flush=True)
            preds_base = None

        if preds_base is not None:
            elapsed = time.time() - t0
            s = summarize_backtest(preds_base)
            s.update({"variant": "base", "season": season, "elapsed_sec": round(elapsed, 1)})
            print(f"[base / {season}] done in {elapsed:.0f}s  "
                  f"acc={s.get('accuracy',0):.1%}  brier={s.get('brier_score',0):.3f}",
                  flush=True)
            preds_base.to_csv(RAW_DIR / f"carryover_base_{season}.csv", index=False)
            summaries.append(s)

        # ---- CARRYOVER (use priors from previous season's full-season fit) ----
        t0 = time.time()
        priors_this_season = None
        if carryover_priors is not None:
            # Need to align prior team-index to this season's team-index
            _, teams_this = load_game_data(season, weeks=[1])  # cheap teams index
            priors_this_season = align_priors_to_season(carryover_priors, teams_this)

        variant_label = "carryover" if priors_this_season is not None else "carryover-cold"
        print(f"[{variant_label} / {season}] backtest starting...", flush=True)
        try:
            preds_carry = backtest(season=season, train_window=TRAIN_WINDOW,
                                   nsims=NSIMS, samples=SAMPLES,
                                   team_priors=priors_this_season)
        except Exception as e:
            print(f"[{variant_label} / {season}] FAILED: {e}", flush=True)
            preds_carry = None

        if preds_carry is not None:
            elapsed = time.time() - t0
            s = summarize_backtest(preds_carry)
            s.update({"variant": variant_label, "season": season,
                      "elapsed_sec": round(elapsed, 1)})
            print(f"[{variant_label} / {season}] done in {elapsed:.0f}s  "
                  f"acc={s.get('accuracy',0):.1%}  brier={s.get('brier_score',0):.3f}",
                  flush=True)
            preds_carry.to_csv(RAW_DIR / f"carryover_{variant_label}_{season}.csv",
                               index=False)
            summaries.append(s)

        # Persist after each season
        pd.DataFrame(summaries).to_csv(RESULTS_DIR / "carryover_summary.csv", index=False)

        # Refit current season on full data to extract priors for the NEXT season
        if i < len(SEASONS) - 1:
            t0 = time.time()
            print(f"[handoff fit / {season}] fitting full season for next-year priors...",
                  flush=True)
            idata, teams_full, _ = fit_season_full(season, team_priors=priors_this_season)
            carryover_priors = extract_posterior_means(idata, teams_full)
            print(f"[handoff fit / {season}] done in {time.time()-t0:.0f}s; "
                  f"sample atts means: "
                  f"min={carryover_priors['atts_mean'].min():+.3f}, "
                  f"max={carryover_priors['atts_mean'].max():+.3f}",
                  flush=True)

    return pd.DataFrame(summaries)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Experiment 2: Multi-season carryover")
    print("=" * 70)
    print(f"  carryover_sd = {CARRYOVER_SD}")

    total_start = time.time()
    summaries = run_chain()

    print(f"\n{'='*70}\n  TOTAL: {(time.time() - total_start)/60:.1f} min\n{'='*70}")

    cols = ["variant", "season", "n_games", "accuracy", "brier_score", "mae_spread"]
    available = [c for c in cols if c in summaries.columns]
    print("\n" + summaries[available].to_string(index=False))


if __name__ == "__main__":
    main()
