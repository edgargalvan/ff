"""Experiment 4: Per-team NB dispersion alpha.

Compares:
  - base:            shared scalar alpha (current default, weak prior)
  - per_team_alpha:  hierarchical per-team alpha[t] with shared
                     log-normal hyperprior

Outputs to results/raw/perteam_alpha_{variant}_{...} and
results/perteam_alpha_summary.csv.
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

VARIANTS = [
    {"name": "base",            "per_team_alpha": False},
    {"name": "per_team_alpha",  "per_team_alpha": True},
]

TRAIN_WINDOW = 8
SAMPLES = 500
NSIMS = 500

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"


def run_backtest_grid() -> pd.DataFrame:
    summaries = []
    for variant in VARIANTS:
        for season in SEASONS:
            t0 = time.time()
            print(f"[{variant['name']} / {season}] backtest starting...", flush=True)
            try:
                preds = backtest(
                    season=season,
                    train_window=TRAIN_WINDOW,
                    nsims=NSIMS,
                    samples=SAMPLES,
                    per_team_alpha=variant.get("per_team_alpha", False),
                )
            except Exception as e:
                print(f"[{variant['name']} / {season}] FAILED: {e}", flush=True)
                summaries.append({"variant": variant["name"], "season": season, "error": str(e)})
                continue

            elapsed = time.time() - t0
            summary = summarize_backtest(preds)
            summary["variant"] = variant["name"]
            summary["season"] = season
            summary["elapsed_sec"] = round(elapsed, 1)
            print(
                f"[{variant['name']} / {season}] done in {elapsed:.0f}s  "
                f"n={summary['n_games']}  acc={summary.get('accuracy', 0):.1%}  "
                f"brier={summary.get('brier_score', 0):.3f}",
                flush=True,
            )
            preds.to_csv(RAW_DIR / f"perteam_alpha_{variant['name']}_{season}.csv", index=False)
            summaries.append(summary)
            pd.DataFrame(summaries).to_csv(RESULTS_DIR / "perteam_alpha_summary.csv", index=False)

    return pd.DataFrame(summaries)


def run_inspection() -> None:
    """Look at per-team alpha posterior on a full-2024 fit."""
    df, teams = load_game_data(2024)
    print(f"\nInspection fit on {len(df)} games (full 2024)\n", flush=True)
    idata = bhm(df, samples=1000, per_team_alpha=True)

    alpha_team_post = idata.posterior["alpha_team"].values
    means = alpha_team_post.mean(axis=(0, 1))
    sds = alpha_team_post.std(axis=(0, 1))
    log_alpha_mean = float(idata.posterior["log_alpha_mean"].mean())
    sigma_log_alpha = float(idata.posterior["sigma_log_alpha"].mean())

    teams_sorted = teams.copy()
    teams_sorted["alpha_mean"] = means
    teams_sorted["alpha_sd"] = sds
    teams_sorted = teams_sorted.sort_values("alpha_mean", ascending=False)

    print(f"  Global log_alpha_mean: {log_alpha_mean:.3f}  (alpha ~ {np.exp(log_alpha_mean):.2f})")
    print(f"  Global sigma_log_alpha: {sigma_log_alpha:.3f}  (cross-team scale)")
    print(f"  alpha range: {means.min():.2f} to {means.max():.2f}")
    print(f"  Top 5 alpha (most consistent / least dispersed):")
    for _, row in teams_sorted.head(5).iterrows():
        print(f"    {row['team']}: alpha={row['alpha_mean']:.2f} ± {row['alpha_sd']:.2f}")
    print(f"  Bottom 5 alpha (most boom-or-bust):")
    for _, row in teams_sorted.tail(5).iterrows():
        print(f"    {row['team']}: alpha={row['alpha_mean']:.2f} ± {row['alpha_sd']:.2f}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Experiment 4: Per-team NB dispersion (alpha[t])")
    print("=" * 70)

    total_start = time.time()
    summaries = run_backtest_grid()
    print(f"\nBacktest grid done in {(time.time() - total_start)/60:.1f} min")

    print(f"\n{'='*70}\n  Posterior inspection (full 2024)\n{'='*70}")
    run_inspection()

    print(f"\n{'='*70}\n  TOTAL: {(time.time() - total_start)/60:.1f} min\n{'='*70}")

    cols = ["variant", "season", "n_games", "accuracy", "brier_score", "mae_spread"]
    available = [c for c in cols if c in summaries.columns]
    print("\n" + summaries[available].to_string(index=False))


if __name__ == "__main__":
    main()
