"""Experiment 1: AR(1) state-space with hierarchical anchor.

Compares:
  - base:    static team strengths (current default)
  - ar1:     hierarchical anchor + RW deviation (the GS-style fix to the
             rejected `time_varying` variant)

For each variant:
  1. Rolling 4-season backtest 2022-2025.
  2. Single-season PPC + spread coverage on 2024.

Outputs to results/raw/ar1_{variant}_{...} and results/ar1_summary.csv.
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
    {"name": "base", "time_varying": False},
    {"name": "ar1",  "time_varying": True},   # rebuilt with hierarchical anchor
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
                    time_varying=variant.get("time_varying", False),
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
            preds.to_csv(RAW_DIR / f"ar1_{variant['name']}_{season}.csv", index=False)
            summaries.append(summary)
            pd.DataFrame(summaries).to_csv(RESULTS_DIR / "ar1_summary.csv", index=False)

    return pd.DataFrame(summaries)


def run_ppc_per_variant() -> None:
    df, teams = load_game_data(2024, weeks=list(range(1, 19)))
    print(f"\nPPC fit on {len(df)} games (full 2024 regular season)\n", flush=True)

    for variant in VARIANTS:
        t0 = time.time()
        print(f"[{variant['name']}] PPC fit...", flush=True)
        idata = bhm(df, metric="score", samples=1000,
                    time_varying=variant.get("time_varying", False))

        post = idata.posterior
        n_total = post.sizes["chain"] * post.sizes["draw"]
        is_negbin = "alpha" in post
        is_time_varying = "delta_atts" in post

        if is_time_varying:
            # `atts` is (chains, draws, n_weeks, n_teams) — use last week
            atts_last = post["atts"].values[..., -1, :].reshape(n_total, -1)
            defs_last = post["defs"].values[..., -1, :].reshape(n_total, -1)
        else:
            atts_last = post["atts"].values.reshape(n_total, -1)
            defs_last = post["defs"].values.reshape(n_total, -1)
        home_adv = post["home"].values.flatten()
        intercept = post["intercept"].values.flatten()
        alpha = post["alpha"].values.flatten() if is_negbin else None

        n_games = len(df)
        i_home = df["i_home"].values
        i_away = df["i_away"].values

        np.random.seed(0)
        n_pp = 500
        idx = np.random.choice(n_total, n_pp, replace=False)

        sim_home = np.zeros((n_pp, n_games))
        sim_away = np.zeros((n_pp, n_games))

        for k, s in enumerate(idx):
            log_h = intercept[s] + home_adv[s] + atts_last[s, i_home] + defs_last[s, i_away]
            log_a = intercept[s] + atts_last[s, i_away] + defs_last[s, i_home]
            mu_h = np.exp(log_h)
            mu_a = np.exp(log_a)
            if is_negbin:
                a = alpha[s]
                p_h = a / (a + mu_h)
                p_a = a / (a + mu_a)
                sim_home[k] = np.random.negative_binomial(a, p_h)
                sim_away[k] = np.random.negative_binomial(a, p_a)
            else:
                sim_home[k] = np.random.poisson(mu_h)
                sim_away[k] = np.random.poisson(mu_a)

        elapsed = time.time() - t0
        obs_home = df["home_score"].values
        obs_away = df["away_score"].values
        sim_home_pool = sim_home.flatten()
        sim_away_pool = sim_away.flatten()

        print(
            f"[{variant['name']}] done in {elapsed:.0f}s  "
            f"obs std (h/a): {obs_home.std():.2f}/{obs_away.std():.2f}  "
            f"sim std (h/a): {sim_home_pool.std():.2f}/{sim_away_pool.std():.2f}",
            flush=True,
        )

        sim_spread = sim_away - sim_home
        obs_spread = obs_away - obs_home
        cov_lines = []
        for level in [0.50, 0.80, 0.90, 0.95]:
            lo = (1 - level) / 2
            hi = 1 - lo
            ci_lo = np.percentile(sim_spread, lo * 100, axis=0)
            ci_hi = np.percentile(sim_spread, hi * 100, axis=0)
            in_ci = (obs_spread >= ci_lo) & (obs_spread <= ci_hi)
            cov_lines.append(f"{int(level*100)}%CI={in_ci.mean():.3f}")
        print(f"  spread coverage: {'  '.join(cov_lines)}", flush=True)

        np.savez_compressed(
            RAW_DIR / f"ar1_ppc_{variant['name']}_2024.npz",
            sim_home=sim_home, sim_away=sim_away,
            obs_home=obs_home, obs_away=obs_away,
            week=df["week"].values,
            home_team=df["home_team"].values,
            away_team=df["away_team"].values,
        )


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Experiment 1: AR(1) state-space with hierarchical anchor")
    print("=" * 70)

    total_start = time.time()
    summaries = run_backtest_grid()
    print(f"\n{'='*70}\nBacktest grid done in {(time.time() - total_start)/60:.1f} min")
    print("=" * 70)

    print(f"\n{'='*70}\n  Posterior predictive checks (2024 full-season)\n{'='*70}")
    run_ppc_per_variant()

    print(f"\n{'='*70}\n  TOTAL: {(time.time() - total_start)/60:.1f} min\n{'='*70}")

    cols = ["variant", "season", "n_games", "accuracy", "brier_score", "mae_spread"]
    available = [c for c in cols if c in summaries.columns]
    print("\n" + summaries[available].to_string(index=False))


if __name__ == "__main__":
    main()
