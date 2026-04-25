"""Three-way calibration experiment: Poisson vs NB-weak vs NB-tight.

For each variant, runs:
  1. Rolling 4-season backtest (2022-2025) — saves per-game predictions
     and per-season summary metrics.
  2. Single-season posterior predictive check on full 2024 — saves
     observed/simulated score arrays.

Outputs go to results/raw/calib_{variant}_{...} and results/calib_summary.csv.
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
    {"name": "poisson", "likelihood": "poisson", "alpha_prior": "weak"},
    {"name": "nb_weak", "likelihood": "negbin", "alpha_prior": "weak"},
    {"name": "nb_tight", "likelihood": "negbin", "alpha_prior": "tight"},
]

TRAIN_WINDOW = 8
SAMPLES = 500
NSIMS = 500

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"


def run_backtest_grid() -> pd.DataFrame:
    """Run all (variant, season) backtests, save raw + summary."""
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
                    likelihood=variant["likelihood"],
                    alpha_prior=variant["alpha_prior"],
                )
            except Exception as e:
                print(f"[{variant['name']} / {season}] FAILED: {e}", flush=True)
                summaries.append({
                    "variant": variant["name"], "season": season, "error": str(e),
                })
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
            preds.to_csv(RAW_DIR / f"calib_{variant['name']}_{season}.csv", index=False)
            summaries.append(summary)
            # Persist summary incrementally
            pd.DataFrame(summaries).to_csv(RESULTS_DIR / "calib_summary.csv", index=False)

    return pd.DataFrame(summaries)


def run_ppc_per_variant() -> None:
    """Single-season PPC + coverage on full 2024 for each variant."""
    df, teams = load_game_data(2024, weeks=list(range(1, 19)))
    print(f"\nPPC fit on {len(df)} games (full 2024 regular season)\n", flush=True)

    for variant in VARIANTS:
        t0 = time.time()
        print(f"[{variant['name']}] PPC fit...", flush=True)
        idata = bhm(
            df, metric="score", samples=1000,
            likelihood=variant["likelihood"],
            alpha_prior=variant["alpha_prior"],
        )

        # Posterior predictive samples
        post = idata.posterior
        n_total = post.sizes["chain"] * post.sizes["draw"]
        atts = post["atts"].values.reshape(n_total, -1)
        defs = post["defs"].values.reshape(n_total, -1)
        home_adv = post["home"].values.flatten()
        intercept = post["intercept"].values.flatten()
        is_negbin = "alpha" in post
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
            log_h = intercept[s] + home_adv[s] + atts[s, i_home] + defs[s, i_away]
            log_a = intercept[s] + atts[s, i_away] + defs[s, i_home]
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

        # Quick summary
        print(
            f"[{variant['name']}] done in {elapsed:.0f}s  "
            f"obs std (h/a): {obs_home.std():.2f}/{obs_away.std():.2f}  "
            f"sim std (h/a): {sim_home_pool.std():.2f}/{sim_away_pool.std():.2f}",
            flush=True,
        )

        # Spread coverage
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

        # Save artifacts
        np.savez_compressed(
            RAW_DIR / f"calib_ppc_{variant['name']}_2024.npz",
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
    print("  Calibration experiment: Poisson vs NB-weak vs NB-tight")
    print("=" * 70)

    total_start = time.time()
    summaries = run_backtest_grid()
    print(f"\n{'='*70}\nBacktest grid done in {(time.time() - total_start)/60:.1f} min")
    print("=" * 70)

    print(f"\n{'='*70}\n  Posterior predictive checks (2024 full-season)\n{'='*70}")
    run_ppc_per_variant()

    print(f"\n{'='*70}\n  TOTAL: {(time.time() - total_start)/60:.1f} min\n{'='*70}")

    # Final pretty summary
    cols = ["variant", "season", "n_games", "accuracy", "brier_score", "mae_spread"]
    available = [c for c in cols if c in summaries.columns]
    print("\n" + summaries[available].to_string(index=False))


if __name__ == "__main__":
    main()
