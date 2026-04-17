"""Multi-season backtest runner.

Runs each (variant, season) combination, saves raw per-game predictions
to results/raw/, and writes a summary to results/summary.csv.

Usage:
    python scripts/run_multiseason_backtest.py
"""

import logging
import time
from pathlib import Path

import pandas as pd

# Add project root to path so `src` imports work when run as a script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import backtest, summarize_backtest


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEASONS = [2022, 2023, 2024]

VARIANTS = [
    {"name": "base", "time_varying": False, "covariates": None},
    {"name": "covariates", "time_varying": False,
     "covariates": ["rest_advantage", "temp_std", "wind_std"]},
    {"name": "time_varying", "time_varying": True, "covariates": None},
]

TRAIN_WINDOW = 8
SAMPLES = 500
NSIMS = 500

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(variant: dict, season: int) -> tuple[pd.DataFrame, dict]:
    """Run one (variant, season) backtest. Returns (predictions, summary)."""
    t0 = time.time()
    print(f"[{variant['name']} / {season}] starting...", flush=True)

    preds = backtest(
        season=season,
        train_window=TRAIN_WINDOW,
        nsims=NSIMS,
        samples=SAMPLES,
        time_varying=variant.get("time_varying", False),
        covariates=variant.get("covariates"),
    )

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
    return preds, summary


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    total_start = time.time()

    for variant in VARIANTS:
        for season in SEASONS:
            try:
                preds, summary = run_one(variant, season)
            except Exception as e:
                print(f"[{variant['name']} / {season}] FAILED: {e}", flush=True)
                all_summaries.append({
                    "variant": variant["name"], "season": season,
                    "error": str(e),
                })
                continue

            # Save raw predictions
            out_path = RAW_DIR / f"{variant['name']}_{season}.csv"
            preds.to_csv(out_path, index=False)

            # Save summary so far (so we have partial results if interrupted)
            all_summaries.append(summary)
            pd.DataFrame(all_summaries).to_csv(RESULTS_DIR / "summary.csv", index=False)

    total_elapsed = time.time() - total_start
    print(f"\nTotal elapsed: {total_elapsed / 60:.1f} min", flush=True)

    # Final summary
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    print("\n=== Summary ===", flush=True)
    cols = ["variant", "season", "n_games", "accuracy", "brier_score", "mae_spread"]
    available = [c for c in cols if c in summary_df.columns]
    print(summary_df[available].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
