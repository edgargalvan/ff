"""Follow-up backtest for 2025 season across all 3 variants.

Appends results to the existing results/summary.csv and saves raw predictions
to results/raw/ alongside the 2022-2024 runs.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import backtest, summarize_backtest

SEASON = 2025
VARIANTS = [
    {"name": "base", "time_varying": False, "covariates": None},
    {"name": "covariates", "time_varying": False,
     "covariates": ["rest_advantage", "temp_std", "wind_std"]},
    # time_varying skipped: rejected in 2022-2024 backtest (~100x slower,
    # ~0.024 worse Brier). See results/summary.csv.
]

TRAIN_WINDOW = 8
SAMPLES = 500
NSIMS = 500

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RAW_DIR = RESULTS_DIR / "raw"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    import time
    new_summaries = []

    for variant in VARIANTS:
        t0 = time.time()
        print(f"[{variant['name']} / {SEASON}] starting...", flush=True)
        try:
            preds = backtest(
                season=SEASON, train_window=TRAIN_WINDOW,
                nsims=NSIMS, samples=SAMPLES,
                time_varying=variant.get("time_varying", False),
                covariates=variant.get("covariates"),
            )
            elapsed = time.time() - t0
            summary = summarize_backtest(preds)
            summary["variant"] = variant["name"]
            summary["season"] = SEASON
            summary["elapsed_sec"] = round(elapsed, 1)
            print(
                f"[{variant['name']} / {SEASON}] done in {elapsed:.0f}s  "
                f"n={summary['n_games']}  acc={summary.get('accuracy', 0):.1%}  "
                f"brier={summary.get('brier_score', 0):.3f}",
                flush=True,
            )
            preds.to_csv(RAW_DIR / f"{variant['name']}_{SEASON}.csv", index=False)
            new_summaries.append(summary)
        except Exception as e:
            print(f"[{variant['name']} / {SEASON}] FAILED: {e}", flush=True)
            new_summaries.append({"variant": variant["name"], "season": SEASON, "error": str(e)})

    # Merge with existing summary
    summary_path = RESULTS_DIR / "summary.csv"
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        # Drop any existing 2025 rows (idempotent re-runs)
        existing = existing[existing["season"] != SEASON] if "season" in existing.columns else existing
        combined = pd.concat([existing, pd.DataFrame(new_summaries)], ignore_index=True)
    else:
        combined = pd.DataFrame(new_summaries)

    combined.to_csv(summary_path, index=False)
    print(f"\nUpdated {summary_path}", flush=True)


if __name__ == "__main__":
    main()
