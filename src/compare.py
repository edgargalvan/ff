"""Compare multiple model variants via backtesting."""

from __future__ import annotations

import pandas as pd

from src.backtest import backtest, summarize_backtest


def compare_models(season: int, configs: list[dict],
                   train_window: int = 8, nsims: int = 500,
                   samples: int = 500) -> pd.DataFrame:
    """
    Run backtests for multiple model configurations and compare results.

    Args:
        season: NFL season year
        configs: list of dicts, each with:
            - name: display name
            - time_varying: bool (default False)
            - covariates: list[str] or None (default None)
        train_window: weeks of training data per fit
        nsims: simulations per prediction
        samples: MCMC samples per fit

    Returns:
        DataFrame with one row per model, columns for each metric
    """
    rows = []

    for config in configs:
        name = config["name"]
        print(f"\n>>> Running: {name}")

        results = backtest(
            season,
            train_window=train_window,
            nsims=nsims,
            samples=samples,
            time_varying=config.get("time_varying", False),
            covariates=config.get("covariates"),
        )

        summary = summarize_backtest(results)
        summary["name"] = name
        rows.append(summary)

    comparison = pd.DataFrame(rows)
    cols = ["name", "accuracy", "n_games", "n_correct", "mae_spread", "brier_score"]
    available = [c for c in cols if c in comparison.columns]
    return comparison[available]


def print_comparison(comparison: pd.DataFrame) -> None:
    """Pretty-print a model comparison table."""
    print("\n" + "=" * 65)
    print("  Model Comparison")
    print("=" * 65)
    print(f"  {'Model':<25} {'Accuracy':>10} {'MAE Spread':>12} {'Brier':>8}")
    print("-" * 65)
    for _, row in comparison.iterrows():
        print(f"  {row['name']:<25} {row['accuracy']:>9.1%} {row['mae_spread']:>11.1f} {row['brier_score']:>8.3f}")
    print("=" * 65)
    print()


if __name__ == "__main__":
    import sys

    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

    configs = [
        {"name": "base", "time_varying": False, "covariates": None},
        {"name": "+covariates", "time_varying": False,
         "covariates": ["rest_advantage", "temp_std", "wind_std"]},
        {"name": "+time_varying", "time_varying": True, "covariates": None},
    ]

    comparison = compare_models(season, configs, samples=500, nsims=500)
    print_comparison(comparison)
