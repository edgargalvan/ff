"""Rolling backtest for the hierarchical Bayesian model.

Trains on a window of past weeks, predicts the next week's games,
rolls forward, and measures prediction accuracy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data import load_game_data, munge_game_data
from src.model import bhm, simulate_team_season

logger = logging.getLogger(__name__)


def predict_week(idata, df_test: pd.DataFrame, nsims: int = 500,
                 covariates: list[str] | None = None) -> pd.DataFrame:
    """
    Predict outcomes for a set of games using a fitted model.

    For each game, runs nsims simulations from the posterior and computes:
    - home_win_prob: fraction of sims where home team wins
    - pred_spread: median of (away_score - home_score) across sims

    Args:
        idata: fitted ArviZ InferenceData
        df_test: DataFrame of games to predict (must have i_home, i_away, etc.)
        nsims: number of simulations per game
        covariates: covariates used during fitting

    Returns:
        DataFrame with one row per game, prediction columns added
    """
    home_wins = np.zeros(len(df_test))
    spreads = np.zeros((nsims, len(df_test)))

    for i in range(nsims):
        sim = simulate_team_season(df_test, idata, metric="score",
                                   burnin=0, covariates=covariates)
        home_won = sim["home_score"].values > sim["away_score"].values
        home_wins += home_won
        spreads[i] = sim["away_score"].values - sim["home_score"].values

    results = df_test[["week", "home_team", "away_team",
                       "home_score", "away_score"]].copy()
    results = results.rename(columns={
        "home_score": "home_score_actual",
        "away_score": "away_score_actual",
    })

    results["home_win_prob"] = home_wins / nsims
    results["pred_spread"] = np.median(spreads, axis=0)
    results["actual_spread"] = results["away_score_actual"] - results["home_score_actual"]

    results["predicted_winner"] = np.where(
        results["home_win_prob"] > 0.5,
        results["home_team"], results["away_team"]
    )
    results["actual_winner"] = np.where(
        results["home_score_actual"] > results["away_score_actual"],
        results["home_team"], results["away_team"]
    )
    # Ties go to away team (convention)
    results["correct"] = results["predicted_winner"] == results["actual_winner"]
    results["confidence"] = np.abs(results["home_win_prob"] - 0.5) + 0.5

    return results


def backtest(season: int, train_window: int = 8, nsims: int = 500,
             samples: int = 500, time_varying: bool = False,
             covariates: list[str] | None = None,
             start_week: int | None = None) -> pd.DataFrame:
    """
    Rolling backtest across a season.

    For each test week, trains the model on the prior train_window weeks
    and predicts that week's games.

    Args:
        season: NFL season year
        train_window: number of weeks to train on
        nsims: simulations per game for prediction
        samples: MCMC samples per model fit
        time_varying: use time-varying team strengths
        covariates: covariate columns to include
        start_week: first week to predict (default: train_window + 1)

    Returns:
        DataFrame of all predictions with accuracy columns
    """
    # Load full season
    df_all, teams = load_game_data(season)
    all_weeks = sorted(df_all["week"].unique())

    if start_week is None:
        start_week = train_window + 1

    test_weeks = [w for w in all_weeks if w >= start_week]
    all_results = []

    for test_week in test_weeks:
        # Training data: prior train_window weeks
        train_weeks = [w for w in all_weeks if w < test_week][-train_window:]
        df_train = df_all[df_all["week"].isin(train_weeks)].copy()
        df_test = df_all[df_all["week"] == test_week].copy()

        if df_train.empty or df_test.empty:
            continue

        logger.info("Week %d: training on weeks %s", test_week, train_weeks)

        # Fit model
        try:
            idata = bhm(df_train, metric="score", samples=samples,
                        time_varying=time_varying, covariates=covariates)
        except Exception as e:
            logger.warning("Model failed for week %d: %s", test_week, e)
            continue

        # Predict
        preds = predict_week(idata, df_test, nsims=nsims, covariates=covariates)
        all_results.append(preds)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def summarize_backtest(results: pd.DataFrame) -> dict:
    """
    Compute summary metrics from backtest results.

    Returns dict with:
        accuracy: overall % correct winner picks
        n_games: number of games predicted
        n_correct: number correct
        mae_spread: mean absolute error on spread prediction
        brier_score: Brier score (lower is better, 0.25 = coin flip)
        accuracy_by_confidence: dict of confidence tier -> accuracy
    """
    if results.empty:
        return {"accuracy": 0, "n_games": 0}

    n = len(results)
    n_correct = results["correct"].sum()

    # Brier score: mean of (predicted_prob - actual_outcome)^2
    # where actual_outcome = 1 if home wins, 0 otherwise
    actual_home_win = (results["home_score_actual"] > results["away_score_actual"]).astype(float)
    brier = float(np.mean((results["home_win_prob"] - actual_home_win) ** 2))

    # MAE on spread
    mae = float(np.mean(np.abs(results["pred_spread"] - results["actual_spread"])))

    # Accuracy by confidence tier
    tiers = {}
    for threshold in [0.55, 0.60, 0.65, 0.70]:
        mask = results["confidence"] >= threshold
        if mask.sum() > 0:
            tier_acc = results.loc[mask, "correct"].mean()
            tiers[f">={threshold:.0%}"] = {
                "accuracy": float(tier_acc),
                "n_games": int(mask.sum()),
            }

    return {
        "accuracy": float(n_correct / n),
        "n_games": int(n),
        "n_correct": int(n_correct),
        "mae_spread": mae,
        "brier_score": brier,
        "accuracy_by_confidence": tiers,
    }


def print_summary(results: pd.DataFrame, name: str = "Model") -> None:
    """Print a formatted backtest summary."""
    s = summarize_backtest(results)
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Games predicted:  {s['n_games']}")
    print(f"  Correct:          {s['n_correct']}/{s['n_games']} ({s['accuracy']:.1%})")
    print(f"  Spread MAE:       {s.get('mae_spread', 0):.1f} points")
    print(f"  Brier score:      {s.get('brier_score', 0):.3f} (0.250 = coin flip)")

    if s.get("accuracy_by_confidence"):
        print(f"\n  By confidence:")
        for tier, info in s["accuracy_by_confidence"].items():
            print(f"    {tier}: {info['accuracy']:.1%} ({info['n_games']} games)")
    print()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

    print(f"Running backtest for {season} season...")
    results = backtest(season, train_window=8, nsims=500, samples=500)
    print_summary(results, name=f"Base model ({season})")
