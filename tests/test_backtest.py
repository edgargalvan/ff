"""Tests for backtest and prediction functions."""

import pytest
import pandas as pd
import numpy as np

from src.backtest import predict_week, summarize_backtest
from src.model import bhm


# ---------------------------------------------------------------------------
# Reuse the session-scoped model fixture from test_model.py via conftest
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def backtest_synthetic_data():
    np.random.seed(42)
    games = []
    for week in range(1, 6):
        for h, a in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]:
            home_boost = 5 if h == 0 else (-3 if h == 3 else 0)
            away_boost = 5 if a == 0 else (-3 if a == 3 else 0)
            games.append({
                "home_team": f"T{h}", "away_team": f"T{a}",
                "home_score": max(0, np.random.poisson(22 + home_boost)),
                "away_score": max(0, np.random.poisson(20 + away_boost)),
                "i_home": h, "i_away": a, "week": week,
            })
    return pd.DataFrame(games)


@pytest.fixture(scope="session")
def backtest_fitted_model(backtest_synthetic_data):
    return bhm(backtest_synthetic_data, metric="score", samples=100)


# ---------------------------------------------------------------------------
# predict_week tests
# ---------------------------------------------------------------------------

class TestPredictWeek:
    def test_returns_dataframe(self, backtest_synthetic_data, backtest_fitted_model):
        df_test = backtest_synthetic_data[backtest_synthetic_data["week"] == 5].copy()
        result = predict_week(backtest_fitted_model, df_test, nsims=20)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, backtest_synthetic_data, backtest_fitted_model):
        df_test = backtest_synthetic_data[backtest_synthetic_data["week"] == 5].copy()
        result = predict_week(backtest_fitted_model, df_test, nsims=20)
        for col in ["home_win_prob", "pred_spread", "actual_spread",
                     "predicted_winner", "actual_winner", "correct", "confidence"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_home_win_prob_bounded(self, backtest_synthetic_data, backtest_fitted_model):
        df_test = backtest_synthetic_data[backtest_synthetic_data["week"] == 5].copy()
        result = predict_week(backtest_fitted_model, df_test, nsims=50)
        assert (result["home_win_prob"] >= 0).all()
        assert (result["home_win_prob"] <= 1).all()

    def test_confidence_bounded(self, backtest_synthetic_data, backtest_fitted_model):
        df_test = backtest_synthetic_data[backtest_synthetic_data["week"] == 5].copy()
        result = predict_week(backtest_fitted_model, df_test, nsims=50)
        assert (result["confidence"] >= 0.5).all()
        assert (result["confidence"] <= 1.0).all()

    def test_correct_is_boolean(self, backtest_synthetic_data, backtest_fitted_model):
        df_test = backtest_synthetic_data[backtest_synthetic_data["week"] == 5].copy()
        result = predict_week(backtest_fitted_model, df_test, nsims=20)
        assert result["correct"].dtype == bool

    def test_one_row_per_game(self, backtest_synthetic_data, backtest_fitted_model):
        df_test = backtest_synthetic_data[backtest_synthetic_data["week"] == 5].copy()
        result = predict_week(backtest_fitted_model, df_test, nsims=20)
        assert len(result) == len(df_test)


# ---------------------------------------------------------------------------
# summarize_backtest tests
# ---------------------------------------------------------------------------

class TestSummarizeBacktest:
    def test_perfect_predictions(self):
        results = pd.DataFrame({
            "home_score_actual": [30, 20, 25],
            "away_score_actual": [10, 25, 20],
            "home_win_prob": [0.9, 0.2, 0.8],
            "pred_spread": [-20, 5, -5],
            "actual_spread": [-20, 5, -5],
            "predicted_winner": ["A", "D", "E"],
            "actual_winner": ["A", "D", "E"],
            "correct": [True, True, True],
            "confidence": [0.9, 0.8, 0.8],
        })
        s = summarize_backtest(results)
        assert s["accuracy"] == 1.0
        assert s["n_games"] == 3
        assert s["mae_spread"] == 0.0

    def test_coin_flip_brier(self):
        """All 50/50 predictions should give Brier score ~0.25."""
        n = 100
        results = pd.DataFrame({
            "home_score_actual": [20] * n,
            "away_score_actual": [10] * n,  # home always wins
            "home_win_prob": [0.5] * n,
            "pred_spread": [0] * n,
            "actual_spread": [-10] * n,
            "predicted_winner": ["A"] * n,
            "actual_winner": ["A"] * n,
            "correct": [True] * n,
            "confidence": [0.5] * n,
        })
        s = summarize_backtest(results)
        assert abs(s["brier_score"] - 0.25) < 0.01

    def test_empty_results(self):
        s = summarize_backtest(pd.DataFrame())
        assert s["accuracy"] == 0
        assert s["n_games"] == 0

    def test_confidence_tiers(self):
        results = pd.DataFrame({
            "home_score_actual": [30, 20, 25, 28],
            "away_score_actual": [10, 25, 20, 14],
            "home_win_prob": [0.9, 0.3, 0.6, 0.55],
            "pred_spread": [-20, 5, -5, -14],
            "actual_spread": [-20, 5, -5, -14],
            "predicted_winner": ["A", "D", "E", "G"],
            "actual_winner": ["A", "D", "E", "G"],
            "correct": [True, True, True, True],
            "confidence": [0.9, 0.7, 0.6, 0.55],
        })
        s = summarize_backtest(results)
        tiers = s["accuracy_by_confidence"]
        assert ">=55%" in tiers
        assert tiers[">=55%"]["n_games"] == 4
        assert ">=70%" in tiers
        assert tiers[">=70%"]["n_games"] == 2  # only 0.9 and 0.7
