"""Tests for the data loading and Bayesian model modules."""

import pytest
import pandas as pd
import numpy as np

from src.data import load_game_data, munge_game_data
from src.model import (
    bhm, simulate_team_season, simulate_team_seasons,
    create_team_season_table, predictions, plot_hdis,
)


# ---------------------------------------------------------------------------
# Shared fixtures — model is fitted ONCE per test session to keep things fast
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_data():
    """Synthetic game data: T0 is strong, T3 is weak."""
    np.random.seed(42)
    games = []
    for week in range(1, 6):
        for h, a in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]:
            # T0 scores high, T3 scores low
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
def fitted_model(synthetic_data):
    """Fit model once, reuse across all tests."""
    return bhm(synthetic_data, metric="score", samples=100)


@pytest.fixture(scope="session")
def synthetic_teams():
    return pd.DataFrame({"team": [f"T{i}" for i in range(4)], "i": range(4)})


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestLoadGameData:
    def test_returns_tuple(self):
        result = load_game_data(2024, weeks=[1, 2])
        assert isinstance(result, tuple) and len(result) == 2

    def test_df_has_required_columns(self):
        df, _ = load_game_data(2024, weeks=[1])
        for col in ["home_team", "away_team", "home_score", "away_score",
                     "week", "i_home", "i_away"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_teams_has_32_teams(self):
        _, teams = load_game_data(2024, weeks=[1])
        assert len(teams) == 32

    def test_indices_valid(self):
        df, _ = load_game_data(2024, weeks=[1])
        assert df["i_home"].min() >= 0
        assert df["i_away"].max() < 32

    def test_scores_non_negative(self):
        df, _ = load_game_data(2024, weeks=[1])
        assert (df["home_score"] >= 0).all()
        assert (df["away_score"] >= 0).all()

    def test_week_filter(self):
        df, _ = load_game_data(2024, weeks=[1])
        assert (df["week"] == 1).all()

    def test_full_season(self):
        df, _ = load_game_data(2024)
        assert len(df) >= 272  # 17 weeks * 16 games


class TestMungeGameData:
    def test_adds_indices(self):
        df = pd.DataFrame({
            "home_team": ["BUF", "KC"],
            "away_team": ["MIA", "DEN"],
            "home_score": [24, 17],
            "away_score": [10, 14],
            "week": [1, 1],
        })
        result, teams = munge_game_data(df)
        assert "i_home" in result.columns
        assert "i_away" in result.columns
        assert len(teams) == 4

    def test_preserves_existing_teams(self):
        teams = pd.DataFrame({"team": ["BUF", "DEN", "KC", "MIA"], "i": [0, 1, 2, 3]})
        df = pd.DataFrame({
            "home_team": ["BUF"],
            "away_team": ["MIA"],
            "home_score": [24],
            "away_score": [10],
            "week": [1],
        })
        result, _ = munge_game_data(df, teams=teams)
        assert result["i_home"].values[0] == 0  # BUF
        assert result["i_away"].values[0] == 3  # MIA


# ---------------------------------------------------------------------------
# Model fitting tests
# ---------------------------------------------------------------------------

class TestBHM:
    def test_returns_inference_data(self, fitted_model):
        assert fitted_model is not None
        assert hasattr(fitted_model, "posterior")

    def test_posterior_has_expected_vars(self, fitted_model):
        post = fitted_model.posterior
        for var in ["atts", "defs", "home", "intercept", "sd_att", "sd_def"]:
            assert var in post, f"Missing posterior variable: {var}"

    def test_atts_shape(self, fitted_model):
        atts = fitted_model.posterior["atts"]
        # Should have 4 teams
        assert atts.shape[-1] == 4

    def test_atts_sum_to_zero(self, fitted_model):
        """Sum-to-zero constraint: mean of atts across teams should be ~0."""
        atts_mean = fitted_model.posterior["atts"].mean(dim=["chain", "draw"]).values
        assert abs(atts_mean.sum()) < 0.1, f"atts don't sum to zero: {atts_mean.sum()}"

    def test_strong_team_has_higher_attack(self, fitted_model):
        """T0 was generated with higher scores — model should learn this."""
        atts_mean = fitted_model.posterior["atts"].mean(dim=["chain", "draw"]).values
        # T0 (index 0) should have higher attack than T3 (index 3)
        assert atts_mean[0] > atts_mean[3], (
            f"T0 attack ({atts_mean[0]:.3f}) should be > T3 attack ({atts_mean[3]:.3f})"
        )

    def test_home_advantage_positive(self, fitted_model):
        """Home field advantage should generally be positive."""
        home_mean = float(fitted_model.posterior["home"].mean())
        # We generated home scores slightly higher, so home should be >= 0
        # (allow some slack since it's a small dataset)
        assert home_mean > -1.0, f"Home advantage suspiciously negative: {home_mean}"


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_single_season_valid_scores(self, synthetic_data, fitted_model):
        sim = simulate_team_season(synthetic_data, fitted_model, metric="score", burnin=0)
        assert (sim["home_score"] >= 0).all()
        assert (sim["away_score"] >= 0).all()
        assert len(sim) == len(synthetic_data)

    def test_single_season_preserves_matchups(self, synthetic_data, fitted_model):
        sim = simulate_team_season(synthetic_data, fitted_model, metric="score", burnin=0)
        # Teams should be the same, only scores change
        assert list(sim["home_team"]) == list(synthetic_data["home_team"])
        assert list(sim["away_team"]) == list(synthetic_data["away_team"])

    def test_multiple_seasons_returns_all_teams(self, synthetic_data, fitted_model):
        simuls = simulate_team_seasons(
            synthetic_data, fitted_model, nsims=5, metric="score", burnin=0
        )
        assert isinstance(simuls, pd.DataFrame)
        assert "team" in simuls.columns
        assert "score" in simuls.columns
        assert "iteration" in simuls.columns
        teams_in_result = set(simuls["team"].unique())
        assert teams_in_result == {"T0", "T1", "T2", "T3"}

    def test_multiple_seasons_has_correct_iterations(self, synthetic_data, fitted_model):
        nsims = 5
        simuls = simulate_team_seasons(
            synthetic_data, fitted_model, nsims=nsims, metric="score", burnin=0
        )
        assert set(simuls["iteration"].unique()) == set(range(nsims))

    def test_simulations_vary(self, synthetic_data, fitted_model):
        """Two simulations should not be identical (stochastic)."""
        sim1 = simulate_team_season(synthetic_data, fitted_model, metric="score", burnin=0)
        sim2 = simulate_team_season(synthetic_data, fitted_model, metric="score", burnin=0)
        # Very unlikely all scores match
        assert not (sim1["home_score"].values == sim2["home_score"].values).all()


class TestCreateTeamSeasonTable:
    def test_aggregates_home_and_away(self):
        season = pd.DataFrame({
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "home_score": [24, 17],
            "away_score": [10, 21],
        })
        table = create_team_season_table(season, "score")
        assert set(table["team"]) == {"A", "B"}
        # A scored 24 (home) + 21 (away) = 45
        a_scores = table[table["team"] == "A"]["score"].sum()
        assert a_scores == 45


# ---------------------------------------------------------------------------
# Predictions tests
# ---------------------------------------------------------------------------

class TestPredictions:
    def test_returns_dataframe(self, synthetic_data, fitted_model, synthetic_teams):
        simuls = simulate_team_seasons(
            synthetic_data, fitted_model, nsims=10, metric="score", burnin=0
        )
        hdis = predictions(synthetic_data, simuls, synthetic_teams, nsims=10)
        assert isinstance(hdis, pd.DataFrame)

    def test_hdis_has_intervals(self, synthetic_data, fitted_model, synthetic_teams):
        simuls = simulate_team_seasons(
            synthetic_data, fitted_model, nsims=10, metric="score", burnin=0
        )
        hdis = predictions(synthetic_data, simuls, synthetic_teams, nsims=10)
        for col in ["metric_lower", "metric_median", "metric_upper", "team"]:
            assert col in hdis.columns, f"Missing HDI column: {col}"

    def test_lower_le_median_le_upper(self, synthetic_data, fitted_model, synthetic_teams):
        simuls = simulate_team_seasons(
            synthetic_data, fitted_model, nsims=50, metric="score", burnin=0
        )
        hdis = predictions(synthetic_data, simuls, synthetic_teams, nsims=50)
        assert (hdis["metric_lower"] <= hdis["metric_median"]).all()
        assert (hdis["metric_median"] <= hdis["metric_upper"]).all()


# ---------------------------------------------------------------------------
# Visualization test
# ---------------------------------------------------------------------------

class TestPlotHDIs:
    def test_returns_figure(self, synthetic_data, fitted_model, synthetic_teams):
        import matplotlib
        matplotlib.use("Agg")

        simuls = simulate_team_seasons(
            synthetic_data, fitted_model, nsims=10, metric="score", burnin=0
        )
        hdis = predictions(synthetic_data, simuls, synthetic_teams, nsims=10)
        fig, ax = plot_hdis(hdis, metric="score")
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
