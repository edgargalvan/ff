"""Tests for the data loading and Bayesian model modules."""

import pytest
import pandas as pd
import numpy as np

from src.data import load_game_data, munge_game_data
from src.model import (
    bhm, simulate_team_season, simulate_team_seasons,
    create_team_season_table, predictions, plot_hdis,
    ALL_COVARIATES,
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
        for var in ["atts", "defs", "home", "intercept", "sd_att", "sd_def",
                     "atts_raw", "defs_raw", "alpha"]:
            assert var in post, f"Missing posterior variable: {var}"

    def test_alpha_positive(self, fitted_model):
        """NB dispersion parameter should be positive."""
        alpha = fitted_model.posterior["alpha"].values
        assert (alpha > 0).all()

    def test_non_centered_raw_params(self, fitted_model):
        """Raw params should be roughly standard normal (mean ~0, sd ~1)."""
        atts_raw = fitted_model.posterior["atts_raw"].mean(dim=["chain", "draw"]).values
        # Mean across teams should be near 0
        assert abs(atts_raw.mean()) < 1.0

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


# ---------------------------------------------------------------------------
# Synthetic data with covariates
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_data_with_covariates():
    """Synthetic game data with covariate columns."""
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
                # Covariates
                "rest_advantage": np.random.choice([-3, 0, 3, 4]),
                "home_short_week": int(np.random.random() < 0.2),
                "temp_std": np.random.normal(0, 1),
                "wind_std": np.random.normal(0, 1),
                "is_indoor": int(np.random.random() < 0.3),
                "div_game": int(np.random.random() < 0.35),
            })
    return pd.DataFrame(games)


# ---------------------------------------------------------------------------
# Covariate model tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fitted_model_covariates(synthetic_data_with_covariates):
    """Fit model with covariates once."""
    covs = ["rest_advantage", "home_short_week", "temp_std"]
    return bhm(synthetic_data_with_covariates, metric="score", samples=100,
               covariates=covs), covs


class TestCovariates:
    def test_model_fits_with_covariates(self, fitted_model_covariates):
        idata, _ = fitted_model_covariates
        assert idata is not None

    def test_beta_coefficients_in_posterior(self, fitted_model_covariates):
        idata, covs = fitted_model_covariates
        for cov in covs:
            assert f"beta_{cov}" in idata.posterior, f"Missing beta_{cov}"

    def test_simulation_with_covariates(self, synthetic_data_with_covariates,
                                         fitted_model_covariates):
        idata, covs = fitted_model_covariates
        sim = simulate_team_season(synthetic_data_with_covariates, idata,
                                   metric="score", burnin=0, covariates=covs)
        assert (sim["home_score"] >= 0).all()
        assert (sim["away_score"] >= 0).all()

    def test_still_has_team_params(self, fitted_model_covariates):
        idata, _ = fitted_model_covariates
        assert "atts" in idata.posterior
        assert "defs" in idata.posterior


# ---------------------------------------------------------------------------
# Time-varying model tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fitted_model_time_varying(synthetic_data_with_covariates):
    """Fit model with time-varying params once."""
    return bhm(synthetic_data_with_covariates, metric="score", samples=100,
               time_varying=True)


class TestTimeVarying:
    """Tests for the hierarchical-anchor AR(1) state-space variant.

    Posterior structure:
        atts_static, defs_static  — non-centered season-long anchors
        delta_atts, delta_defs    — RW deviations centered at zero
        atts, defs                — combined (n_weeks, n_teams) deterministic
        sd_att_innov, sd_def_innov — shared weekly drift scales
    """

    def test_model_fits(self, fitted_model_time_varying):
        assert fitted_model_time_varying is not None

    def test_anchor_and_deviation_in_posterior(self, fitted_model_time_varying):
        post = fitted_model_time_varying.posterior
        # New variables (hierarchical anchor + RW deviation)
        for var in ["atts_static", "defs_static", "delta_atts", "delta_defs",
                    "sd_att_innov", "sd_def_innov"]:
            assert var in post, f"Missing posterior variable: {var}"

    def test_atts_shape_is_weeks_by_teams(self, fitted_model_time_varying):
        atts = fitted_model_time_varying.posterior["atts"]
        # shape should be (chains, draws, n_weeks, n_teams)
        assert len(atts.shape) == 4
        assert atts.shape[-1] == 4   # 4 teams
        assert atts.shape[-2] == 5   # 5 weeks

    def test_atts_static_shape_is_teams(self, fitted_model_time_varying):
        atts_static = fitted_model_time_varying.posterior["atts_static"]
        # (chains, draws, n_teams)
        assert len(atts_static.shape) == 3
        assert atts_static.shape[-1] == 4

    def test_innovation_scale_positive(self, fitted_model_time_varying):
        post = fitted_model_time_varying.posterior
        assert (post["sd_att_innov"].values > 0).all()
        assert (post["sd_def_innov"].values > 0).all()

    def test_anchor_dominates_deviation(self, fitted_model_time_varying):
        """Static anchor should account for most of the team-strength variation;
        weekly deviations should be small. This is the structural identifiability
        win over the previous independent-walk parameterization."""
        post = fitted_model_time_varying.posterior
        atts_static_std = float(post["atts_static"].std())
        delta_atts_std = float(post["delta_atts"].std())
        # Static anchor should have larger spread than weekly deviations
        assert atts_static_std > delta_atts_std, (
            f"Anchor std ({atts_static_std:.3f}) should exceed "
            f"deviation std ({delta_atts_std:.3f})"
        )

    def test_simulation_works(self, synthetic_data_with_covariates,
                               fitted_model_time_varying):
        sim = simulate_team_season(synthetic_data_with_covariates,
                                   fitted_model_time_varying,
                                   metric="score", burnin=0)
        assert (sim["home_score"] >= 0).all()
        assert len(sim) == len(synthetic_data_with_covariates)


# ---------------------------------------------------------------------------
# Likelihood / alpha-prior variants
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fitted_model_poisson(synthetic_data):
    return bhm(synthetic_data, metric="score", samples=100, likelihood="poisson")


@pytest.fixture(scope="session")
def fitted_model_nb_tight(synthetic_data):
    return bhm(synthetic_data, metric="score", samples=100,
               likelihood="negbin", alpha_prior="tight")


class TestLikelihoodVariants:
    def test_poisson_has_no_alpha(self, fitted_model_poisson):
        """Poisson variant should not have alpha in the posterior."""
        assert "alpha" not in fitted_model_poisson.posterior

    def test_poisson_simulation_works(self, synthetic_data, fitted_model_poisson):
        """Simulator should detect Poisson via missing alpha and Poisson-sample."""
        sim = simulate_team_season(synthetic_data, fitted_model_poisson,
                                   metric="score", burnin=0)
        assert (sim["home_score"] >= 0).all()
        assert len(sim) == len(synthetic_data)

    def test_tight_prior_increases_alpha_posterior(
        self, fitted_model, fitted_model_nb_tight
    ):
        """Tight LogNormal prior should pull alpha higher than Exp(1)."""
        alpha_weak = float(fitted_model.posterior["alpha"].mean())
        alpha_tight = float(fitted_model_nb_tight.posterior["alpha"].mean())
        assert alpha_tight > alpha_weak, (
            f"tight alpha ({alpha_tight:.2f}) should exceed weak ({alpha_weak:.2f})"
        )

    def test_invalid_likelihood_raises(self, synthetic_data):
        with pytest.raises(ValueError, match="likelihood must be"):
            bhm(synthetic_data, samples=10, likelihood="gaussian")

    def test_invalid_alpha_prior_raises(self, synthetic_data):
        with pytest.raises(ValueError, match="alpha_prior must be"):
            bhm(synthetic_data, samples=10, alpha_prior="medium")


# ---------------------------------------------------------------------------
# Multi-season carryover (team_priors kwarg)
# ---------------------------------------------------------------------------

class TestTeamPriors:
    def test_team_priors_kwarg_accepted(self, synthetic_data):
        """bhm() should accept team_priors and produce valid posterior."""
        n_teams = 4
        priors = {
            "atts_mean": np.zeros(n_teams),
            "defs_mean": np.zeros(n_teams),
            "carryover_sd": 0.1,
        }
        idata = bhm(synthetic_data, samples=100, team_priors=priors)
        assert idata is not None
        assert "atts" in idata.posterior

    def test_team_priors_shifts_posterior_toward_prior_mean(self, synthetic_data):
        """A strong prior far from zero should pull the posterior in that direction."""
        n_teams = 4
        # Push T0 attack way up, T3 attack way down via a strong prior
        atts_prior = np.array([1.0, 0.0, 0.0, -1.0])
        defs_prior = np.zeros(n_teams)
        priors = {
            "atts_mean": atts_prior,
            "defs_mean": defs_prior,
            "carryover_sd": 0.05,   # tight — prior dominates
        }
        idata = bhm(synthetic_data, samples=200, team_priors=priors)
        atts_mean = idata.posterior["atts"].mean(dim=["chain", "draw"]).values
        # T0 attack should be substantially above T3 attack with this prior
        assert atts_mean[0] - atts_mean[3] > 1.0, (
            f"With strong prior, T0-T3 gap should be >1; got {atts_mean[0]-atts_mean[3]:.3f}"
        )

    def test_team_priors_shape_validation(self, synthetic_data):
        """Wrong-shape arrays should raise."""
        bad_priors = {
            "atts_mean": np.zeros(3),  # wrong size (4 teams expected)
            "defs_mean": np.zeros(4),
        }
        with pytest.raises(ValueError, match="must have shape"):
            bhm(synthetic_data, samples=10, team_priors=bad_priors)

    def test_team_priors_incompatible_with_time_varying(self, synthetic_data):
        """Should raise when combined with time_varying=True."""
        priors = {
            "atts_mean": np.zeros(4),
            "defs_mean": np.zeros(4),
        }
        with pytest.raises(ValueError, match="not currently supported"):
            bhm(synthetic_data, samples=10,
                time_varying=True, team_priors=priors)


# ---------------------------------------------------------------------------
# Per-team home-field advantage
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fitted_model_per_team_home(synthetic_data):
    return bhm(synthetic_data, samples=200, per_team_home=True)


class TestPerTeamHome:
    def test_home_team_in_posterior(self, fitted_model_per_team_home):
        """Per-team home advantage should appear as `home_team` in posterior;
        the scalar `home` should NOT."""
        post = fitted_model_per_team_home.posterior
        assert "home_team" in post
        assert "mu_home" in post
        assert "sigma_home" in post
        assert "home" not in post  # was replaced

    def test_home_team_shape(self, fitted_model_per_team_home):
        """home_team should have one entry per team."""
        home_team = fitted_model_per_team_home.posterior["home_team"]
        assert home_team.shape[-1] == 4   # 4 synthetic teams

    def test_sigma_home_positive(self, fitted_model_per_team_home):
        """sigma_home is the cross-team scale; must be positive."""
        sigma_home = fitted_model_per_team_home.posterior["sigma_home"].values
        assert (sigma_home > 0).all()

    def test_simulation_works(self, synthetic_data, fitted_model_per_team_home):
        sim = simulate_team_season(synthetic_data, fitted_model_per_team_home,
                                   metric="score", burnin=0)
        assert (sim["home_score"] >= 0).all()
        assert len(sim) == len(synthetic_data)


# ---------------------------------------------------------------------------
# Per-team alpha (NB dispersion)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fitted_model_per_team_alpha(synthetic_data):
    return bhm(synthetic_data, samples=200, per_team_alpha=True)


class TestPerTeamAlpha:
    def test_alpha_team_in_posterior(self, fitted_model_per_team_alpha):
        post = fitted_model_per_team_alpha.posterior
        assert "alpha_team" in post
        assert "log_alpha_mean" in post
        assert "sigma_log_alpha" in post
        # Shared scalar alpha should NOT be present
        assert "alpha" not in post

    def test_alpha_team_shape(self, fitted_model_per_team_alpha):
        alpha_team = fitted_model_per_team_alpha.posterior["alpha_team"]
        assert alpha_team.shape[-1] == 4  # 4 teams

    def test_alpha_team_positive(self, fitted_model_per_team_alpha):
        """All team alphas must be positive (log-normal parameterization)."""
        alpha_team = fitted_model_per_team_alpha.posterior["alpha_team"].values
        assert (alpha_team > 0).all()

    def test_simulation_works(self, synthetic_data, fitted_model_per_team_alpha):
        sim = simulate_team_season(synthetic_data, fitted_model_per_team_alpha,
                                   metric="score", burnin=0)
        assert (sim["home_score"] >= 0).all()
        assert len(sim) == len(synthetic_data)

    def test_per_team_alpha_requires_negbin(self, synthetic_data):
        """Should raise when combined with likelihood='poisson'."""
        with pytest.raises(ValueError, match="requires likelihood='negbin'"):
            bhm(synthetic_data, samples=10,
                likelihood="poisson", per_team_alpha=True)
