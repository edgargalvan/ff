"""
Hierarchical Bayesian model for NFL game outcome prediction.

Uses a Negative Binomial log-linear model with non-centered parameterization
to estimate per-team attack and defense strengths, then simulates game
outcomes by drawing from the posterior.

Base model:
    home_score ~ NegBin(mu=exp(intercept + home + atts[home] + defs[away] + X*beta), alpha)
    away_score ~ NegBin(mu=exp(intercept + atts[away] + defs[home] + X*beta), alpha)

Options:
    time_varying=True:  atts/defs evolve per-week via GaussianRandomWalk
    covariates=[...]:   add game-level predictors (rest, weather, etc.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Covariates that affect both teams symmetrically (added to both home and away theta)
SYMMETRIC_COVARIATES = ["temp_std", "wind_std", "is_indoor", "div_game"]

# Covariates that only affect the home team's rate
HOME_COVARIATES = ["rest_advantage", "home_short_week"]

# All available covariates
ALL_COVARIATES = SYMMETRIC_COVARIATES + HOME_COVARIATES


def bhm(df: pd.DataFrame, metric: str = "score", K: int = 1,
        nu: float = 3.0, sigma: float = 2.5, samples: int = 1000,
        time_varying: bool = False,
        covariates: list[str] | None = None) -> az.InferenceData:
    """
    Fit a hierarchical Bayesian model to game data.

    Args:
        df: DataFrame with columns: i_home, i_away, week, home_{metric}, away_{metric}
        metric: which stat to model ('score', 'passing_yds', etc.)
        K: divisor for the metric (e.g. K=7 to model in units of touchdowns)
        nu: degrees of freedom for HalfStudentT priors
        sigma: scale for HalfStudentT priors
        samples: number of MCMC samples to draw
        time_varying: if True, team strengths evolve week-to-week via GaussianRandomWalk
        covariates: list of covariate column names to include, or None for no covariates.
                    Use ALL_COVARIATES for all available.

    Returns:
        ArviZ InferenceData object
    """
    n_teams = int(max(df["i_home"].max(), df["i_away"].max()) + 1)
    home_metric = f"home_{metric}"
    away_metric = f"away_{metric}"

    with pm.Model() as model:
        # Global parameters
        home = pm.Normal("home", mu=0, sigma=5)
        intercept = pm.Normal("intercept", mu=0, sigma=5)
        sd_att = pm.HalfStudentT("sd_att", nu=nu, sigma=sigma)
        sd_def = pm.HalfStudentT("sd_def", nu=nu, sigma=sigma)

        if time_varying:
            atts, defs = _build_time_varying_params(df, n_teams, sd_att, sd_def)
        else:
            atts, defs = _build_static_params(n_teams, sd_att, sd_def)

        # Log-linear model for scoring rates
        log_home = intercept + home + atts[df["i_home"].values] + defs[df["i_away"].values]
        log_away = intercept + atts[df["i_away"].values] + defs[df["i_home"].values]

        # Add covariates
        if covariates:
            log_home, log_away = _add_covariates(df, covariates, log_home, log_away)

        home_theta = pt.exp(log_home)
        away_theta = pt.exp(log_away)

        # Overdispersion parameter for Negative Binomial
        alpha = pm.Exponential("alpha", lam=1.0)

        # Negative Binomial likelihood
        pm.NegativeBinomial("home_obs", mu=home_theta, alpha=alpha,
                            observed=df[home_metric].values / K)
        pm.NegativeBinomial("away_obs", mu=away_theta, alpha=alpha,
                            observed=df[away_metric].values / K)

        idata = pm.sample(samples)

    return idata


def _build_static_params(n_teams, sd_att, sd_def):
    """Non-centered static team parameters."""
    atts_raw = pm.Normal("atts_raw", mu=0, sigma=1, shape=n_teams)
    defs_raw = pm.Normal("defs_raw", mu=0, sigma=1, shape=n_teams)
    atts_star = pm.Deterministic("atts_star", sd_att * atts_raw)
    defs_star = pm.Deterministic("defs_star", sd_def * defs_raw)

    atts = pm.Deterministic("atts", atts_star - pt.mean(atts_star))
    defs = pm.Deterministic("defs", defs_star - pt.mean(defs_star))
    return atts, defs


def _build_time_varying_params(df, n_teams, sd_att, sd_def):
    """
    Time-varying team parameters via GaussianRandomWalk.

    Each team's attack and defense strength evolves week-to-week.
    The model indexes into the appropriate week for each game.
    """
    weeks = sorted(df["week"].unique())
    n_weeks = len(weeks)
    week_to_idx = {w: i for i, w in enumerate(weeks)}
    week_indices = df["week"].map(week_to_idx).values

    # Innovation scale (how much strength can change per week)
    sd_att_innov = pm.HalfNormal("sd_att_innov", sigma=0.1)
    sd_def_innov = pm.HalfNormal("sd_def_innov", sigma=0.1)

    # GaussianRandomWalk: shape (n_weeks, n_teams)
    # Each team gets an independent random walk over weeks
    atts_walk = pm.GaussianRandomWalk(
        "atts_walk", sigma=sd_att_innov, init_dist=pm.Normal.dist(0, sd_att),
        shape=(n_weeks, n_teams),
    )
    defs_walk = pm.GaussianRandomWalk(
        "defs_walk", sigma=sd_def_innov, init_dist=pm.Normal.dist(0, sd_def),
        shape=(n_weeks, n_teams),
    )

    # Sum-to-zero constraint per week
    atts_centered = pm.Deterministic(
        "atts", atts_walk - pt.mean(atts_walk, axis=1, keepdims=True)
    )
    defs_centered = pm.Deterministic(
        "defs", defs_walk - pt.mean(defs_walk, axis=1, keepdims=True)
    )

    # Index into the right week for each game
    atts = atts_centered[week_indices, :]
    defs = defs_centered[week_indices, :]

    # For each game, select the home/away team's strength at that week
    game_indices = np.arange(len(df))
    atts_home = atts[game_indices, df["i_home"].values]
    atts_away = atts[game_indices, df["i_away"].values]
    defs_home = defs[game_indices, df["i_home"].values]
    defs_away = defs[game_indices, df["i_away"].values]

    # Return as indexable-like objects (just the game-level values)
    # We wrap in a container so the caller can use atts[df["i_home"].values] syntax
    return _GameIndexed(atts_home, atts_away), _GameIndexed(defs_home, defs_away)


class _GameIndexed:
    """Helper to let time-varying params be indexed like static ones."""
    def __init__(self, home_vals, away_vals):
        self._home = home_vals
        self._away = away_vals
        self._is_home_call = True

    def __getitem__(self, idx):
        # First call is home, second is away (matches the bhm() call pattern)
        if self._is_home_call:
            self._is_home_call = False
            return self._home
        else:
            self._is_home_call = True
            return self._away


def _add_covariates(df, covariates, log_home, log_away):
    """Add covariate effects to log-linear predictors."""
    for cov in covariates:
        if cov not in df.columns:
            continue

        values = df[cov].values.astype(float)
        # Check for NaN
        if np.any(np.isnan(values)):
            values = np.nan_to_num(values, nan=0.0)

        beta = pm.Normal(f"beta_{cov}", mu=0, sigma=1)

        if cov in HOME_COVARIATES:
            # Only affects home team rate
            log_home = log_home + beta * values
        else:
            # Affects both teams (environmental factor)
            log_home = log_home + beta * values
            log_away = log_away + beta * values

    return log_home, log_away


def simulate_team_season(df: pd.DataFrame, idata: az.InferenceData,
                         metric: str = "score", K: int = 1, burnin: int = 100,
                         covariates: list[str] | None = None) -> pd.DataFrame:
    """
    Simulate one season using a random draw from the posterior.

    Args:
        df: DataFrame with game matchups (i_home, i_away, and covariate columns)
        idata: ArviZ InferenceData from bhm()
        metric: stat being modeled
        K: divisor used during fitting
        burnin: minimum sample index to draw from
        covariates: list of covariates used during fitting (must match)

    Returns:
        DataFrame with simulated home/away scores
    """
    home_metric = f"home_{metric}"
    away_metric = f"away_{metric}"

    posterior = idata.posterior
    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]

    chain = np.random.randint(0, n_chains)
    draw = np.random.randint(burnin, n_draws)

    # Check if model used time-varying params
    is_time_varying = "atts_walk" in posterior

    if is_time_varying:
        # atts has shape (n_weeks, n_teams) — use last week's values for prediction
        atts_all = posterior["atts"].values[chain, draw, :, :]  # (n_weeks, n_teams)
        defs_all = posterior["defs"].values[chain, draw, :, :]
        # Use last available week's strength estimates
        atts = atts_all[-1, :]
        defs = defs_all[-1, :]
    else:
        atts = posterior["atts"].values[chain, draw, :]
        defs = posterior["defs"].values[chain, draw, :]

    home_adv = float(posterior["home"].values[chain, draw])
    intercept = float(posterior["intercept"].values[chain, draw])
    alpha = float(posterior["alpha"].values[chain, draw])

    season = df.copy()

    # Compute log-linear rates
    log_home = intercept + home_adv + atts[season["i_home"].values] + defs[season["i_away"].values]
    log_away = intercept + atts[season["i_away"].values] + defs[season["i_home"].values]

    # Add covariate effects
    if covariates:
        for cov in covariates:
            if cov not in season.columns:
                continue
            beta_key = f"beta_{cov}"
            if beta_key not in posterior:
                continue
            beta = float(posterior[beta_key].values[chain, draw])
            values = season[cov].values.astype(float)
            values = np.nan_to_num(values, nan=0.0)

            if cov in HOME_COVARIATES:
                log_home = log_home + beta * values
            else:
                log_home = log_home + beta * values
                log_away = log_away + beta * values

    home_theta = np.exp(log_home)
    away_theta = np.exp(log_away)

    # Simulate via Negative Binomial
    def nb_draw(mu, a):
        p = a / (a + mu)
        return K * np.random.negative_binomial(a, p)

    season[home_metric] = nb_draw(home_theta, alpha)
    season[away_metric] = nb_draw(away_theta, alpha)

    return season


def simulate_team_seasons(df: pd.DataFrame, idata: az.InferenceData,
                          nsims: int = 1000, metric: str = "score",
                          K: int = 1, burnin: int = 100,
                          covariates: list[str] | None = None) -> pd.DataFrame:
    """
    Run multiple season simulations and aggregate results by team.

    Returns:
        DataFrame with columns: team, {metric}, iteration
    """
    dfs = []
    for i in range(nsims):
        season = simulate_team_season(df, idata, metric=metric, K=K,
                                      burnin=burnin, covariates=covariates)
        table = create_team_season_table(season, metric)
        table["iteration"] = i
        dfs.append(table)
    return pd.concat(dfs, ignore_index=True)


def create_team_season_table(season: pd.DataFrame, metric: str = "score") -> pd.DataFrame:
    """Aggregate simulated game results into per-team totals."""
    home_metric = f"home_{metric}"
    away_metric = f"away_{metric}"

    home = season[["home_team", home_metric]].rename(
        columns={"home_team": "team", home_metric: metric}
    )
    away = season[["away_team", away_metric]].rename(
        columns={"away_team": "team", away_metric: metric}
    )
    return pd.concat([home, away], axis=0, ignore_index=True)


def predictions(df: pd.DataFrame, simuls: pd.DataFrame, teams: pd.DataFrame,
                nsims: int = 1000, metric: str = "score") -> pd.DataFrame:
    """
    Generate prediction intervals from simulations.

    Returns:
        hdis DataFrame with median, 10th/90th percentile, and actuals per team
    """
    df_observed = create_team_season_table(df, metric)
    observed_agg = df_observed.groupby("team")[metric].sum().reset_index()

    sim_agg = simuls.groupby(["team", "iteration"])[metric].sum().reset_index()
    g = sim_agg.groupby("team")
    hdis = pd.DataFrame({
        "metric_lower": g[metric].quantile(0.1),
        "metric_median": g[metric].median(),
        "metric_upper": g[metric].quantile(0.9),
    })

    hdis = hdis.merge(observed_agg, left_index=True, right_on="team")
    hdis["relative_upper"] = hdis["metric_upper"] - hdis["metric_median"]
    hdis["relative_lower"] = hdis["metric_median"] - hdis["metric_lower"]
    hdis = hdis.sort_values(by=metric).reset_index(drop=True)
    hdis["x"] = hdis.index + 0.5

    return hdis


def plot_hdis(hdis: pd.DataFrame, metric: str = "score") -> tuple[Figure, Axes]:
    """Plot actual values vs simulated 90% credible intervals by team."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(hdis["x"], hdis[metric], c="steelblue", zorder=10,
               label=f"Actual {metric}")
    ax.errorbar(hdis["x"], hdis["metric_median"],
                yerr=hdis[["relative_lower", "relative_upper"]].values.T,
                fmt="s", c="coral", label="Simulated 90% interval")

    ax.set_title(f"Actual {metric} and 90% Interval from Simulations, by Team")
    ax.set_xlabel("Team")
    ax.set_ylabel(metric.capitalize())
    ax.legend()

    ax.set_xticks(hdis["x"])
    ax.set_xticklabels(hdis["team"].values, rotation=45, ha="right")
    fig.tight_layout()

    return fig, ax
