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
        covariates: list[str] | None = None,
        likelihood: str = "negbin",
        alpha_prior: str = "weak",
        team_priors: dict | None = None,
        per_team_home: bool = False) -> az.InferenceData:
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
        likelihood: 'negbin' (default) or 'poisson'. Poisson is the Baio/Blangiardo
                    classic. Negative Binomial adds a learned dispersion parameter
                    suitable for overdispersed counts.
        alpha_prior: only meaningful when likelihood='negbin'. 'weak' (default,
                     Exponential(1) — what we shipped with) or 'tight' (LogNormal
                     centered around alpha=15). 'tight' was motivated by the
                     calibration inspection finding that 'weak' produces a
                     predictive distribution ~10% wider than observed scores.
        team_priors: optional dict for multi-season carryover. Shape:
                     {"atts_mean": np.ndarray of length n_teams,
                      "defs_mean": np.ndarray of length n_teams,
                      "carryover_sd": float (default 0.1)}.
                     When provided, team-strength priors are centered at the
                     supplied means rather than zero, and the cross-team scale
                     is `carryover_sd` instead of the learned `sd_att/sd_def`.
                     Not currently combinable with `time_varying=True`.
        per_team_home: if True, replace the single shared `home` scalar with a
                       hierarchical per-team home-field advantage:
                           mu_home   ~ Normal(0, 5)
                           sigma_home ~ HalfStudentT(3, 0.3)
                           home_team[t] = mu_home + sigma_home * raw[t]
                       Each team's home games use that team's home_team[t]
                       additive log-rate term.

    Returns:
        ArviZ InferenceData object
    """
    if likelihood not in {"negbin", "poisson"}:
        raise ValueError(f"likelihood must be 'negbin' or 'poisson', got {likelihood!r}")
    if alpha_prior not in {"weak", "tight"}:
        raise ValueError(f"alpha_prior must be 'weak' or 'tight', got {alpha_prior!r}")
    if team_priors is not None and time_varying:
        raise ValueError("team_priors is not currently supported with time_varying=True")

    n_teams = int(max(df["i_home"].max(), df["i_away"].max()) + 1)
    home_metric = f"home_{metric}"
    away_metric = f"away_{metric}"

    with pm.Model() as model:
        # Global parameters
        intercept = pm.Normal("intercept", mu=0, sigma=5)
        sd_att = pm.HalfStudentT("sd_att", nu=nu, sigma=sigma)
        sd_def = pm.HalfStudentT("sd_def", nu=nu, sigma=sigma)

        # Home-field advantage: scalar (default) or per-team hierarchical.
        if per_team_home:
            mu_home = pm.Normal("mu_home", mu=0, sigma=5)
            sigma_home = pm.HalfStudentT("sigma_home", nu=3, sigma=0.3)
            home_raw = pm.Normal("home_raw", mu=0, sigma=1, shape=n_teams)
            home_team = pm.Deterministic("home_team",
                                          mu_home + sigma_home * home_raw)
            # Per-game home advantage indexed by home team
            home_term = home_team[df["i_home"].values]
        else:
            home = pm.Normal("home", mu=0, sigma=5)
            home_term = home  # scalar; broadcasts when added below

        if time_varying:
            atts, defs = _build_time_varying_params(df, n_teams, sd_att, sd_def)
        else:
            atts, defs = _build_static_params(
                n_teams, sd_att, sd_def, team_priors=team_priors
            )

        # Log-linear model for scoring rates
        log_home = intercept + home_term + atts[df["i_home"].values] + defs[df["i_away"].values]
        log_away = intercept + atts[df["i_away"].values] + defs[df["i_home"].values]

        # Add covariates
        if covariates:
            log_home, log_away = _add_covariates(df, covariates, log_home, log_away)

        home_theta = pt.exp(log_home)
        away_theta = pt.exp(log_away)

        if likelihood == "poisson":
            # Baio/Blangiardo classic — no dispersion parameter.
            pm.Poisson("home_obs", mu=home_theta,
                       observed=df[home_metric].values / K)
            pm.Poisson("away_obs", mu=away_theta,
                       observed=df[away_metric].values / K)
        else:
            # Negative Binomial with learned dispersion alpha.
            if alpha_prior == "weak":
                # Default since this project's inception: very loose Exp(1).
                alpha = pm.Exponential("alpha", lam=1.0)
            else:
                # Informed prior. LogNormal(mu=2.7, sigma=0.4) has:
                #   median exp(2.7) ≈ 14.9
                #   ~90% mass in [7, 28]
                # This places the bulk of prior mass on alpha values that produce
                # predictive standard deviations close to observed NFL game variance.
                alpha = pm.LogNormal("alpha", mu=2.7, sigma=0.4)

            pm.NegativeBinomial("home_obs", mu=home_theta, alpha=alpha,
                                observed=df[home_metric].values / K)
            pm.NegativeBinomial("away_obs", mu=away_theta, alpha=alpha,
                                observed=df[away_metric].values / K)

        idata = pm.sample(samples)

    # Stash the configuration so downstream simulators can detect it
    idata.attrs["likelihood"] = likelihood
    idata.attrs["alpha_prior"] = alpha_prior

    return idata


def _build_static_params(n_teams, sd_att, sd_def, team_priors=None):
    """Non-centered static team parameters.

    If `team_priors` is provided (dict with `atts_mean`, `defs_mean` arrays
    of length n_teams and optional `carryover_sd` scalar), the prior on each
    team's strength is centered at the supplied mean instead of zero, and
    the cross-team scale is `carryover_sd` instead of the learned `sd_att`/
    `sd_def`. This implements multi-season carryover: prior-year posterior
    means inform the current-year prior.
    """
    atts_raw = pm.Normal("atts_raw", mu=0, sigma=1, shape=n_teams)
    defs_raw = pm.Normal("defs_raw", mu=0, sigma=1, shape=n_teams)

    if team_priors is not None:
        carryover_sd = team_priors.get("carryover_sd", 0.1)
        atts_prior_mean = np.asarray(team_priors["atts_mean"], dtype=float)
        defs_prior_mean = np.asarray(team_priors["defs_mean"], dtype=float)
        if atts_prior_mean.shape != (n_teams,) or defs_prior_mean.shape != (n_teams,):
            raise ValueError(
                f"team_priors arrays must have shape ({n_teams},); got "
                f"atts={atts_prior_mean.shape}, defs={defs_prior_mean.shape}"
            )
        atts_star = pm.Deterministic(
            "atts_star", atts_prior_mean + carryover_sd * atts_raw
        )
        defs_star = pm.Deterministic(
            "defs_star", defs_prior_mean + carryover_sd * defs_raw
        )
    else:
        atts_star = pm.Deterministic("atts_star", sd_att * atts_raw)
        defs_star = pm.Deterministic("defs_star", sd_def * defs_raw)

    atts = pm.Deterministic("atts", atts_star - pt.mean(atts_star))
    defs = pm.Deterministic("defs", defs_star - pt.mean(defs_star))
    return atts, defs


def _build_time_varying_params(df, n_teams, sd_att, sd_def):
    """
    Time-varying team parameters with hierarchical anchor (AR(1) state-space).

    Each team has a static intercept (the season-long anchor) plus a
    GaussianRandomWalk deviation centered at zero. This is the canonical
    Glickman-Stern-style structure: the static term is well-identified by
    pooling across all weeks of data; the deviation is constrained tight so
    it captures real week-to-week change (form, injuries) without absorbing
    sampling noise.

    This replaces the previous independent-per-team random walk, which
    suffered from identifiability collapse (see status doc §6 post-mortem).

    Posterior variables introduced:
        atts_static_raw, defs_static_raw  — non-centered N(0,1)
        atts_static, defs_static          — sum-to-zero anchors
        sd_att_innov, sd_def_innov        — shared weekly drift scales
        delta_atts, delta_defs            — RW deviations from anchor
        atts, defs                        — combined (n_weeks, n_teams)
    """
    weeks = sorted(df["week"].unique())
    n_weeks = len(weeks)
    week_to_idx = {w: i for i, w in enumerate(weeks)}
    week_indices = df["week"].map(week_to_idx).values

    # Static anchor — non-centered, sum-to-zero
    atts_static_raw = pm.Normal("atts_static_raw", mu=0, sigma=1, shape=n_teams)
    defs_static_raw = pm.Normal("defs_static_raw", mu=0, sigma=1, shape=n_teams)
    atts_static_unc = sd_att * atts_static_raw
    defs_static_unc = sd_def * defs_static_raw
    atts_static = pm.Deterministic(
        "atts_static", atts_static_unc - pt.mean(atts_static_unc)
    )
    defs_static = pm.Deterministic(
        "defs_static", defs_static_unc - pt.mean(defs_static_unc)
    )

    # Tight innovation scale — week-to-week drift is small
    sd_att_innov = pm.HalfNormal("sd_att_innov", sigma=0.05)
    sd_def_innov = pm.HalfNormal("sd_def_innov", sigma=0.05)

    # AR(1) deviations from the static anchor, anchored at 0
    delta_atts = pm.GaussianRandomWalk(
        "delta_atts", sigma=sd_att_innov,
        init_dist=pm.Normal.dist(0, 0.01),
        shape=(n_weeks, n_teams),
    )
    delta_defs = pm.GaussianRandomWalk(
        "delta_defs", sigma=sd_def_innov,
        init_dist=pm.Normal.dist(0, 0.01),
        shape=(n_weeks, n_teams),
    )

    # Combined: static anchor (broadcast over weeks) + RW deviation
    atts = pm.Deterministic("atts", atts_static[None, :] + delta_atts)
    defs = pm.Deterministic("defs", defs_static[None, :] + delta_defs)

    # Index into the right week for each game
    atts_indexed = atts[week_indices, :]
    defs_indexed = defs[week_indices, :]

    # For each game, select the home/away team's strength at that week
    game_indices = np.arange(len(df))
    atts_home = atts_indexed[game_indices, df["i_home"].values]
    atts_away = atts_indexed[game_indices, df["i_away"].values]
    defs_home = defs_indexed[game_indices, df["i_home"].values]
    defs_away = defs_indexed[game_indices, df["i_away"].values]

    # Wrap in stateful indexer so the caller can use atts[df["i_home"]...] syntax
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

    # Check if model used time-varying params (now: hierarchical-anchor AR(1))
    is_time_varying = "delta_atts" in posterior

    if is_time_varying:
        # `atts` is a deterministic with shape (n_weeks, n_teams) =
        # atts_static[team] + delta_atts[week, team].
        # For prediction we use the last training week's combined estimate
        # (static term dominates; delta provides recent-form adjustment).
        atts_all = posterior["atts"].values[chain, draw, :, :]
        defs_all = posterior["defs"].values[chain, draw, :, :]
        atts = atts_all[-1, :]
        defs = defs_all[-1, :]
    else:
        atts = posterior["atts"].values[chain, draw, :]
        defs = posterior["defs"].values[chain, draw, :]

    intercept = float(posterior["intercept"].values[chain, draw])

    # Home-field advantage: scalar (default) or per-team vector.
    is_per_team_home = "home_team" in posterior
    if is_per_team_home:
        home_team_vec = posterior["home_team"].values[chain, draw, :]
        home_term = home_team_vec  # indexed below per-game
    else:
        home_adv = float(posterior["home"].values[chain, draw])
        home_term = None  # scalar use below

    # Detect likelihood by presence/absence of alpha in posterior.
    is_negbin = "alpha" in posterior
    alpha = float(posterior["alpha"].values[chain, draw]) if is_negbin else None

    season = df.copy()

    # Compute log-linear rates
    if is_per_team_home:
        log_home = (intercept
                    + home_term[season["i_home"].values]
                    + atts[season["i_home"].values]
                    + defs[season["i_away"].values])
    else:
        log_home = (intercept + home_adv
                    + atts[season["i_home"].values]
                    + defs[season["i_away"].values])
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

    if is_negbin:
        # Negative Binomial sampling with numpy's (n, p) parameterization.
        def _draw(mu, a):
            p = a / (a + mu)
            return K * np.random.negative_binomial(a, p)
        season[home_metric] = _draw(home_theta, alpha)
        season[away_metric] = _draw(away_theta, alpha)
    else:
        # Poisson sampling.
        season[home_metric] = K * np.random.poisson(home_theta)
        season[away_metric] = K * np.random.poisson(away_theta)

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
