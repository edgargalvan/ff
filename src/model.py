"""
Hierarchical Bayesian model for NFL game outcome prediction.
Ported from PyMC3/Theano to PyMC v5/PyTensor.

The model estimates per-team attack and defense strength parameters,
then simulates game outcomes via Poisson draws from the posterior.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt


def bhm(df, metric="score", K=1, nu=3.0, sigma=2.5, samples=1000):
    """
    Fit a hierarchical Bayesian model to game data.

    The model assumes home/away scores follow Poisson distributions with
    log-linear rates determined by team-specific attack/defense parameters
    and a home-field advantage term.

    Args:
        df: DataFrame with columns: i_home, i_away, home_{metric}, away_{metric}
        metric: which stat to model ('score', 'passing_yds', etc.)
        K: divisor for the metric (e.g. K=7 to model in units of touchdowns)
        nu: degrees of freedom for HalfStudentT priors
        sigma: scale for HalfStudentT priors
        samples: number of MCMC samples to draw

    Returns:
        ArviZ InferenceData object
    """
    n_teams = int(max(df["i_home"].max(), df["i_away"].max()) + 1)
    home_metric = f"home_{metric}"
    away_metric = f"away_{metric}"

    with pm.Model() as model:
        # Global parameters (weakly informative priors)
        home = pm.Normal("home", mu=0, sigma=5)
        intercept = pm.Normal("intercept", mu=0, sigma=5)
        sd_att = pm.HalfStudentT("sd_att", nu=nu, sigma=sigma)
        sd_def = pm.HalfStudentT("sd_def", nu=nu, sigma=sigma)

        # Team-specific parameters (centered)
        atts_star = pm.Normal("atts_star", mu=0, sigma=sd_att, shape=n_teams)
        defs_star = pm.Normal("defs_star", mu=0, sigma=sd_def, shape=n_teams)

        # Sum-to-zero constraint
        atts = pm.Deterministic("atts", atts_star - pt.mean(atts_star))
        defs = pm.Deterministic("defs", defs_star - pt.mean(defs_star))

        # Log-linear model for scoring rates
        home_theta = pt.exp(intercept + home + atts[df["i_home"].values] + defs[df["i_away"].values])
        away_theta = pt.exp(intercept + atts[df["i_away"].values] + defs[df["i_home"].values])

        # Poisson likelihood
        pm.Poisson("home_obs", mu=home_theta, observed=df[home_metric].values / K)
        pm.Poisson("away_obs", mu=away_theta, observed=df[away_metric].values / K)

        # Fit — use default NUTS initialization (jitter+adapt_diag)
        # MAP init can be unstable with Flat priors, so we let PyMC handle it
        idata = pm.sample(samples)

    return idata


def simulate_team_season(df, idata, metric="score", K=1, burnin=100):
    """
    Simulate one season using a random draw from the posterior.

    Args:
        df: DataFrame with game matchups (i_home, i_away)
        idata: ArviZ InferenceData from bhm()
        metric: stat being modeled
        K: divisor used during fitting
        burnin: minimum sample index to draw from

    Returns:
        DataFrame with simulated home/away scores
    """
    home_metric = f"home_{metric}"
    away_metric = f"away_{metric}"

    # Get posterior samples
    posterior = idata.posterior
    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]

    # Random draw from posterior (skip burnin)
    chain = np.random.randint(0, n_chains)
    draw = np.random.randint(burnin, n_draws)

    atts = posterior["atts"].values[chain, draw, :]
    defs = posterior["defs"].values[chain, draw, :]
    home_adv = float(posterior["home"].values[chain, draw])
    intercept = float(posterior["intercept"].values[chain, draw])

    season = df.copy()

    # Compute scoring rates
    home_theta = np.exp(intercept + home_adv + atts[season["i_home"].values] + defs[season["i_away"].values])
    away_theta = np.exp(intercept + atts[season["i_away"].values] + defs[season["i_home"].values])

    # Simulate scores
    season[home_metric] = K * np.random.poisson(home_theta)
    season[away_metric] = K * np.random.poisson(away_theta)

    return season


def simulate_team_seasons(df, idata, nsims=1000, metric="score", K=1, burnin=100):
    """
    Run multiple season simulations and aggregate results by team.

    Returns:
        DataFrame with columns: team, {metric}, iteration
    """
    dfs = []
    for i in range(nsims):
        season = simulate_team_season(df, idata, metric=metric, K=K, burnin=burnin)
        table = create_team_season_table(season, metric)
        table["iteration"] = i
        dfs.append(table)
    return pd.concat(dfs, ignore_index=True)


def create_team_season_table(season, metric="score"):
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


def predictions(df, simuls, teams, nsims=1000, metric="score"):
    """
    Generate prediction intervals from simulations.

    Args:
        df: DataFrame of actual game results
        simuls: DataFrame from simulate_team_seasons()
        teams: team mapping DataFrame
        nsims: number of simulations that were run
        metric: stat being modeled

    Returns:
        hdis DataFrame with median, 10th/90th percentile, and actuals per team
    """
    # Observed results (aggregate actual scores by team)
    df_observed = create_team_season_table(df, metric)
    observed_agg = df_observed.groupby("team")[metric].sum().reset_index()

    # Compute HDIs from simulations (aggregate simulated scores by team per iteration)
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


def plot_hdis(hdis, metric="score"):
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
