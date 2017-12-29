import math
import numpy as np
import pandas as pd
import theano.tensor as tt
import pymc3 as pm
from scipy import optimize
import matplotlib.pyplot as plt


def spread(df, week, num_simulations):

    # pull week to simulate
    df_test = df[df['week'] == week]
    df = df[df['week'] != week]

    # priors
    model = pm.Model()
    with pm.Model() as model:
        # global model parameters
        home = pm.Normal('home',      0, tau=.0001)
        tau_att = pm.Gamma('tau_att',   .1, .1)
        tau_def = pm.Gamma('tau_def',   .1, .1)
        intercept = pm.Normal('intercept', 0, tau=.0001)

        # team-specific parameters
        atts_star = pm.Normal('atts_star',
                              mu=0,
                              tau=tau_att,
                              shape=32)
        defs_star = pm.Normal('defs_star',
                              mu=0,
                              tau=tau_def,
                              shape=32)

    # constraints
    with model:
        atts = pm.Deterministic('atts', atts_star - tt.mean(atts_star))
        defs = pm.Deterministic('defs', defs_star - tt.mean(defs_star))
        home_theta = tt.exp(intercept + home + atts[df.i_home.values] +
                            defs[df.i_away.values])
        away_theta = tt.exp(intercept + atts[df.i_away.values] +
                            defs[df.i_home.values])

    # update beleifs with observations
    metric = 'score'
    home_metric = 'home_'+metric
    away_metric = 'away_'+metric

    K = 1  # TODO: let K be a function of the metric. 10yards, 1score
    with model:
        # likelihood of observed data
        home_score = pm.Poisson('home_score',
                                mu=home_theta,
                                observed=df[home_metric].values/K)
        away_score = pm.Poisson('away_score',
                                mu=away_theta,
                                observed=df[away_metric].values/K)

    # sampling
    with model:
        start = pm.find_MAP(fmin=optimize.fmin_powell, maxeval=50000)
        step = pm.NUTS()
        trace = pm.sample(20000, step, start=start)

    fig, ax = plt.subplots(8,2, figsize=(10, 15))
    pm.traceplot(trace, ax=ax)
    fig.savefig('traceplot_week_%s.png' % (week))


    # conduct simulation
    simuls = simulate_team_seasons(df_test, num_simulations, trace, metric, K)
    simuls.set_index('team', inplace=True)
    df_observed = create_team_season_table(df_test, metric)
    g = simuls.groupby('team')
    season_hdis = pd.DataFrame({'metric_lower': g[metric].quantile(.1),
                                'metric_median': g[metric].median(),
                                'metric_upper': g[metric].quantile(.9)
                                })

    season_hdis = pd.merge(season_hdis,
                           df_observed, left_index=True, right_on='team')
    season_hdis['relative_upper'] = season_hdis.metric_upper- season_hdis.metric_median
    season_hdis['relative_lower'] = season_hdis.metric_median - season_hdis.metric_lower
    season_hdis = season_hdis.sort_index(by=metric)
    season_hdis = season_hdis.reset_index()
    season_hdis['x'] = season_hdis.index + .5

    # simulation spread
    win_record = np.ndarray([num_simulations, len(df_test)])
    spr_record = np.ndarray([num_simulations, len(df_test)])
    for i in range(0, num_simulations):
        home_scores = simuls[simuls.iteration == i].loc[
            df_test['home_team']]['score'].values
        away_scores = simuls[simuls.iteration == i].loc[
            df_test['away_team']]['score'].values

        win_record[i] = home_scores > away_scores
        spr_record[i] = away_scores - home_scores

    spread = pd.DataFrame(spr_record)
    spread.columns = df_test[df_test.week == week].home_team.values
    spread['week'] = week
    return spread


def simulate_team_seasons(games, n, trace, metric, K):
    dfs = []
    for i in range(n):
        season = simulate_team_season(games, trace, metric, K)
        t = create_team_season_table(season, metric)
        t['iteration'] = i
        dfs.append(t)
    return pd.concat(dfs, ignore_index=True)


def simulate_team_season(games, trace, metric, K):
    # metric for home and away
    home_metric = 'home_'+metric
    away_metric = 'away_'+metric

    num_samples = trace['atts'].shape[0]
    draw = np.random.randint(10000, num_samples)
    atts_draw = pd.DataFrame({'att': trace['atts'][draw, :], })
    defs_draw = pd.DataFrame({'def': trace['defs'][draw, :], })
    home_draw = trace['home'][draw]
    intercept_draw = trace['intercept'][draw]
    season = games.copy()
    season = pd.merge(season, atts_draw, left_on='i_home', right_index=True)
    season = pd.merge(season, defs_draw, left_on='i_home', right_index=True)
    season = season.rename(columns={'att': 'att_home', 'def': 'def_home'})
    season = pd.merge(season, atts_draw, left_on='i_away', right_index=True)
    season = pd.merge(season, defs_draw, left_on='i_away', right_index=True)
    season = season.rename(columns={'att': 'att_away', 'def': 'def_away'})
    season['home'] = home_draw
    season['intercept'] = intercept_draw
    season['home_theta'] = season.apply(lambda x: math.exp(x['intercept'] +
                                        x['home'] +
                                        x['att_home'] +
                                        x['def_away']), axis=1)
    season['away_theta'] = season.apply(lambda x: math.exp(x['intercept'] +
                                        x['att_away'] +
                                        x['def_home']), axis=1)
    season[home_metric] = season.apply(lambda x: K*np.random.poisson(x['home_theta']), axis=1)
    season[away_metric] = season.apply(lambda x: K*np.random.poisson(x['away_theta']), axis=1)
    return season


def create_team_season_table(season, metric):
    # metric for home and away
    home_metric = 'home_'+metric
    away_metric = 'away_'+metric

    home = season[['home_team', home_metric]].rename(
        columns={'home_team': 'team', home_metric: metric})
    # metric for away team
    away = season[['away_team', away_metric]].rename(
        columns={'away_team': 'team', away_metric: metric})

    return pd.concat([home, away], axis=0)
