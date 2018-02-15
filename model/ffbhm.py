import numpy as np
import pandas as pd
import theano.tensor as T
import pymc3 as pm
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns


def bhm(df, metric='score', K=1, nu=3., sd=2.5):
    # priors
    model = pm.Model()
    with model:
        # global model parameters
        home = pm.Flat('home')
        sd_att = pm.HalfStudentT('sd_att', nu=nu, sd=sd)
        sd_def = pm.HalfStudentT('sd_def', nu=nu, sd=sd)
        intercept = pm.Flat('intercept')

        # team-specific parameters
        atts_star = pm.Normal('atts_star',
                              mu=0,
                              tau=sd_att,
                              shape=32)
        defs_star = pm.Normal('defs_star',
                              mu=0,
                              tau=sd_def,
                              shape=32)

    # constraints
    with model:
        atts = pm.Deterministic('atts', atts_star - T.mean(atts_star))
        defs = pm.Deterministic('defs', defs_star - T.mean(defs_star))
        home_theta = T.exp(intercept + home + atts[df.i_home]+defs[df.i_away])
        away_theta = T.exp(intercept + atts[df.i_away]+defs[df.i_home])

    # update beleifs with observations
    home_metric = 'home_'+metric
    away_metric = 'away_'+metric

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
        trace = pm.sample(1000, step=step, start=start)

    #fig, ax = plt.subplots(8, 2, figsize=(10, 15))
    #pm.traceplot(trace, ax=ax)
    #fig.savefig('traceplot_week_%s.png' % (week))
    return trace


def simulate_weeks(df_observed, trace, nsims=1000, metric='score',
                   K=1, burnin=1000):
    dfs = []
    for i in range(nsims):
        df_week = simulate_games(df_observed, trace, metric=metric,
                                K=K, burnin=burnin)
        t = create_table_week(df_week, metric)
        t['iteration'] = i
        dfs.append(t)
    return pd.concat(dfs, ignore_index=True)


def simulate_seasons(df_observed, trace, teams, nsims=1000, metric='score',
                     K=1, burnin=1000):
    dfs = []
    for i in range(nsims):
        season = simulate_games(df_observed, trace, metric=metric,
                                 K=K, burnin=burnin)
        table = create_table_season(season, teams, metric)
        table['iteration'] = i
        dfs.append(table)
    return pd.concat(dfs, ignore_index=True)


def simulate_games(df_observed, trace, metric='score', K=1, burnin=1000):
    """
    Simulate a games for a week or season once, using one random draw from the
    mcmc chain.
    """
    # metric for home and away
    home_metric = 'home_'+metric
    away_metric = 'away_'+metric

    num_samples = trace['atts'].shape[0]
    draw = np.random.randint(burnin, num_samples)
    atts_draw = pd.DataFrame({'att': trace['atts'][draw, :]})
    defs_draw = pd.DataFrame({'def': trace['defs'][draw, :]})
    home_draw = trace['home'][draw]
    intercept_draw = trace['intercept'][draw]

    df_games = df_observed.copy()
    df_games = pd.merge(df_games, atts_draw, left_on='i_home', right_index=True)
    df_games = pd.merge(df_games, defs_draw, left_on='i_home', right_index=True)
    df_games = df_games.rename(columns={'att': 'att_home', 'def': 'def_home'})
    df_games = pd.merge(df_games, atts_draw, left_on='i_away', right_index=True)
    df_games = pd.merge(df_games, defs_draw, left_on='i_away', right_index=True)
    df_games = df_games.rename(columns={'att': 'att_away', 'def': 'def_away'})
    df_games['home'] = home_draw
    df_games['intercept'] = intercept_draw
    df_games['home_theta'] = df_games.apply(lambda x:
                                        np.exp(x['intercept'] +
                                               x['home'] +
                                               x['att_home'] +
                                               x['def_away']), axis=1)
    df_games['away_theta'] = df_games.apply(lambda x:
                                        np.exp(x['intercept'] +
                                               x['att_away'] +
                                               x['def_home']), axis=1)
    df_games[home_metric] = df_games.apply(lambda x:
                                       K*np.random.poisson(x['home_theta']),
                                       axis=1)
    df_games[away_metric] = df_games.apply(lambda x:
                                       K*np.random.poisson(x['away_theta']),
                                       axis=1)
    return df_games


def create_table_week(df_week, metric):
    # metric for home and away
    home_metric = 'home_'+metric
    away_metric = 'away_'+metric

    home = df_week[['home_team', home_metric]].rename(
        columns={'home_team': 'team', home_metric: metric})
    # metric for away team
    away = df_week[['away_team', away_metric]].rename(
        columns={'away_team': 'team', away_metric: metric})

    return pd.concat([home, away], axis=0)


def create_table_season(df_season, teams, metric='score'):
    '''
    use dataframe output of simulate_season to creat summary season table
    '''
    home_metric = 'home_' + metric
    away_metric = 'away_' + metric

    g = df_season.groupby('i_home')
    home = pd.DataFrame({'home_for': g[home_metric].sum(),
                         'home_against': g[away_metric].sum()})
    g = df_season.groupby('i_away')
    away = pd.DataFrame({'away_for': g[away_metric].sum(),
                         'away_against': g[home_metric].sum()})
    df_table = home.join(away)
    df_table[metric] = df_table.home_for + df_table.away_for
    df_table[metric+'_against'] = df_table.home_against + df_table.away_against
    df_table = pd.merge(teams, df_table, left_on='i', right_index=True)
    return df_table


def predictions(df, simuls, teams, nsims=5000, metric='score'):
    # season or week?
    if len(df.week.unique()) == 1:
        season_sim = False
    else:
        season_sim = True

    # simulation record
    home_record = np.ndarray([nsims, len(df.groupby('home_team'))])
    away_record = np.ndarray([nsims, len(df.groupby('home_team'))])

    for i in range(0, nsims):
        s = simuls[simuls.iteration == i]
        home_scores = s.loc[s.team.isin(df.home_team)].score
        away_scores = s.loc[s.team.isin(df.away_team)].score

        home_record[i] = home_scores
        away_record[i] = away_scores

    df_home = pd.DataFrame(home_record)
    df_away = pd.DataFrame(away_record)

    df_home.columns = df.home_team.unique()
    df_away.columns = df.away_team.unique()

    scores = pd.concat([df_home, df_away], axis=1)

    simuls.set_index('team', inplace=True)

    if season_sim:
        df_observed = create_table_season(df, teams, metric)
    else:
        df_observed = create_table_week(df, metric)

    g = simuls.groupby('team')
    hdis = pd.DataFrame({'metric_lower': g[metric].quantile(.1),
                         'metric_median': g[metric].median(),
                         'metric_upper': g[metric].quantile(.9)})

    hdis = pd.merge(hdis, df_observed, left_index=True, right_on='team')
    hdis['relative_upper'] = hdis.metric_upper - hdis.metric_median
    hdis['relative_lower'] = hdis.metric_median - hdis.metric_lower
    hdis = hdis.sort_index(by=metric)
    hdis = hdis.reset_index()
    hdis['x'] = hdis.index + .5

    return scores, hdis


def plot_hdis(hdis, metric):
    fig, axs = plt.subplots(figsize=(10, 6))
    axs.scatter(hdis.x, hdis[metric],
                c=sns.palettes.color_palette()[4], zorder = 10,
                label='Actual  '+metric+'For')

    axs.errorbar(hdis.x, hdis.metric_median,
                 yerr=(hdis[['relative_lower', 'relative_upper']].values).T,
                 fmt='s', c=sns.palettes.color_palette()[5],
                 label='Simulations')

    axs.set_title('Actual '+metric+' For, and 90% Interval from Simulations, by Team')
    axs.set_xlabel('Team')
    axs.set_ylabel(metric)
    axs.set_xlim(0, 20)
    axs.legend()

    _= axs.set_xticks(hdis.index + .5)
    _= axs.set_xticklabels(hdis['team'].values, rotation=45)
