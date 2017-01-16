# BLOCK
import math
import sys
# import ipdb
from IPython.core.debugger import Tracer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import pymc3 as pm
import theano.tensor as tt

# set figure size
rcParams['figure.figsize'] = 3, 1.5

# check python version, should be python3
sys.version


# BLOCK
data_file = 'premier_league.txt'
df = pd.read_csv(data_file, sep='\t', index_col=0,)
df.head()

# BLOCK
df.index = df.columns
rows = []
for i in df.index:
    for j in df.columns:
        if i == j: continue
        score = df.ix[i, j]
        score = [int(row) for row in score.split('-')]
        rows.append([i, j, score[0], score[1]])
df = pd.DataFrame(rows, columns = ['home', 'away', 'home_score', 'away_score'])
df.head()

'''
RESULTS:
 :   home away  home_score  away_score
 : 0  ARS  AST           1           3
 : 1  ARS  CAR           2           0
 : 2  ARS  CHE           0           0
 : 3  ARS  CRY           2           0
 : 4  ARS  EVE           1           1
'''

# BLOCK
teams = df.home.unique()
teams = pd.DataFrame(teams, columns=['team'])
teams['i'] = teams.index
teams.head()

'''
RESULTS:
 :   team  i
 : 0  ARS  0
 : 1  AST  1
 : 2  CAR  2
 : 3  CHE  3
 : 4  CRY  4
'''

# BLOCK
df = pd.merge(df, teams, left_on='home', right_on='team', how='left')
df = df.rename(columns = {'i': 'i_home'}).drop('team', 1)
df = pd.merge(df, teams, left_on='away', right_on='team', how='left')
df = df.rename(columns = {'i': 'i_away'}).drop('team', 1)
df.head()

'''
RESULTS:
 :   home away  home_score  away_score  i_home  i_away
 : 0  ARS  AST           1           3       0       1
 : 1  ARS  CAR           2           0       0       2
 : 2  ARS  CHE           0           0       0       3
 : 3  ARS  CRY           2           0       0       4
 : 4  ARS  EVE           1           1       0       5
'''

# BLOCK
observed_home_goals = df.home_score.values
observed_away_goals = df.away_score.values
home_team = df.i_home.values
away_team = df.i_away.values
num_teams = len(df.i_home.unique())
num_games = len(home_team)


# BLOCK
g = df.groupby('i_away')
att_starting_points = np.log(g.away_score.mean())
g = df.groupby('i_home')
def_starting_points = -np.log(g.away_score.mean())


# BLOCK
model = pm.Model()
with pm.Model() as model:
    home = pm.Normal('home', mu=0, tau=.0001)
    intercept = pm.Normal('intercept', mu=0, tau=.0001)


# BLOCK
with model:
    tau_att = pm.Gamma('tau_att', alpha=.1, beta=.1)
    tau_def = pm.Gamma('tau_def', alpha=.1, beta=.1)


# BLOCK
with model:
    atts_star = pm.Normal("atts_star",
                          mu=0,
                          tau=tau_att,
                          shape=num_teams)
                          # testval=att_starting_points.values)

    defs_star = pm.Normal("defs_star",
                          mu=0,
                          tau=tau_def,
                          shape=num_teams)
                          # testval=def_starting_points.values)


# BLOCK
with model:
        atts = pm.Deterministic("atts", atts_star - tt.mean(atts_star))
        defs = pm.Deterministic("defs", defs_star - tt.mean(defs_star))


# BLOCK
with model:
    home_theta = tt.exp(intercept + home + atts[home_team] + defs[away_team])
    away_theta = tt.exp(intercept + atts[away_team] + defs[home_team])

    home_goals = pm.Poisson('home_goals',
                            mu=home_theta,
                            # testval=observed_home_goals,
                            observed=True)

    away_goals = pm.Poisson('away_goals',
                            mu=away_theta,
                            # testval=observed_away_goals,
                            observed=True)
Tracer()()
# BLOCK
with model:
    start = pm.find_MAP()
    step = pm.NUTS(state=start)
    trace = pm.sample(20000, step, init=start)
    pm.traceplot(trace)
    burned_trace = trace[1000:]


# BLOCK
plt.hist(trace['tau_def'], histtype='stepfilled', bins=25, alpha=0.85)


# BLOCK
def simulate_season():
    """
    Simulate a season once, using one random draw from the mcmc chain.
    """
    num_samples = trace['atts'].shape[0]
    draw = np.random.randint(0, num_samples)
    atts_draw = pd.DataFrame({'att': trace['atts'][draw, :], })
    defs_draw = pd.DataFrame({'def': trace['defs'][draw, :], })
    home_draw = trace['home'][draw]
    intercept_draw = trace['intercept'][draw]
    season = df.copy()
    season = pd.merge(season, atts_draw, left_on='i_home', right_index=True)
    season = pd.merge(season, defs_draw, left_on='i_home', right_index=True)
    season = season.rename(columns={'att': 'att_home', 'def': 'def_home'})
    season = pd.merge(season, atts_draw, left_on='i_away', right_index=True)
    season = pd.merge(season, defs_draw, left_on='i_away', right_index=True)
    season = season.rename(columns={'att': 'att_away', 'def': 'def_away'})
    season['home'] = home_draw
    season['intercept'] = intercept_draw
    season['home_theta'] = season.apply(lambda x: math.exp(
        x['intercept'] +
        x['home'] +
        x['att_home'] +
        x['def_away']), axis=1)
    season['away_theta'] = season.apply(lambda x: math.exp(
        x['intercept'] +
        x['att_away'] +
        x['def_home']), axis=1)
    season['home_goals'] = season.apply(lambda x: np.random.poisson(
        x['home_theta']), axis=1)
    season['away_goals'] = season.apply(lambda x: np.random.poisson(x['away_theta']), axis=1)
    season['home_outcome'] = season.apply(lambda x: 'win' if x['home_goals'] > x['away_goals'] else
                                                    'loss' if x['home_goals'] < x['away_goals'] else 'draw', axis=1)
    season['away_outcome'] = season.apply(lambda x: 'win' if x['home_goals'] < x['away_goals'] else
                                                    'loss' if x['home_goals'] > x['away_goals'] else 'draw', axis=1)
    season = season.join(pd.get_dummies(season.home_outcome, prefix='home'))
    season = season.join(pd.get_dummies(season.away_outcome, prefix='away'))
    return season


def create_season_table(season):
    """
    Using a season dataframe output by simulate_season(),
    create a summary dataframe with wins, losses, goals for, etc.

    """
    g = season.groupby('i_home')
    home = pd.DataFrame({'home_goals': g.home_goals.sum(),
                         'home_goals_against': g.away_goals.sum(),
                         'home_wins': g.home_win.sum(),
                         'home_draws': g.home_draw.sum(),
                         'home_losses': g.home_loss.sum()
                         })
    g = season.groupby('i_away')
    away = pd.DataFrame({'away_goals': g.away_goals.sum(),
                         'away_goals_against': g.home_goals.sum(),
                         'away_wins': g.away_win.sum(),
                         'away_draws': g.away_draw.sum(),
                         'away_losses': g.away_loss.sum()
                         })
    df = home.join(away)
    df['wins'] = df.home_wins + df.away_wins
    df['draws'] = df.home_draws + df.away_draws
    df['losses'] = df.home_losses + df.away_losses
    df['points'] = df.wins * 3 + df.draws
    df['gf'] = df.home_goals + df.away_goals
    df['ga'] = df.home_goals_against + df.away_goals_against
    df['gd'] = df.gf - df.ga
    df = pd.merge(teams, df, left_on='i', right_index=True)
    df = df.sort_index(by='points', ascending=False)
    df = df.reset_index()
    df['position'] = df.index + 1
    df['champion'] = (df.position == 1).astype(int)
    df['qualified_for_CL'] = (df.position < 5).astype(int)
    df['relegated'] = (df.position > 17).astype(int)
    return df


def simulate_seasons(n=100):
    dfs = []
    for i in range(n):
        s = simulate_season()
        t = create_season_table(s)
        t['iteration'] = i
        dfs.append(t)
    return pd.concat(dfs, ignore_index=True)

Tracer()()
# BLOCK
simuls = simulate_seasons(1000)


# BLOCK
ax = simuls.points[simuls.team == 'MCI'].hist(figsize=(7, 5))
median = simuls.points[simuls.team == 'MCI'].median()
ax.set_title('Man City: 2013-14 points, 1000 simulations')
ax.plot([median, median], ax.get_ylim())
plt.annotate('Median: %s' % median, xy=(median + 1, ax.get_ylim()[1]-10))


# BLOCK
observed_season = 'premier_league_13_14_table.csv'
df_observed = pd.read_csv(observed_season)
df_observed.loc[df_observed.QR.isnull(), 'QR'] = ''

g = simuls.groupby('team')
season_hdis = pd.DataFrame({'points_lower': g.points.quantile(.05),
                            'points_upper': g.points.quantile(.95),
                            'goals_for_lower': g.gf.quantile(.05),
                            'goals_for_median': g.gf.median(),
                            'goals_for_upper': g.gf.quantile(.95),
                            'goals_against_lower': g.ga.quantile(.05),
                            'goals_against_upper': g.ga.quantile(.95),
                            })
season_hdis = pd.merge(season_hdis, df_observed, left_index=True, right_on='team')
column_order = ['team', 'points_lower', 'Pts', 'points_upper',
                'goals_for_lower', 'GF', 'goals_for_median', 'goals_for_upper',
                'goals_against_lower', 'GA', 'goals_against_upper',]
season_hdis = season_hdis[column_order]
season_hdis['relative_goals_upper'] = season_hdis.goals_for_upper - season_hdis.goals_for_median
season_hdis['relative_goals_lower'] = season_hdis.goals_for_median - season_hdis.goals_for_lower
season_hdis = season_hdis.sort_index(by='GF')
season_hdis = season_hdis.reset_index()
season_hdis['x'] = season_hdis.index + .5
season_hdis

fig, axs = plt.subplots(figsize=(10,6))
axs.scatter(season_hdis.x, season_hdis.GF, c=sns.palettes.color_palette()[4], zorder = 10, label='Actual Goals For')
axs.errorbar(season_hdis.x, season_hdis.goals_for_median,
             yerr=(season_hdis[['relative_goals_lower', 'relative_goals_upper']].values).T,
             fmt='s', c=sns.palettes.color_palette()[5], label='Simulations')
axs.set_title('Actual Goals For, and 90% Interval from Simulations, by Team')
axs.set_xlabel('Team')
axs.set_ylabel('Goals Scored')
axs.set_xlim(0, 20)
axs.legend()
_= axs.set_xticks(season_hdis.index + .5)
_= axs.set_xticklabels(season_hdis['team'].values, rotation=45)
