import pandas as pd
import nfldb


def nfldbim(season_year, weeks, teams=pd.DataFrame()):
    """
    nfldb import and munging
    """

    # import data from nfldb
    df = nfldbi(season_year, weeks)

    # munge data so that we can use it
    df = nfldbm(df, teams)

    return df


def nfldbi(season_year, weeks):
    """
    Import data from nfldb
    """
    # selftart up nfldb
    db = nfldb.connect()
    q = nfldb.Query(db)

    # plays = g.as_plays()
    # initialize
    home_team = []
    away_team = []
    gamekey = []
    home_passing_yds = []
    away_passing_yds = []
    home_rushing_yds = []
    away_rushing_yds = []
    home_rushing_tds = []
    away_rushing_tds = []
    home_receiving_tds = []
    away_receiving_tds = []
    home_score = []
    away_score = []
    week = []

    # loop through games except the last week
    '''
    if max(weeks) < 3:
        season_types = ['Preseason', 'Regular']
    else:
        season_types = ['Regular']
    '''

    for season_type in ['Preseason', 'Regular']:
#    for season_type in ['Regular']:
        if season_type is 'Preseason':
            _weeks = range(0, 5)
        else:
            _weeks = weeks
        num_games = 0
        for i in _weeks:
            # find out who plays who
            q = nfldb.Query(db).game(season_year=season_year,
                                     season_type=season_type,
                                     week=i)
            for g in q.as_games():
                home_team.append(g.home_team)
                away_team.append(g.away_team)
                home_score.append(g.home_score)
                away_score.append(g.away_score)
                gamekey.append(g.gamekey)
                if season_type is 'Preseason':
                    week.append(i-5)
                else:
                    week.append(i)
                num_games += 1

        # cycle through each playplayer for metrics
        for i in range(0, num_games):
            # home team yards
            q = nfldb.Query(db).game(gamekey=gamekey[i], team=home_team[i])
            q.play_player(team=home_team[i])
            pps = q.as_aggregate()
            home_passing_yds.append(sum(pp.passing_yds for pp in pps))
            home_rushing_yds.append(sum(pp.rushing_yds for pp in pps))    
            home_rushing_tds.append(sum(pp.rushing_tds for pp in pps))
            home_receiving_tds.append(sum(pp.receiving_tds for pp in pps))

            # away team yards
            q = nfldb.Query(db).game(gamekey=gamekey[i], team=away_team[i])
            q.play_player(team=away_team[i])
            pps = q.as_aggregate()
            away_passing_yds.append(sum(pp.passing_yds for pp in pps))
            away_rushing_yds.append(sum(pp.rushing_yds for pp in pps))

            away_rushing_tds.append(sum(pp.rushing_tds for pp in pps))
            away_receiving_tds.append(sum(pp.receiving_tds for pp in pps))

    # save to a new dataframe
    df = pd.DataFrame({'home_team': home_team,
                       'away_team': away_team,
                       'home_passing_yds': home_passing_yds,
                       'away_passing_yds': away_passing_yds,
                       'home_rushing_yds': home_rushing_yds,
                       'away_rushing_yds': away_rushing_yds,
                       'home_rushing_tds': home_rushing_tds,
                       'away_rushing_tds': away_rushing_tds,
                       'home_receiving_tds': home_receiving_tds,
                       'away_receiving_tds': away_receiving_tds,
                       'home_score': home_score,
                       'away_score': away_score,
                       'week': week,
                       'gamekey': gamekey})
    return df


def nfldbm(df, teams=pd.DataFrame()):
    """
    Munges nfldb data
    """
    # Here is where we can "fix" team names, e.g.,
    # df.loc[df.home_team == 'LAR', 'home_team'] = 'LA'
    # df.loc[df.away_team == 'LAR', 'away_team'] = 'LA'
    if teams.empty:
        # Create a look-up table for team names
        teams = pd.concat([df.home_team, df.away_team]).unique()
        teams = pd.DataFrame(teams, columns=['team'])
        teams['i'] = teams.index

    # Create away and home columns
    df = pd.merge(df, teams, left_on='home_team', right_on='team', how='left')
    df = df.rename(columns={'i': 'i_home'}).drop('team', 1)
    df = pd.merge(df, teams, left_on='away_team', right_on='team', how='left')
    df = df.rename(columns={'i': 'i_away'}).drop('team', 1)

    return df
