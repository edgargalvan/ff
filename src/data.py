"""
NFL game data loading via nflreadpy.
Replaces the old nfldbim.py module that used nfldb.
"""

import nflreadpy as nfl
import pandas as pd


def load_game_data(season_year, weeks=None):
    """
    Load NFL game data for a season and return a DataFrame ready for modeling.

    Args:
        season_year: NFL season year (e.g. 2024)
        weeks: list of weeks to include, or None for all regular season

    Returns:
        (df, teams) tuple where:
        - df: DataFrame with home_team, away_team, home_score, away_score,
              week, i_home, i_away
        - teams: DataFrame mapping team abbreviations to integer indices
    """
    schedule = nfl.load_schedules(seasons=[season_year])

    # Filter to regular season, completed games
    games = schedule.filter(
        (schedule["game_type"] == "REG")
        & (schedule["home_score"].is_not_null())
        & (schedule["away_score"].is_not_null())
    )

    if weeks is not None:
        games = games.filter(games["week"].is_in(weeks))

    # Convert to pandas (the model uses pandas DataFrames)
    df = games.select([
        "week", "home_team", "away_team", "home_score", "away_score"
    ]).to_pandas()

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    return munge_game_data(df)


def munge_game_data(df, teams=None):
    """
    Add team integer indices to game data.

    Args:
        df: DataFrame with home_team, away_team columns
        teams: optional existing team mapping DataFrame

    Returns:
        (df, teams) tuple with i_home, i_away columns added
    """
    if teams is None:
        all_teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
        teams = pd.DataFrame({"team": all_teams, "i": range(len(all_teams))})

    df = df.merge(teams, left_on="home_team", right_on="team", how="left")
    df = df.rename(columns={"i": "i_home"}).drop(columns=["team"])
    df = df.merge(teams, left_on="away_team", right_on="team", how="left")
    df = df.rename(columns={"i": "i_away"}).drop(columns=["team"])

    return df, teams
