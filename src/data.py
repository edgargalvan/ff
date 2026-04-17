"""
NFL game data loading via nflreadpy.
Replaces the old nfldbim.py module that used nfldb.
"""

from __future__ import annotations

import nflreadpy as nfl
import pandas as pd
import numpy as np


def load_game_data(season_year: int,
                   weeks: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load NFL game data for a season and return a DataFrame ready for modeling.

    Args:
        season_year: NFL season year (e.g. 2024)
        weeks: list of weeks to include, or None for all regular season

    Returns:
        (df, teams) tuple where:
        - df: DataFrame with game data, team indices, and covariates
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

    # Select core columns + covariates
    columns = [
        "week", "home_team", "away_team", "home_score", "away_score",
        # Covariates
        "home_rest", "away_rest",   # days since last game
        "div_game",                  # divisional game flag
        "roof",                      # dome, outdoors, etc.
        "temp", "wind",              # weather
    ]
    available = [c for c in columns if c in games.columns]
    df = games.select(available).to_pandas()

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Compute derived covariates
    _add_covariates(df)

    return munge_game_data(df)


def _add_covariates(df):
    """Compute derived covariate features in-place."""
    # Rest advantage: positive = home team more rested
    if "home_rest" in df.columns and "away_rest" in df.columns:
        df["home_rest"] = pd.to_numeric(df["home_rest"], errors="coerce").fillna(7)
        df["away_rest"] = pd.to_numeric(df["away_rest"], errors="coerce").fillna(7)
        df["rest_advantage"] = df["home_rest"] - df["away_rest"]
        # Short week flag (< 6 days rest)
        df["home_short_week"] = (df["home_rest"] < 6).astype(int)
        df["away_short_week"] = (df["away_rest"] < 6).astype(int)

    # Divisional game (higher stakes, teams know each other)
    if "div_game" in df.columns:
        df["div_game"] = df["div_game"].astype(int)

    # Indoor game (dome/closed roof)
    if "roof" in df.columns:
        df["is_indoor"] = df["roof"].isin(["dome", "closed"]).astype(int)

    # Temperature: standardize, fill missing (indoor games) with median
    if "temp" in df.columns:
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        median_temp = df["temp"].median()
        df["temp"] = df["temp"].fillna(median_temp)
        temp_mean = df["temp"].mean()
        temp_std = df["temp"].std()
        if temp_std > 0:
            df["temp_std"] = (df["temp"] - temp_mean) / temp_std
        else:
            df["temp_std"] = 0.0

    # Wind: standardize, fill missing with 0 (indoor)
    if "wind" in df.columns:
        df["wind"] = pd.to_numeric(df["wind"], errors="coerce").fillna(0)
        wind_mean = df["wind"].mean()
        wind_std = df["wind"].std()
        if wind_std > 0:
            df["wind_std"] = (df["wind"] - wind_mean) / wind_std
        else:
            df["wind_std"] = 0.0


def munge_game_data(df: pd.DataFrame,
                    teams: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
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
