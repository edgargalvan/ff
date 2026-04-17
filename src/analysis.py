import nflreadpy as nfl
import matplotlib.pyplot as plt
import polars as pl
import numpy as np


def find_opponent(schedule, team, week):
    """Find the opponent for a given team in a given week.

    Returns the opponent team abbreviation, or None if no game found (bye week).
    """
    game = schedule.filter(
        (pl.col("week") == week)
        & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
    )
    if game.is_empty():
        return None

    g = game.row(0, named=True)
    return g["away_team"] if g["home_team"] == team else g["home_team"]


def get_team_passing_yards(stats, team, week):
    """Get total passing yards for a team in a given week."""
    team_stats = stats.filter(
        (pl.col("week") == week)
        & (pl.col("team") == team)
        & (pl.col("passing_yards").is_not_null())
    )
    return team_stats["passing_yards"].sum()


def passing_vs_opponent_strength(player_name, player_team, season_year):
    """Analyze a player's passing yards vs opponent defensive strength.

    For each week the player played (week 2+), computes:
    - The player's passing yards that week
    - How many passing yards the opponent allowed the previous week
      (i.e., the opponent's previous opponent's total passing yards)

    Returns (weeks, passing_yds, passing_yds_givenup) as lists.
    """
    stats = nfl.load_player_stats(seasons=[season_year], summary_level="week")
    schedule = nfl.load_schedules(seasons=[season_year])
    reg_games = schedule.filter(pl.col("game_type") == "REG")

    player_stats = (
        stats.filter(pl.col("player_display_name") == player_name)
        .select(["week", "passing_yards", "team"])
        .sort("week")
    )

    weeks = []
    passing_yds = []
    passing_yds_givenup = []

    for row in player_stats.iter_rows(named=True):
        week = row["week"]
        if week < 2:
            continue

        # Find opponent this week
        opp_team = find_opponent(reg_games, player_team, week)
        if opp_team is None:
            continue

        # Find opponent's opponent from previous week
        opp_opp_team = find_opponent(reg_games, opp_team, week - 1)
        if opp_opp_team is None:
            continue

        # How many passing yards did the opponent give up last week?
        total_givenup = get_team_passing_yards(stats, opp_opp_team, week - 1)

        weeks.append(week)
        passing_yds.append(row["passing_yards"])
        passing_yds_givenup.append(total_givenup)

    return weeks, passing_yds, passing_yds_givenup


if __name__ == "__main__":
    player_name = "Josh Allen"
    player_team = "BUF"
    season_year = 2024

    weeks, passing_yds, passing_yds_givenup = passing_vs_opponent_strength(
        player_name, player_team, season_year
    )

    print(f"{player_name} passing yards: {passing_yds}")
    print(f"Opponent's opponent passing yards (prev week): {passing_yds_givenup}")

    plt.plot(weeks, passing_yds, "-r", label=f"{player_name} passing yards")
    plt.plot(weeks, passing_yds_givenup, "-b", label="Opp gave up (prev week)")
    plt.xlabel("Week")
    plt.ylabel("Passing Yards")
    plt.title(f"{player_name} ({season_year}) - Passing Yards vs Opponent Strength")
    plt.legend()
    plt.show()
