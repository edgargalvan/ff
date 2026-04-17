import nflreadpy as nfl
import polars as pl
import numpy as np
from src.config import scoring, defense_pa_tiers


def get_player_weekly_stats(player_name, season_year):
    """Get weekly fantasy-relevant stats for an offensive player."""
    stats = nfl.load_player_stats(seasons=[season_year], summary_level="week")
    player = stats.filter(pl.col("player_display_name") == player_name)

    if player.is_empty():
        raise ValueError(f"Player '{player_name}' not found in {season_year} data")

    return player



def get_defense_pa_stats(team, season_year):
    """Get defense points-against stats per week for a team."""
    schedule = nfl.load_schedules(seasons=[season_year])

    # Filter to regular season games involving this team
    team_games = schedule.filter(
        (pl.col("game_type") == "REG")
        & ((pl.col("home_team") == team) | (pl.col("away_team") == team))
    )

    # Calculate points allowed
    team_games = team_games.with_columns(
        pl.when(pl.col("home_team") == team)
        .then(pl.col("away_score"))
        .otherwise(pl.col("home_score"))
        .alias("points_allowed")
    )

    return team_games.select(["week", "points_allowed"]).sort("week")


def score_defense_pa(points_allowed):
    """Convert points allowed to fantasy points using tier system."""
    for low, high, pts in defense_pa_tiers:
        if low <= points_allowed <= high:
            return pts
    return 0


def get_defense_stats(team, season_year):
    """Get defensive stats (sacks, INTs, etc.) aggregated by team per week."""
    stats = nfl.load_player_stats(seasons=[season_year], summary_level="week")

    # Filter to defensive players on this team
    def_cols = [
        "week", "team",
        "def_sacks", "def_interceptions", "def_fumbles_forced", "def_tds",
    ]
    available = [c for c in def_cols if c in stats.columns]
    team_def = stats.filter(pl.col("team") == team).select(available)

    # Aggregate by week
    agg_cols = [c for c in available if c not in ("week", "team")]
    weekly = (
        team_def.group_by("week")
        .agg([pl.col(c).sum() for c in agg_cols])
        .sort("week")
    )

    return weekly


def calc_fantasy_points(player_stats_row, stat_keys=None):
    """Calculate fantasy points for a single week from a stats row (dict)."""
    if stat_keys is None:
        stat_keys = scoring.keys()

    total = 0.0
    for stat, multiplier in scoring.items():
        if stat in player_stats_row and player_stats_row[stat] is not None:
            total += player_stats_row[stat] * multiplier
    return total


def player_season_fantasy_points(player_name, season_year):
    """Get weekly fantasy points for a player over a season."""
    stats = get_player_weekly_stats(player_name, season_year)

    weekly_pts = []
    for row in stats.iter_rows(named=True):
        # Sum up fumbles_lost from rushing + receiving
        row["fumbles_lost"] = (row.get("rushing_fumbles_lost") or 0) + (row.get("receiving_fumbles_lost") or 0)
        pts = calc_fantasy_points(row)
        weekly_pts.append({"week": row["week"], "fantasy_points": pts})

    return pl.DataFrame(weekly_pts)


if __name__ == "__main__":
    name = "Josh Allen"
    season_year = 2024

    print(f"\n{name} - {season_year} Weekly Fantasy Points:")
    print("-" * 40)

    weekly = player_season_fantasy_points(name, season_year)
    for row in weekly.iter_rows(named=True):
        print(f"  Week {row['week']:2d}: {row['fantasy_points']:.1f} pts")

    total = weekly["fantasy_points"].sum()
    avg = weekly["fantasy_points"].mean()
    print(f"\n  Total: {total:.1f} pts")
    print(f"  Avg:   {avg:.1f} pts/week")

    # Defense example
    print(f"\nBUF Defense - {season_year} Points Against:")
    print("-" * 40)
    pa = get_defense_pa_stats("BUF", season_year)
    for row in pa.iter_rows(named=True):
        pa_pts = score_defense_pa(row["points_allowed"])
        print(f"  Week {row['week']:2d}: allowed {row['points_allowed']} pts -> {pa_pts} fantasy pts")
