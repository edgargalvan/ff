"""Integration tests for nflreadpy data functions.

These tests hit the nflreadpy API (downloads parquet files on first run).
They verify that our functions return the right shapes and types, not
specific stat values (which change as data is updated).
"""

import pytest
import polars as pl
from test_pts import (
    get_player_weekly_stats,
    get_defense_pa_stats,
    get_defense_stats,
    player_season_fantasy_points,
    calc_fantasy_points,
    score_defense_pa,
)
from config import scoring

SEASON = 2024


class TestGetPlayerWeeklyStats:
    def test_returns_dataframe(self):
        result = get_player_weekly_stats("Josh Allen", SEASON)
        assert isinstance(result, pl.DataFrame)

    def test_has_expected_columns(self):
        result = get_player_weekly_stats("Josh Allen", SEASON)
        for col in ["week", "passing_yards", "passing_tds", "player_display_name"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_filters_to_single_player(self):
        result = get_player_weekly_stats("Josh Allen", SEASON)
        names = result["player_display_name"].unique().to_list()
        assert names == ["Josh Allen"]

    def test_multiple_weeks(self):
        result = get_player_weekly_stats("Josh Allen", SEASON)
        assert result.height >= 10  # played at least 10 games

    def test_unknown_player_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_player_weekly_stats("Fake Player McFakerson", SEASON)


class TestGetDefensePAStats:
    def test_returns_dataframe(self):
        result = get_defense_pa_stats("BUF", SEASON)
        assert isinstance(result, pl.DataFrame)

    def test_has_week_and_points_allowed(self):
        result = get_defense_pa_stats("BUF", SEASON)
        assert "week" in result.columns
        assert "points_allowed" in result.columns

    def test_17_regular_season_games(self):
        result = get_defense_pa_stats("BUF", SEASON)
        assert result.height == 17

    def test_points_allowed_non_negative(self):
        result = get_defense_pa_stats("BUF", SEASON)
        assert (result["points_allowed"] >= 0).all()


class TestGetDefenseStats:
    def test_returns_dataframe(self):
        result = get_defense_stats("BUF", SEASON)
        assert isinstance(result, pl.DataFrame)

    def test_has_defensive_columns(self):
        result = get_defense_stats("BUF", SEASON)
        assert "week" in result.columns
        assert "def_sacks" in result.columns

    def test_sacks_are_non_negative(self):
        result = get_defense_stats("BUF", SEASON)
        assert (result["def_sacks"] >= 0).all()


class TestPlayerSeasonFantasyPoints:
    def test_returns_dataframe(self):
        result = player_season_fantasy_points("Josh Allen", SEASON)
        assert isinstance(result, pl.DataFrame)

    def test_has_week_and_fantasy_points(self):
        result = player_season_fantasy_points("Josh Allen", SEASON)
        assert "week" in result.columns
        assert "fantasy_points" in result.columns

    def test_fantasy_points_are_numeric(self):
        result = player_season_fantasy_points("Josh Allen", SEASON)
        assert result["fantasy_points"].dtype in (pl.Float64, pl.Float32, pl.Int64)

    def test_positive_total_for_star_qb(self):
        result = player_season_fantasy_points("Josh Allen", SEASON)
        assert result["fantasy_points"].sum() > 200  # star QB scores well

    def test_rb_stats_work(self):
        result = player_season_fantasy_points("Saquon Barkley", SEASON)
        assert result.height >= 10
        assert result["fantasy_points"].sum() > 100

    def test_wr_stats_work(self):
        result = player_season_fantasy_points("Ja'Marr Chase", SEASON)
        assert result.height >= 10
        assert result["fantasy_points"].sum() > 50


class TestKickerEndToEnd:
    """Verify kicker FG distance columns flow through scoring correctly."""

    def test_kicker_has_fg_columns(self):
        stats = get_player_weekly_stats("Tyler Bass", SEASON)
        fg_cols = ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
                   "fg_made_40_49", "fg_made_50_59", "fg_made_60_", "pat_made"]
        for col in fg_cols:
            assert col in stats.columns, f"Missing kicker column: {col}"

    def test_kicker_fantasy_points_positive(self):
        result = player_season_fantasy_points("Tyler Bass", SEASON)
        total = result["fantasy_points"].sum()
        assert total > 50, f"Kicker should score well over a season, got {total}"

    def test_kicker_scoring_uses_distance_buckets(self):
        """Verify a kicker's weekly score includes FG distance points."""
        stats = get_player_weekly_stats("Tyler Bass", SEASON)
        # Find a week where the kicker made at least one FG
        for row in stats.iter_rows(named=True):
            total_fgs = sum(
                row.get(c) or 0
                for c in ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
                           "fg_made_40_49", "fg_made_50_59", "fg_made_60_"]
            )
            if total_fgs > 0:
                row["fumbles_lost"] = 0  # kickers don't fumble, but field needs to exist
                pts = calc_fantasy_points(row)
                assert pts > 0, f"Kicker with {total_fgs} FGs should have positive points"
                return
        pytest.fail("Tyler Bass had no FG makes in 2024 — unexpected")


class TestDefenseEndToEnd:
    """Verify defense scoring combines PA tiers + defensive stats."""

    def test_defense_total_fantasy_points(self):
        """Compute full defense fantasy points for a team/week."""
        pa = get_defense_pa_stats("BUF", SEASON)
        def_stats = get_defense_stats("BUF", SEASON)

        # Pick a week that exists in both
        pa_weeks = set(pa["week"].to_list())
        def_weeks = set(def_stats["week"].to_list())
        common = sorted(pa_weeks & def_weeks)
        assert len(common) > 0, "No common weeks found"

        week = common[0]
        pa_row = pa.filter(pl.col("week") == week).row(0, named=True)
        def_row = def_stats.filter(pl.col("week") == week).row(0, named=True)

        pa_pts = score_defense_pa(pa_row["points_allowed"])
        def_pts = calc_fantasy_points(def_row)
        total = pa_pts + def_pts

        # Defense should have some score (could be negative in a bad week, but not absurd)
        assert -20 < total < 40, f"Defense total {total} seems unreasonable"

    def test_defense_stats_have_required_columns(self):
        result = get_defense_stats("BUF", SEASON)
        for col in ["def_sacks", "def_interceptions", "def_fumbles_forced", "def_tds"]:
            assert col in result.columns, f"Missing defense column: {col}"
