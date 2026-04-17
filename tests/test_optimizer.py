"""Unit tests for the lineup optimizer — no network calls needed."""

import pytest
import polars as pl
from src.optimizers.lineup_optimizer import (
    optimize_lineup,
    find_multiple_lineups,
    DEFAULT_SALARY_CAP,
    DEFAULT_ROSTER_SIZE,
    DEFAULT_POSITION_LIMITS,
)


@pytest.fixture
def sample_players():
    return pl.DataFrame({
        "name": [
            "QB1", "QB2",
            "RB1", "RB2", "RB3",
            "WR1", "WR2", "WR3", "WR4",
            "TE1", "TE2",
            "K1", "K2",
            "DEF1", "DEF2",
        ],
        "position": [
            "QB", "QB",
            "RB", "RB", "RB",
            "WR", "WR", "WR", "WR",
            "TE", "TE",
            "K", "K",
            "DEF", "DEF",
        ],
        "team": [
            "A", "B",
            "A", "B", "C",
            "A", "B", "C", "D",
            "A", "B",
            "C", "D",
            "A", "B",
        ],
        "salary": [
            8000, 7500,
            7000, 6500, 6000,
            7500, 7000, 6500, 6000,
            5500, 5000,
            4500, 4000,
            3500, 3000,
        ],
        "projected_points": [
            25.0, 22.0,
            18.0, 16.0, 14.0,
            20.0, 18.0, 16.0, 14.0,
            12.0, 10.0,
            8.0, 7.0,
            9.0, 8.0,
        ],
    })


class TestOptimizeLineup:
    def test_returns_dataframe(self, sample_players):
        result = optimize_lineup(sample_players)
        assert isinstance(result, pl.DataFrame)

    def test_roster_size(self, sample_players):
        result = optimize_lineup(sample_players)
        assert len(result) == DEFAULT_ROSTER_SIZE

    def test_salary_cap_respected(self, sample_players):
        result = optimize_lineup(sample_players)
        assert result["salary"].sum() <= DEFAULT_SALARY_CAP

    def test_position_limits_respected(self, sample_players):
        result = optimize_lineup(sample_players)
        for pos, limit in DEFAULT_POSITION_LIMITS.items():
            count = result.filter(pl.col("position") == pos).height
            assert count <= limit, f"{pos}: got {count}, limit {limit}"

    def test_all_positions_filled(self, sample_players):
        result = optimize_lineup(sample_players)
        positions_in_lineup = set(result["position"].to_list())
        assert positions_in_lineup == {"QB", "RB", "WR", "TE", "K", "DEF"}

    def test_team_stacking_limit(self, sample_players):
        result = optimize_lineup(sample_players, max_per_team=4)
        for team in result["team"].unique().to_list():
            count = result.filter(pl.col("team") == team).height
            assert count <= 4, f"Team {team}: got {count} players, max 4"

    def test_strict_team_limit(self, sample_players):
        """With max_per_team=2, no team should have more than 2."""
        result = optimize_lineup(sample_players, max_per_team=2)
        if result is not None:
            for team in result["team"].unique().to_list():
                count = result.filter(pl.col("team") == team).height
                assert count <= 2

    def test_exclude_players(self, sample_players):
        result = optimize_lineup(sample_players, exclude=["QB1"])
        names = result["name"].to_list()
        assert "QB1" not in names

    def test_exclude_forces_different_pick(self, sample_players):
        result_with = optimize_lineup(sample_players)
        result_without = optimize_lineup(sample_players, exclude=["QB1"])
        # If QB1 was in the original, the new one should differ
        if "QB1" in result_with["name"].to_list():
            assert result_with["name"].to_list() != result_without["name"].to_list()

    def test_maximizes_points(self, sample_players):
        """The optimizer should pick higher-projected players when possible."""
        result = optimize_lineup(sample_players)
        # QB1 (25.0) should be preferred over QB2 (22.0)
        qbs = result.filter(pl.col("position") == "QB")
        assert qbs["name"].to_list() == ["QB1"]

    def test_empty_pool_returns_none(self):
        empty = pl.DataFrame({
            "name": [],
            "position": [],
            "team": [],
            "salary": [],
            "projected_points": [],
        })
        assert optimize_lineup(empty) is None

    def test_infeasible_salary_cap(self, sample_players):
        """Impossible salary cap should return None."""
        result = optimize_lineup(sample_players, salary_cap=1000)
        assert result is None

    def test_custom_salary_cap(self, sample_players):
        result = optimize_lineup(sample_players, salary_cap=50000)
        if result is not None:
            assert result["salary"].sum() <= 50000


class TestFindMultipleLineups:
    def test_returns_list(self, sample_players):
        lineups = find_multiple_lineups(sample_players, num_lineups=2)
        assert isinstance(lineups, list)

    def test_correct_count(self, sample_players):
        lineups = find_multiple_lineups(sample_players, num_lineups=3)
        assert len(lineups) <= 3
        assert len(lineups) >= 1

    def test_lineups_are_distinct(self, sample_players):
        lineups = find_multiple_lineups(sample_players, num_lineups=3)
        if len(lineups) >= 2:
            names_0 = set(lineups[0]["name"].to_list())
            names_1 = set(lineups[1]["name"].to_list())
            assert names_0 != names_1

    def test_all_lineups_valid(self, sample_players):
        lineups = find_multiple_lineups(sample_players, num_lineups=3)
        for lu in lineups:
            assert len(lu) == DEFAULT_ROSTER_SIZE
            assert lu["salary"].sum() <= DEFAULT_SALARY_CAP

    def test_first_lineup_is_best(self, sample_players):
        lineups = find_multiple_lineups(sample_players, num_lineups=3)
        if len(lineups) >= 2:
            assert lineups[0]["projected_points"].sum() >= lineups[1]["projected_points"].sum()
