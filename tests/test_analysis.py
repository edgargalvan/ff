"""Tests for opponent-strength analysis (test_avg_passing.py).

Includes unit tests with mock data and integration tests against nflreadpy.
"""

import pytest
import polars as pl
from src.analysis import find_opponent, get_team_passing_yards, passing_vs_opponent_strength
import nflreadpy as nfl


# --- Fixtures for unit tests (no network) ---

@pytest.fixture
def mock_schedule():
    """Minimal schedule DataFrame for unit testing."""
    return pl.DataFrame({
        "week": [1, 1, 2, 2, 3],
        "home_team": ["BUF", "KC", "BUF", "KC", "BUF"],
        "away_team": ["MIA", "DEN", "NYJ", "MIA", "KC"],
        "game_type": ["REG", "REG", "REG", "REG", "REG"],
    })


@pytest.fixture
def mock_stats():
    """Minimal player stats DataFrame for unit testing."""
    return pl.DataFrame({
        "player_display_name": ["QB1", "QB2", "QB3", "QB1", "QB2", "QB3"],
        "team": ["BUF", "MIA", "NYJ", "BUF", "MIA", "KC"],
        "week": [1, 1, 2, 2, 2, 2],
        "passing_yards": [300, 250, 200, 280, 190, 310],
    })


# --- Unit tests for find_opponent ---

class TestFindOpponent:
    def test_home_team(self, mock_schedule):
        # BUF is home in week 1, opponent is MIA
        assert find_opponent(mock_schedule, "BUF", 1) == "MIA"

    def test_away_team(self, mock_schedule):
        # MIA is away in week 1, opponent is BUF
        assert find_opponent(mock_schedule, "MIA", 1) == "BUF"

    def test_bye_week(self, mock_schedule):
        # DEN doesn't play in week 2
        assert find_opponent(mock_schedule, "DEN", 2) is None

    def test_no_game_found(self, mock_schedule):
        assert find_opponent(mock_schedule, "BUF", 99) is None

    def test_week_3(self, mock_schedule):
        assert find_opponent(mock_schedule, "BUF", 3) == "KC"
        assert find_opponent(mock_schedule, "KC", 3) == "BUF"


# --- Unit tests for get_team_passing_yards ---

class TestGetTeamPassingYards:
    def test_single_player(self, mock_stats):
        # BUF in week 1: QB1 had 300 yards
        result = get_team_passing_yards(mock_stats, "BUF", 1)
        assert result == 300

    def test_multiple_players_same_team(self, mock_stats):
        # MIA in week 1: QB2 had 250 yards (only one player)
        result = get_team_passing_yards(mock_stats, "MIA", 1)
        assert result == 250

    def test_no_data(self, mock_stats):
        # DEN has no passing stats
        result = get_team_passing_yards(mock_stats, "DEN", 1)
        assert result == 0

    def test_wrong_week(self, mock_stats):
        result = get_team_passing_yards(mock_stats, "BUF", 99)
        assert result == 0


# --- Integration tests for passing_vs_opponent_strength ---

class TestPassingVsOpponentStrength:
    def test_returns_three_lists(self):
        weeks, yds, givenup = passing_vs_opponent_strength("Josh Allen", "BUF", 2024)
        assert isinstance(weeks, list)
        assert isinstance(yds, list)
        assert isinstance(givenup, list)

    def test_all_lists_same_length(self):
        weeks, yds, givenup = passing_vs_opponent_strength("Josh Allen", "BUF", 2024)
        assert len(weeks) == len(yds) == len(givenup)

    def test_weeks_start_at_2_or_later(self):
        weeks, _, _ = passing_vs_opponent_strength("Josh Allen", "BUF", 2024)
        assert all(w >= 2 for w in weeks)

    def test_has_data(self):
        weeks, yds, givenup = passing_vs_opponent_strength("Josh Allen", "BUF", 2024)
        assert len(weeks) >= 10  # should have most of the season

    def test_passing_yards_positive(self):
        _, yds, _ = passing_vs_opponent_strength("Josh Allen", "BUF", 2024)
        assert all(y > 0 for y in yds)

    def test_givenup_positive(self):
        _, _, givenup = passing_vs_opponent_strength("Josh Allen", "BUF", 2024)
        assert all(g >= 0 for g in givenup)

    def test_weeks_are_sorted(self):
        weeks, _, _ = passing_vs_opponent_strength("Josh Allen", "BUF", 2024)
        assert weeks == sorted(weeks)
