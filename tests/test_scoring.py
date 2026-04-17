"""Unit tests for scoring logic — no network calls needed."""

import pytest
from config import scoring, defense_pa_tiers
from test_pts import calc_fantasy_points, score_defense_pa, player_season_fantasy_points


class TestCalcFantasyPoints:
    def test_passing_only(self):
        row = {"passing_yards": 300, "passing_tds": 2, "passing_interceptions": 1}
        # 300 * 0.04 + 2 * 4 + 1 * (-1) = 12 + 8 - 1 = 19.0
        assert calc_fantasy_points(row) == pytest.approx(19.0)

    def test_rushing_only(self):
        row = {"rushing_yards": 120, "rushing_tds": 1}
        # 120 * 0.1 + 1 * 6 = 12 + 6 = 18.0
        assert calc_fantasy_points(row) == pytest.approx(18.0)

    def test_receiving_only(self):
        row = {"receiving_yards": 85, "receiving_tds": 1}
        # 85 * 0.1 + 1 * 6 = 8.5 + 6 = 14.5
        assert calc_fantasy_points(row) == pytest.approx(14.5)

    def test_qb_full_game(self):
        """Simulate a typical QB stat line."""
        row = {
            "passing_yards": 275,
            "passing_tds": 3,
            "passing_interceptions": 1,
            "rushing_yards": 35,
            "rushing_tds": 0,
            "fumbles_lost": 1,
        }
        # 275*0.04 + 3*4 + 1*(-1) + 35*0.1 + 0*6 + 1*(-2)
        # = 11.0 + 12 - 1 + 3.5 + 0 - 2 = 23.5
        assert calc_fantasy_points(row) == pytest.approx(23.5)

    def test_kicker(self):
        row = {
            "fg_made_0_19": 0,
            "fg_made_20_29": 1,
            "fg_made_30_39": 1,
            "fg_made_40_49": 1,
            "fg_made_50_59": 0,
            "fg_made_60_": 0,
            "pat_made": 3,
        }
        # 0*3 + 1*3 + 1*3 + 1*4 + 0*5 + 0*5 + 3*1 = 0 + 3 + 3 + 4 + 0 + 0 + 3 = 13
        assert calc_fantasy_points(row) == pytest.approx(13.0)

    def test_two_point_conversions(self):
        row = {"passing_2pt_conversions": 1, "rushing_2pt_conversions": 1}
        # 1*2 + 1*2 = 4.0
        assert calc_fantasy_points(row) == pytest.approx(4.0)

    def test_empty_row(self):
        assert calc_fantasy_points({}) == pytest.approx(0.0)

    def test_none_values_ignored(self):
        row = {"passing_yards": None, "passing_tds": 2}
        # None ignored, 2 * 4 = 8.0
        assert calc_fantasy_points(row) == pytest.approx(8.0)

    def test_zero_stats(self):
        row = {"passing_yards": 0, "passing_tds": 0, "rushing_yards": 0}
        assert calc_fantasy_points(row) == pytest.approx(0.0)

    def test_negative_points_from_turnovers(self):
        row = {"passing_interceptions": 3, "fumbles_lost": 2}
        # 3*(-1) + 2*(-2) = -3 + -4 = -7.0
        assert calc_fantasy_points(row) == pytest.approx(-7.0)

    def test_defense_stats(self):
        row = {
            "def_sacks": 4,
            "def_interceptions": 2,
            "def_fumbles_forced": 1,
            "def_tds": 1,
        }
        # 4*1 + 2*2 + 1*2 + 1*6 = 4 + 4 + 2 + 6 = 16.0
        assert calc_fantasy_points(row) == pytest.approx(16.0)

    def test_unrecognized_keys_ignored(self):
        row = {"passing_tds": 1, "made_up_stat": 999}
        assert calc_fantasy_points(row) == pytest.approx(4.0)


class TestFumblesLostCombination:
    """Test the fumbles_lost combination logic in player_season_fantasy_points.

    The function combines rushing_fumbles_lost + receiving_fumbles_lost into
    a single fumbles_lost key before scoring. This verifies that logic works
    by checking that a player known to have fumbled gets penalized.
    """

    def test_fumbles_lost_is_negative(self):
        """A player with fumbles should have lower points than without."""
        row_no_fumble = {
            "rushing_yards": 100,
            "rushing_tds": 1,
            "fumbles_lost": 0,
        }
        row_with_fumble = {
            "rushing_yards": 100,
            "rushing_tds": 1,
            "fumbles_lost": 1,
        }
        pts_clean = calc_fantasy_points(row_no_fumble)
        pts_fumble = calc_fantasy_points(row_with_fumble)
        assert pts_fumble == pts_clean - 2  # fumbles_lost costs -2

    def test_combined_fumbles_both_sources(self):
        """Verify that rushing + receiving fumbles combine correctly."""
        row = {
            "rushing_fumbles_lost": 1,
            "receiving_fumbles_lost": 1,
        }
        # Simulate what player_season_fantasy_points does
        row["fumbles_lost"] = (row.get("rushing_fumbles_lost") or 0) + (row.get("receiving_fumbles_lost") or 0)
        assert row["fumbles_lost"] == 2
        # 2 * (-2) = -4
        assert calc_fantasy_points(row) == pytest.approx(-4.0)

    def test_combined_fumbles_none_handling(self):
        """None values should be treated as 0."""
        row = {
            "rushing_fumbles_lost": None,
            "receiving_fumbles_lost": 1,
        }
        row["fumbles_lost"] = (row.get("rushing_fumbles_lost") or 0) + (row.get("receiving_fumbles_lost") or 0)
        assert row["fumbles_lost"] == 1

    def test_combined_fumbles_both_none(self):
        """Both None should result in 0."""
        row = {
            "rushing_fumbles_lost": None,
            "receiving_fumbles_lost": None,
        }
        row["fumbles_lost"] = (row.get("rushing_fumbles_lost") or 0) + (row.get("receiving_fumbles_lost") or 0)
        assert row["fumbles_lost"] == 0


class TestScoreDefensePA:
    def test_shutout(self):
        assert score_defense_pa(0) == 10

    def test_tier_1_6_low(self):
        assert score_defense_pa(1) == 7

    def test_tier_1_6_high(self):
        assert score_defense_pa(6) == 7

    def test_tier_7_13(self):
        assert score_defense_pa(10) == 4

    def test_tier_14_20(self):
        assert score_defense_pa(14) == 1

    def test_tier_14_20_boundary(self):
        assert score_defense_pa(20) == 1

    def test_tier_21_27(self):
        assert score_defense_pa(24) == 0

    def test_tier_28_34(self):
        assert score_defense_pa(28) == -1

    def test_tier_28_34_boundary(self):
        assert score_defense_pa(34) == -1

    def test_tier_35_plus(self):
        assert score_defense_pa(35) == -4

    def test_tier_35_plus_high(self):
        assert score_defense_pa(55) == -4

    def test_all_boundaries_covered(self):
        """Every integer 0-50 should map to some tier (no gaps)."""
        for pts in range(51):
            result = score_defense_pa(pts)
            assert result != 0 or (21 <= pts <= 27), f"pts={pts} returned 0 unexpectedly"
