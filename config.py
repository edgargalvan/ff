# Scoring scheme for fantasy football
# Keys match nflreadpy column names where possible

scoring = {
    # Offensive stats
    "passing_yards": 0.04,
    "passing_tds": 4,
    "passing_interceptions": -1,
    "rushing_yards": 0.1,
    "rushing_tds": 6,
    "receiving_yards": 0.1,
    "receiving_tds": 6,
    "passing_2pt_conversions": 2,
    "rushing_2pt_conversions": 2,
    "receiving_2pt_conversions": 2,
    "fumbles_lost": -2,
    "fumbles_rec_tds": 6,

    # Return TDs (player-level)
    "kickret_tds": 6,
    "puntret_tds": 6,

    # Kicking (nflreadpy has built-in FG distance buckets)
    "fg_made_0_19": 3,
    "fg_made_20_29": 3,
    "fg_made_30_39": 3,
    "fg_made_40_49": 4,
    "fg_made_50_59": 5,
    "fg_made_60_": 5,
    "pat_made": 1,

    # Defense (team-level, aggregated from player stats)
    "def_sacks": 1,
    "def_interceptions": 2,
    "def_fumbles_forced": 2,
    "def_tds": 6,
    "defense_misc_tds": 6,
    "defense_safe": 2,
    "defense_fgblk": 2,
    "defense_kickret_tds": 6,
    "defense_puntret_tds": 6,
}

# Defense points-against tiers (points allowed -> fantasy points earned)
defense_pa_tiers = [
    (0, 0, 10),      # 0 points allowed
    (1, 6, 7),       # 1-6 points allowed
    (7, 13, 4),      # 7-13 points allowed
    (14, 20, 1),     # 14-20 points allowed
    (21, 27, 0),     # 21-27 points allowed
    (28, 34, -1),    # 28-34 points allowed
    (35, 999, -4),   # 35+ points allowed
]
