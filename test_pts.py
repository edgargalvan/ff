import nfldb
import numpy as np
from config import *

class ff_stats:
    def __init__(self,name,team,season_type,season_year):

        # player info
        self.full_name = ''
        self.position = ''
        self.team_name = ''
        self.player_id = ''
            
        # stats we wish to track for ff
        self.passing_yds = 0
        self.passing_tds = 0
        self.rushing_yds = 0
        self.rushing_tds = 0
        self.receiving_yds = 0
        self.receiving_tds = 0
        self.kickret_tds = 0
        self.puntret_tds = 0
        self.passing_twoptm = 0
        self.rushing_twoptm = 0
        self.receiving_twoptm = 0
        self.fumbles_lost = 0
        self.fumbles_rec_tds = 0
        self.kicking_xpmade = 0

        # kicker
        self.kicking_fgm_0_39 = 0
        self.kicking_fgm_40_49 = 0
        self.kicking_fgm_50_100 = 0

        # defense
        self.defense_sk = 0
        self.defense_int = 0
        self.defense_frec = 0
        self.defense_int_tds = 0
        self.defense_misc_tds = 0
        self.defense_safe = 0
        self.defense_fgblk = 0

        # defense points against (not sure how to handle this just yet)
        self.defense_pa_35_100 = 0
        self.defense_pa_28_34 = 0
        self.defense_pa_21_27 = 0
        self.defense_pa_14_20 = 0
        self.defense_pa_7_13 = 0
        self.defense_pa_1_6 = 0
        self.defense_pa_0_0 = 0

        # games we're interested in
        games = q.game(season_year=season_year, season_type=season_type, team=team)

        # find the player and games we care about
        if name=='DEF':
            self.full_name = 'DEF'
            self.position = 'DEF'
            self.team = team
    
            #subset of games played by team
            player_games = games.play_player(team=team)
        else:
            dummy, _ = nfldb.player_search(db, name, team=team)
            # subset of games played by player
            player_games = games.player(player_id=dummy.player_id)

            for attr, value in self.__dict__.iteritems():
                if hasattr(dummy, attr):
                    setattr(self, attr, getattr(dummy,attr)) 
        
        for gg in player_games.as_games():
            # add game statistics
            self.update_game(gg)
            for pp in gg.play_players:
                # add play statistics
                if self.full_name=='DEF':
                    if pp.team==team:
                        self.update_play(pp)
                else:
                    if pp.player_id==self.player_id:
                        self.update_play(pp)
    
    # update with nfldb play statistics
    def update_play(self,obj):
        # update matching statistics
        for attr, value in self.__dict__.iteritems():
            if hasattr(obj, attr):
                if isinstance(getattr(obj,attr), (int, float)):
                    new_value = getattr(self,attr) + getattr(obj,attr) # temp var
                    setattr(self, attr, new_value) 
        # update kicking stats
        if hasattr(obj,'kicking_fgm_yds'):
            if obj.kicking_fgm_yds >=50:
                self.kicking_fgm_50_100 += 1
            elif obj.kicking_fgm_yds >=40:
                self.kicking_fgm_40_49 += 1
            elif obj.kicking_fgm_yds > 0:
                self.kicking_fgm_0_39 += 1
        return self

    # update with nfldb game statistics (only game stat of interest is final score)
    def update_game(self,obj):
        # only keep track of this if player is DEF
        if self.full_name=='DEF':
            if obj.away_team==self.team:
                pts  = obj.home_score
            else:
                pts = obj.away_score

            if pts >=35:
                self.defense_pa_35_100 += 1
            elif pts >=28:
                self.defense_pa_28_34 += 1
            elif pts >=21:
                self.defense_pa_21_27 += 1
            elif pts >=14:
                self.defense_pa_14_20 += 1
            elif pts >=7:
                self.defense_pa_7_13 += 1
            elif pts >=1:
                self.defense_pa_1_6 += 1
            else:
                self.defense_pa_0_0 += 1
                
        return self


# start up db
db = nfldb.connect()
q = nfldb.Query(db)

# input
name = 'Matt Ryan'
team = 'ATL'
season_type = 'Regular'
season_year = 2015

a = ff_stats(name,team,season_type,season_year)
print a.passing_yds
# Alternatively if we just want the aggregated stats, it's pretty easy
#player_games_agg = player_games.as_aggregate()[0]
#print player_games_agg.passing_yds

# do this if player is defense

# defense_kickret_tds
# defense_puntret_tds
