import nfldb
import matplotlib.pyplot as plt
import numpy as np

which_name = 'Cam Newton'
which_team = 'CAR'
which_year = 2014


db = nfldb.connect()
q = nfldb.Query(db)

# get player id
player, _ = nfldb.player_search(db, which_name, team=which_team)

# initialize
week = np.empty(0, dtype=int)
passing_yds = np.empty(0,dtype=int)
passing_yds_givenup = np.empty(0,dtype=int)

# loop through games
for i in range(2,16):
	q = nfldb.Query(db).player(player_id=player.player_id).game(season_year=which_year,season_type='Regular',week=i)

	# grab opponent information
	for pp in q.as_games():
		if pp.away_team==which_team:
			opp_team = pp.home_team
		else:
			opp_team = pp.away_team

	# get statistical info
	for pp in q.as_aggregate():
		passing_yds = np.append(passing_yds,[pp.passing_yds])
		week = np.append(week,[i])

		# this is inside loop to avoide situation where pp is empty (bye-week)
		r = nfldb.Query(db).game(season_year=which_year,season_type='Regular',week=i-1).player(team=opp_team)#prev opp game
		for pp in r.as_games():
			if pp.away_team==opp_team:
				opp_team2 = pp.home_team
			else:
				opp_team2 = pp.away_team

		s = nfldb.Query(db).game(season_year=which_year,season_type='Regular',week=i-1).player(team=opp_team2)# opp's opp
		# grap passing yards of opp's opp, for prev week. [TODO: this will bug if opp's opp had bye-week]
		pps = s.as_aggregate()
		passing_yds_givenup = np.append(passing_yds_givenup,sum(pp.passing_yds for pp in pps))
	

print passing_yds
print passing_yds_givenup

plt.plot(week,passing_yds,'-r')
plt.plot(week,passing_yds_givenup,'-b')
plt.show()
