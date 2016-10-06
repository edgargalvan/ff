require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'csv'

team = Array.new
teamLong = Array.new
time = Array.new

fname = "gametimes.csv"
File.new fname
fid = File.open(fname, "w")
csv = CSV.new(fid)


url = "http://www.fantasypros.com/nfl/schedule.php"
doc = Nokogiri::HTML(open(url))

j = 0;
re = /(\w+)/i

doc.xpath('//*[@id="main"]/div[3]/div[5]/table/tbody/tr').each do |node|

	teamLong[j] = node.css('td')[0].css('a')[0].text
	teamLong[j+1] = node.css('td')[0].css('a')[1].text
	time[j] = node.css('td')[1].text.match re
	time[j+1] = time[j]
	j += 2;
end

# these are the names of all the teams and their abbreviations. should prolly be in another file
nameabbr = 'NYG','SEA','CAR','CIN','CLE','SF','TEN','BUF','JAX','PHI',\
'CHI','PIT','DET','ARI','WAS','TB','STL','MIA','DAL','NYJ','NO','NE',\
'SD','ATL','BAL','KC','HOU','GB','DEN','OAK','MIN','IND'

namefull = 'New York Giants','Seattle Seahawks','Carolina Panthers',\
'Cincinnati Bengals','Cleveland Browns','San Francisco 49ers','Tennessee Titans',\
'Buffalo Bills','Jacksonville Jaguars','Philadelphia Eagles','Chicago Bears',\
'Pittsburgh Steelers','Detroit Lions','Arizona Cardinals','Washington Redskins',\
'Tampa Bay Buccaneers','St. Louis Rams','Miami Dolphins','Dallas Cowboys',\
'New York Jets','New Orleans Saints','New England Patriots','San Diego Chargers',\
'Atlanta Falcons','Baltimore Ravens','Kansas City Chiefs','Houston Texans',\
'Green Bay Packers','Denver Broncos','Oakland Raiders','Minnesota Vikings','Indianapolis Colts'

k=0
while k<j
	m=0
	while teamLong[k]!=namefull[m]
		m += 1
	end
	team[k]=nameabbr[m]
	k += 1
end 
puts team
puts time

k=0
while k<j
	csv << [team[k], time[k]]
	k += 1
end

