#!/usr/bin/env ruby
require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'csv'

name = Array.new
nameid = Array.new
team = Array.new
position = Array.new
time = Array.new
fpts = Array.new

fname = "fantasydata.csv"
File.new fname
fid = File.open(fname, "w")
csv = CSV.new(fid)


cids = [0, 2, 4, 6, 16, 17] # the category id for different positions
i = 0
while i<cids.length

	url = "http://games.espn.go.com/ffl/tools/projections?slotCategoryId=#{cids[i]}"
	i += 1

	doc = Nokogiri::HTML(open(url))

	j = 0;
	doc.xpath('//tr//td[contains(concat(" ", @class, " "), " playertablePlayerName ")]').each do |node|
		name[j] = node.children.first.text
		nameid[j] = node.css("a").attribute('playerid')
		
		node.children.first.remove
		
		raw = node.children.first.text.delete ","
		team[j] = raw.strip[0,3]
		position[j] = raw.strip[3,5]
		j += 1;
	end

	j = 0;
	doc.xpath('//tr//td[contains(concat(" ", @class, " "), " gameStatusDiv ")]//a').each do |node|
		time[j] = node.text.strip[0,3]
		j += 1;
	end

	j = 0;
	doc.xpath('//tr//td[contains(concat(" ", @class, " "), " playertableStat appliedPoints sortedCell ")]').each do |node|
		fpts[j] = node.text
		j += 1;
	end


	k=0
	while k<j
		csv << [nameid[k],name[k], team[k], position[k], time[k], fpts[k]]
		k += 1
	end

end
