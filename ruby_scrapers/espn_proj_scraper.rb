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

csv = CSV.open("espn_proj.csv","wb")

#cids = [0, 2, 4, 6, 16, 17, 23] # the category id for different positions
#       QB RB WR TW DST K   FLEX
cids = [0]
weeks = *(1..5)
year = 2016
i = 0
while i<cids.length
  j = 0
  while j<weeks.length
    # for projected values
    url = "http://games.espn.com/ffl/tools/projections?slotCategoryId=#{cids[i]}&scoringPeriodId=#{weeks[j]}&seasonId=#{year}"


    doc = Nokogiri::HTML(open(url))

    k = 0;
    doc.xpath('//tr//td[contains(concat(" ", @class, " "), " playertablePlayerName ")]').each do |node|
      name[k] = node.children.first.text
      nameid[k] = node.css("a").attribute('playerid').text
		
      node.children.first.remove
		
      raw = node.children.first.text.delete ","
      team[k] = raw.strip[0,3]
      position[k] = raw.strip[3,5]
      k += 1;
    end

    k = 0;
    doc.xpath('//tr//td[contains(concat(" ", @class, " "), " gameStatusDiv ")]//a').each do |node|
      time[k] = node.text.strip[0,3]
      k += 1;
    end

    k = 0;
    doc.xpath('//tr//td[contains(concat(" ", @class, " "), " playertableStat appliedPoints sortedCell ")]').each do |node|
      fpts[k] = node.text
      k += 1;
    end

    # I want these names in the same order
    #org_inds = name.map.with_index.sort.map(&:last)
    #k = 0
    #while k<org_inds.length
    #  m = org_inds[k]
    #  csv << [nameid[m],name[m],team[m],position[m],time[m],fpts[m],weeks[j]]
    #  k += 1;
    #end
    
    m=0
    while m<k
      csv << [nameid[m],name[m], team[m], position[m], time[m], fpts[m], weeks[j]]
      m += 1
    end
    j += 1
  end
  i += 1
end
