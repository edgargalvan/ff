#!/usr/bin/env ruby
require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'csv'
require 'histogram'

# projections
name = Array.new
nameid = Array.new
team = Array.new
position = Array.new
time = Array.new
fpts = Array.new

# actual values
nameid_actl = Array.new
fpts_actl = Array.new

csv = CSV.open("espn_err.csv","wb")

#cids = [0, 2, 4, 6, 16, 17, 23] # the category id for different positions
#       QB RB WR TW DST K   FLEX
cids = [0]
weeks = *(1..4)
year = 2016
i = 0
while i<cids.length
  j = 0
  while j<weeks.length
    ## Grab the actual values
    url_proj ="http://games.espn.com/ffl/tools/projections?slotCategoryId=#{cids[i]}&scoringPeriodId=#{weeks[j]}&seasonId=#{year}"
    doc_proj = Nokogiri::HTML(open(url_proj))
    
    k = 0;
    doc_proj.xpath('//tr//td[contains(concat(" ", @class, " "), " playertablePlayerName ")]').each do |node|
      name[k] = node.children.first.text
      nameid[k] = node.css("a").attribute('playerid').text
      node.children.first.remove		
      raw = node.children.first.text.delete ","
      team[k] = raw.strip[0,3]
      position[k] = raw.strip[3,5]
      k += 1;
    end

    k = 0;
    doc_proj.xpath('//tr//td[contains(concat(" ", @class, " "), " gameStatusDiv ")]//a').each do |node|
      time[k] = node.text.strip[0,3]
      k += 1;
    end

    k = 0;
    doc_proj.xpath('//tr//td[contains(concat(" ", @class, " "), " playertableStat appliedPoints sortedCell ")]').each do |node|
      fpts[k] = node.text
      k += 1;
    end

    # grab projected values
    url_actl = "http://games.espn.com/ffl/leaders?&slotCategoryId=#{cids[i]}&startIndex=0&scoringPeriodId=#{weeks[j]}&seasonId=#{year}"
    doc_actl = Nokogiri::HTML(open(url_actl))

    k = 0;
    doc_actl.xpath('//tr//td[contains(concat(" ", @class, " "), " playertablePlayerName ")]').each do |node|
      nameid_actl[k] = node.css("a").attribute('playerid').text
      k += 1;
    end
    
    k = 0;
    doc_actl.xpath('//tr//td[contains(concat(" ", @class, " "), " playertableStat appliedPoints appliedPointsProGameFinal ")]').each do |node|
      fpts_actl[k] = node.text
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
    m=0;
    while m<name.length
      nameid_index = nameid_actl.index(nameid[m])
      if nameid_index.nil?
        actl_score = '0'
      else
        actl_score = fpts_actl[nameid_index]
      end
      # actually, I only want to consider times where projections were greater than 0
      if fpts[m].to_i>0
        csv << [nameid[m],name[m], team[m], position[m], time[m], weeks[j],fpts[m],actl_score]
      end
      # TODO: what about when projection of zero was wrong?
      m += 1;
    end
    j += 1;
  end
  i += 1;
end

puts bins
