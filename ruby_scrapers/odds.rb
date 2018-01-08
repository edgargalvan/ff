#!/usr/bin/env ruby
require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'csv'

home_team = Array.new
spread = Array.new

csv = CSV.open("spread.csv","wb",
               :write_headers=> true,
               :headers => ["year","week","home_team","away_team","home_spread","away_spread"]
              )
 

weeks = *(1..17)
years = *(2013..2017)

years.each do |year|
  weeks.each do |week|
    url = "https://fantasysupercontest.com/nfl-lines-#{year}-week-#{week}?year=#{year}&line_option=both&team_option=acronym&record_option=ats&selnum=on"

    doc = Nokogiri::HTML(open(url))
    puts "week #{week}"
    doc.xpath("/html/body/div/div/div/div/div/div/div/div/table[contains(@class, 'table table-hover table-bordered table-striped schedule-table')]/tr").each do |node|
      away_team = node.css('td[1]').text.gsub("\n", ' ').squeeze(' ').tr(')', '').split("(")
      away_score = node.css('td[2]').text.gsub("\n", ' ').squeeze(' ')
      home_team = node.css('td[4]').text.gsub("\n", ' ').squeeze(' ').tr(')', '').split("(")
      home_score = node.css('td[5]').text.gsub("\n", ' ').squeeze(' ')
      if home_team.any?
        puts [year, week, home_team[0], away_team[0], home_team[1], away_team[1]]
        csv << [year, week, home_team[0], away_team[0], home_team[1], away_team[1]]
      end
    end
  end
end
