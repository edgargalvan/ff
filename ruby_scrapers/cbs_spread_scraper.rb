#!/usr/bin/env ruby
require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'csv'

home_team = Array.new
spread = Array.new

csv = CSV.open("cbs_spread_2017.csv","wb",
               :write_headers=> true,
               :headers => ["week","home_team","spread"]
              )
 

weeks = *(1..18)

weeks.each do |week|
  url = "https://www.cbssports.com/nfl/features/writers/expert/picks/against-the-spread/#{week}"

  doc = Nokogiri::HTML(open(url))
  puts "week #{week}"
  doc.xpath("//*[@id='oddsTable']/tr/td[4]/div[contains(@class, 'gameLineCtr')]").each do |node|
    spread = node.css('span')[0].text
    home_team = node.css('span')[1].text
    puts spread+' for '+home_team
    csv << [week, home_team, spread]
  end
end
