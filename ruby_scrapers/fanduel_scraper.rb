#!/usr/bin/env ruby
require 'rubygems'
require 'phantomjs'
require 'nokogiri'
require 'open-uri'
require 'csv'

playername = Array.new
playerteam = Array.new
playerid = Array.new
position = Array.new
playercost = Array.new

script = File.expand_path('./fanduel_load_dom.js')
url = 'https://www.fanduel.com/games/13347/contests/13347-17305107/enter';
puts 'oppening page'
p = Phantomjs.run(script,url)
puts 'done'
doc = Nokogiri::HTML(p)
puts p

i = 0
#doc.xpath('//table[contains(concat(" ", @class, " "), " condensed player-list-table ")]//tbody//tr').each do |node|
doc.xpath('//*[@id="ui-skeleton"]/div/section/div[2]/div[4]/div[1]/section[2]/div[2]/table/tbody/tr').each do |node|
puts node
position[i] = node.children.first.text
playername[i] = node.css('div').text
playerid[i] = node.css('.button').attribute('data-player-id')
playercost[i] = node.children[10].text.delete ","
playerteam[i] = node.children[8].css('b').text
i += 1
end
puts playername

fname = "fandueldata.csv"
File.new fname
fid = File.open(fname, "w")
csv = CSV.new(fid)

k=0
while k<i
	csv << [playerid[k], playername[k], playerteam[k], position[k], playercost[k]]
	k += 1
end
fid.close
