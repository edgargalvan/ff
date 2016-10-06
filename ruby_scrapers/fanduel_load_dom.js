var page = require('webpage').create();
var url = phantom.args[0];

page.open(url);
page.onLoadFinished = function(status) {
 	console.log(page.content);
	phantom.exit();
};
