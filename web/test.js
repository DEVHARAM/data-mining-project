const ps = require('python-shell')

console.log((ps))

var options = {
	mode: 'text',
	pythonPath: '',
	pythonOptions: ['-u'],
	scriptPath: '',
	args: ['���ڴ�']
};

ps.PythonShell.run('load.py', options, function(err,results){
		if(err)  throw err;

		console.log('result: %j', results);

		});


