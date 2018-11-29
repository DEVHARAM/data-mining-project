const express=require('express');
const app=express();
const bodyParser=require('body-parser');
const ps = require('python-shell')

app.use(bodyParser.urlencoded({extended:false}));
app.use(bodyParser.json());

app.get('/',function(req,res){
	res.sendFile(__dirname+'/index.html');
});
app.post('/result',function(req,res){
	var options = {
		mode : 'text',
		pythonPath: '',
		pythonOptions: ['-u'],
		scriptPath: '',
		args: [req.body.review],
	};
	console.log(req.body.review);
	ps.PythonShell.run('load.py', options, (err,results)=>{

			if(err) throw err;
			console.log(results);
			res.send(results);
	});
});
app.listen(3000,(err)=>{
		console.log("starting http://localhost:3000");
		}
		);

