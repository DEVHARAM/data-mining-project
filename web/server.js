var express=require('express');
var app=express();
const bodyParser=require('body-parser');
app.use(bodyParser.urlencoded({extended:false}));
app.use(bodyParser.json());

app.get('/',function(req,res){
	res.sendFile(__dirname+'/index.html');
});
app.post('/result',function(req,res){
	
	var spawn=require("child_process").spawn;
	var process=spawn('python',["./save.py",
	req.body.review]);

	process.stdout.on('data',function(data){
		console.log(data)
		res.send(data.toString());
	})
	
});
app.listen(3000);
