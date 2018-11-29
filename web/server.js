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
	var process=spawn('python',["./save.py",req.body.review]);//save.py로 review값 보내기

	process.stdout.on('data',function(data){ //save.py에서 print내용 가져와서 localhost로 send함
		console.log(data)
		res.send(data.toString());
	})
	
});
app.listen(3000);
