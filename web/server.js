var express=require('express');
var app=express();
var fs=require('fs');
const bodyParser=require('body-parser');
app.use(bodyParser.urlencoded({extended:false}));
app.use(bodyParser.json());

app.get('/',function(req,res){
	res.sendFile(__dirname+'/index.html');
});
app.post('/',function(req,res){
	
	var spawn=require("child_process").spawn;
	var process=spawn('python',["./save.py",req.body.review]);//save.py로 review값 보내기
	var button=('<input type=button value="돌아가기" onclick="history.back(-1)">')

	process.stdout.on('data',function(data){ //save.py에서 print내용 가져와서 localhost로 send함
		console.log(data)
		res.send(data.toString()+<br>+button);
	})
	
	
});
app.listen(3000);
