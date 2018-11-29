#-*- coding: utf-8 -*-

import numpy as np
def convert_index(path1,path2,p):
	 delete = ["!","@","#","$","%","^","&","*","(",")","-","=","_","+","~",",",".","?","/",">","<"," ","\t"]
	 
	 max_indexs = []
	 head =[]
	 count =[]
	 read = ""
	 with open(path1,'r') as f:
		  read=f.read()
		  for d in delete:
				 read=read.replace(d,'')

	 with open(path2,'w') as f:
		  f.write(read)

	 with open(path2,'r') as f:
		  for read in iter(lambda: f.readline(),''):
				 for word in read:
					   if word is '\n':
						    1+1
					   elif not word in head:
						    head.append(word)
						    count.append(1)
					   else:
						    count[head.index(word)]+=1

	 while True:
		  if max(count) >= (len(head)*p):
				 max_index=count.index(max(count))
				 max_indexs.append(max_index)
				 count[max_index]=-1
		  else:
				 break
	 result=[head[value] for value in max_indexs]
	 return result


if __name__ == '__main__':
   print(convert_index(0.3))

