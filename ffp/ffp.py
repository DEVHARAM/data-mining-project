#-*- coding: cp949 -*-
import numpy as np

def matrix(path,features,k):
	
	num_lines=0

	#count line
	with open(path, 'r') as f:
		for line in f:
			num_lines += 1
	
	array = np.zeros([num_lines,len(features)],dtype='i')
	
	count=0
	with open(path,'r') as file_in:
		for read in iter(lambda: file_in.readline(),''):

			for num in range(0,int(len(read)/k)):
				feature=read[num:num+k]
				array[count][features.index(feature)]+=1
			
			count+=1
				
		
	return array

def parsing(path,k):

	features=[]
	with open(path,'r') as file_in:
		for read in iter(lambda: file_in.readline(),''):
			for num in range(0,int(len(read)/k)):
				feature=read[num:num+k]
				if not feature in features:
					features.append(feature)

	return features	
	
def main():
	features=parsing("./test.txt",3)
	result=matrix("./test.txt",features,3)
	print(features)
	print(result)

if __name__ == "__main__":
	main()

