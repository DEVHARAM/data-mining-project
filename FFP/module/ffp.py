#-*- coding: cp949 -*-

"""
parsing()method export features in the comments
parsing() arguments have path and k
path argument is comment.txt file path
k is length of a feature

matrix() method make the comment A FFP Matrix
matrix() arguments have path, features and k
path argument is commnet.txt file path
features argument is features processed from parsing() method
k is length of a feature

"""
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


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
				 for num in range(1,int(len(read)/k)):
					 feature=read[num:num+k]
					 try:
						  array[count][features.index(feature)]+=1
					 except Exception:
						  1+1
					 if read[0]=='2':
						  array[count][-1]=2
					 else:
						  array[count][-1]=0
				 count+=1

	 return array

def parsing(path,k):

    features=[]
    with open(path,'r') as file_in:
        for read in iter(lambda: file_in.readline(),''):
            for num in range(1,int(len(read)/k)):
                feature=read[num:num+k]
                if not feature in features:
                    features.append(feature)
    features.append("score")
    return features

def main():
    k_range=list(range(2,4))
    p_range=list(range(0.5,1,1.5))
    for i in k_range:
        features=parsing("test.txt",i)
        result = matrix("test.txt",features,i)
        final=np.shape(result)
    '''
    print(np.shape(result))
    np.savetxt('foo.csv',result,delim
	 '''
