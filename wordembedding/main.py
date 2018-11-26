from konlpy.tag import Twitter
import matplotlib as mpl
import matplotlib.pylab as plt
#import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB #Navie Bayes
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn import svm #SVM
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import nltk
from time import time

twitter= Twitter()

def summary(model,x_train,y_train,x_test,y_test):
	 start = time()
	 model.fit(x_train,y_train)
	 end = time()

	 print('Time: %.2fs' %(end-start))
	 
	 y_pred = model.predict(x_train)
	 print("Train 정확도: {:.3f}".format(accuracy_score(y_train, y_pred)))
	 print(confusion_matrix(y_train, y_pred))
	 print(classification_report(y_train, y_pred, target_names=['class 0','class 2']))

	 y_pred = model.predict(x_test)
	 print("Test 정확도: {:.3f}".format(accuracy_score(y_test, y_pred)))
	 print(confusion_matrix(y_test, y_pred))
	 print(classification_report(y_test, y_pred, target_names=['class 0','class 2']))

def tokenizer_morphs(doc):
	 return twitter.morphs(doc)

emoticons = ["!","@","#","$","%","^","&","*","(",")","-","=","_","+","~",",",".","?","/",">","<","\t"]

comments=[]

with open("simple.txt","r") as In:
	 with open("public/first.txt","w") as Out:
		  read = In.read()
		  for emoticon in emoticons:
				 read=read.replace(emoticon,"")
		  Out.write(read)

## Load Comment
with open("public/first.txt","r") as f:
	 for line in iter(lambda: f.readline(),''):
		  score = line[0]
		  line = line[1:].replace("\n","")
		  if score=='2' or score=='0' :
				 comment={"score": score,"text": line}
				 comments.append(comment)

tfidf = TfidfVectorizer(tokenizer=tokenizer_morphs)

multi_nbc = Pipeline([('vect', tfidf), ('nbc', MultinomialNB())])

y_train = [ d["score"] for d in comments[100:]]
x_train = [ d["text"] for d in comments[100:]]
x_test = [ d["text"] for d in comments[:100]]
y_test = [ d["score"] for d in comments[:100]]

print("===========Navie Bayes===========")
summary(multi_nbc,x_train,y_train,x_test,y_test)

"""
knn = Pipeline([('vect', tfidf), ('knn', KNeighborsClassifier(n_neighbors=3))])

print("===========KNN===========")
summary(knn,x_train,y_train,x_test,y_test)

print("===========SVM===========")
svm = Pipeline([('vect', tfidf), ('svc',svm.SVC(gamma='auto'))])
summary(svm,x_train,y_train,x_test,y_test)
"""
