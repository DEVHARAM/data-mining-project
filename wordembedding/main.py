from konlpy.tag import Twitter
import matplotlib as mpl
import matplotlib.pylab as plt
#import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB #Navie Bayes
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn import svm #SVM
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import nltk
from time import time
import pickle
import matplotlib.pyplot as plt

twitter= Twitter()

def summary(model,x_train,y_train,x_test,y_test,name):
	 start = time()
	 model.fit(x_train,y_train)
	 end = time()
	 
	 data={}
	 filename = 'saved_model/'+name+'.sav'
	 pickle.dump(model, open(filename, 'wb'))
    
	 print('Time: %.2fs' %(end-start))
	 
	 y_pred = model.predict(x_train)
	 data["Train"]=accuracy_score(y_train, y_pred)
	 print("Train 정확도: {:.3f}".format(data["Train"]))
	 print(confusion_matrix(y_train, y_pred))
	 print(classification_report(y_train, y_pred, target_names=['class 0','class 2']))

	 y_pred = model.predict(x_test)
	 data["Test"]=accuracy_score(y_test, y_pred)
	 print("Test 정확도: {:.3f}".format(data["Test"]))
	 print(confusion_matrix(y_test, y_pred))
	 print(classification_report(y_test, y_pred, target_names=['class 0','class 2']))
	 
	 return data
	 
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



y_train = [ d["score"] for d in comments[100:]]
x_train = [ d["text"] for d in comments[100:]]
x_test = [ d["text"] for d in comments[:100]]
y_test = [ d["score"] for d in comments[:100]]

obj=[]

print("===========Navie Bayes===========")
multi_nbc = Pipeline([('vect', tfidf), ('nbc', MultinomialNB())])
obj.append(summary(multi_nbc,x_train,y_train,x_test,y_test,"nav"))

print("===========KNN===========")
knn = Pipeline([('vect', tfidf), ('knn', KNeighborsClassifier(n_neighbors=5))])
obj.append(summary(knn,x_train,y_train,x_test,y_test,"knn"))

print("===========SVM===========")
svm = Pipeline([('vect', tfidf), ('svc',svm.SVC(gamma="auto"))])
obj.append(summary(svm,x_train,y_train,x_test,y_test,"svm"))

print("=======Random Forest=====")
rf = Pipeline([('vect', tfidf), ('rf',RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456))])
obj.append(summary(rf,x_train,y_train,x_test,y_test,"rf"))