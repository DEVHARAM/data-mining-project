from konlpy.tag import Twitter
#import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB #Navie Bayes
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn import svm #SVM
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
import nltk
import pickle
import matplotlib.pyplot as plt

twitter= Twitter()

def summary(x_train,y_train,x_test,y_test,name):
	 
	 filename="saved_model/"+name+".sav"
	 model = pickle.load(open(filename, 'rb'))
 
	 data={"Name":name}
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
	 
def tokenizer_twitter_noun(doc):
    return twitter.nouns(doc)

def tokenizer_twitter_pos(doc):
    return twitter.pos(doc, norm=True, stem=True)

comments=[]

## Load Comment
with open("public/first.txt","r") as f:
	 for line in iter(lambda: f.readline(),''):
		  score = line[0]
		  line = line[1:].replace("\n","")
		  if score=='2' or score=='0' :
				 comment={"score": score,"text": line}
				 comments.append(comment)

tfidf = TfidfVectorizer(tokenizer=tokenizer_morphs)


y_train = [ d["score"] for d in comments[200:]]
x_train = [ d["text"] for d in comments[200:]]
x_test = [ d["text"] for d in comments[:200]]
y_test = [ d["score"] for d in comments[:200]]

obj=[]

print("===========Navie Bayes===========")
obj.append(summary(x_train,y_train,x_test,y_test,"nav_count_morphs"))
obj.append(summary(x_train,y_train,x_test,y_test,"nav_count_noun"))
obj.append(summary(x_train,y_train,x_test,y_test,"nav_count_pos"))

obj.append(summary(x_train,y_train,x_test,y_test,"nav_tfidf_morphs"))
obj.append(summary(x_train,y_train,x_test,y_test,"nav_tfidf_noun"))
obj.append(summary(x_train,y_train,x_test,y_test,"nav_tfidf_pos"))

print("===========KNN===========")
obj.append(summary(x_train,y_train,x_test,y_test,"knn_count_morphs"))
obj.append(summary(x_train,y_train,x_test,y_test,"knn_count_noun"))
obj.append(summary(x_train,y_train,x_test,y_test,"knn_count_pos"))

obj.append(summary(x_train,y_train,x_test,y_test,"knn_tfidf_morphs"))
obj.append(summary(x_train,y_train,x_test,y_test,"knn_tfidf_noun"))
obj.append(summary(x_train,y_train,x_test,y_test,"knn_tfidf_pos"))
"""
print("===========SVM===========")
obj.append(summary(x_train,y_train,x_test,y_test,"svm_count_morphs"))
obj.append(summary(x_train,y_train,x_test,y_test,"svm_count_noun"))
obj.append(summary(x_train,y_train,x_test,y_test,"svm_count_pos"))

obj.append(summary(x_train,y_train,x_test,y_test,"svm_tfidf_morphs"))
obj.append(summary(x_train,y_train,x_test,y_test,"svm_tfidf_noun"))
obj.append(summary(x_train,y_train,x_test,y_test,"svm_tfidf_pos"))
"""
print("=======Random Forest=====")
obj.append(summary(x_train,y_train,x_test,y_test,"rf_count_morphs"))
obj.append(summary(x_train,y_train,x_test,y_test,"rf_count_noun"))
obj.append(summary(x_train,y_train,x_test,y_test,"rf_count_pos"))

obj.append(summary(x_train,y_train,x_test,y_test,"rf_tfidf_morphs"))
obj.append(summary(x_train,y_train,x_test,y_test,"rf_tfidf_noun"))
obj.append(summary(x_train,y_train,x_test,y_test,"rf_tfidf_pos"))

print("=============SGD=================")
obj.append(summary(x_train,y_train,x_test,y_test,"sgd_count_morphs"))
obj.append(summary(x_train,y_train,x_test,y_test,"sgd_count_noun"))
obj.append(summary(x_train,y_train,x_test,y_test,"sgd_count_pos"))

obj.append(summary(x_train,y_train,x_test,y_test,"sgd_tfidf_morphs"))
obj.append(summary(x_train,y_train,x_test,y_test,"sgd_tfidf_noun"))
obj.append(summary(x_train,y_train,x_test,y_test,"sgd_tfidf_pos"))

# create plot
plt.figure(figsize=(30, 30))
train=[value['Train'] for value in obj]
test=[value['Test'] for value in obj]
name=[value['Name'] for value in obj]

nav=[value['Test'] for value in obj if value["Name"].split("_")[0] =="nav"]
knn=[value['Test'] for value in obj if value["Name"].split("_")[0] =="knn"]
rf=[value['Test'] for value in obj if value["Name"].split("_")[0] == "rf"]
sgd=[value['Test'] for value in obj if value["Name"].split("_")[0] == "sgd"]

bottom = ["count_morphs","count_noun","count_pos","tfidf_morphs","tfidf_noun","tfidf_pos"]

plt.plot(bottom,nav,'--s',label="nav")
plt.plot(bottom,knn,'--s',label="knn")
plt.plot(bottom,rf,'--s',label="rf")
plt.plot(bottom,sgd,'--s',label="sgd")

plt.xlabel('Model')
plt.ylabel('Accuracy')

plt.title('Classifier Comment')
plt.legend()
plt.grid() 
plt.show()
