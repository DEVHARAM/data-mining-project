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


y_train = [ d["score"] for d in comments[100:]]
x_train = [ d["text"] for d in comments[100:]]
x_test = [ d["text"] for d in comments[:100]]
y_test = [ d["score"] for d in comments[:100]]

obj=[]

print("===========Navie Bayes===========")
obj.append(summary(x_train,y_train,x_test,y_test,"nav"))

print("===========KNN===========")
obj.append(summary(x_train,y_train,x_test,y_test,"knn"))

print("===========SVM===========")
obj.append(summary(x_train,y_train,x_test,y_test,"svm"))

print("=======Random Forest=====")
obj.append(summary(x_train,y_train,x_test,y_test,"rf"))

# create plot
fig, ax = plt.subplots()
index = np.arange(len(obj))
bar_width = 0.35
opacity = 0.8

train=[value['Train'] for value in obj]
test=[value['Test'] for value in obj]
name=[value['Name'] for value in obj]

rects1 = plt.bar(index, train, bar_width,
                 alpha=opacity,
                 color='b',
                 label='train')
 
rects2 = plt.bar(index + bar_width, test, bar_width,
                 alpha=opacity,
                 color='g',
                 label='test')
 
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('TF-IDF')
plt.xticks(index + bar_width/2, name)
plt.legend()
plt.grid() 
plt.tight_layout()
plt.show()
