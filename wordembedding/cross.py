from konlpy.tag import Twitter
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier #KNN
from time import time
from sklearn.model_selection import GridSearchCV
import numpy as np
import os

twitter = Twitter()


def summary(model, x_train, y_train, x_test, y_test, name):
	start = time()
	model.fit(x_train, y_train)
	end = time()

	data = {}
	#filename = 'saved_model/'+name+'.sav'
	#pickle.dump(model, open(filename, 'wb'))

	print('Time: %.2fs' %(end-start))

	y_pred = model.predict(x_train)
	data["Train"]=accuracy_score(y_train, y_pred)
	print("Train 정확도: {:.3f}".format(data["Train"]))

	y_pred = model.predict(x_test)
	data["Test"]=accuracy_score(y_test, y_pred)
	print("Test 정확도: {:.3f}".format(data["Test"]))

	return data


def tokenizer_morphs(doc):
	return twitter.morphs(doc)


def report(x_train, y_train, x_test, y_test, token, name):
	countV = CountVectorizer(tokenizer=token)
	tfidf = TfidfVectorizer(tokenizer=token)

	print("=============SGD=================")
	sgd_clf = Pipeline([('vect', countV), ('sgd', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])
	summary(sgd_clf, x_train, y_train, x_test, y_test, "sgd_count_"+name)

	print("=======Random Forest=====")

	rf = Pipeline([('vect', tfidf), ('rf', RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456))])
	summary(rf, x_train, y_train, x_test, y_test, "rf_tfidf_"+name)

emoticons = ["!","@","#","$","%","^","&","*","(",")","-","=","_","+","~",",",".","?","/",">","<","\t"]
comments = []

prepro = os.path.abspath(__file__ + "/../../") + "/prepro/second.txt"
with open(prepro, "r", encoding="utf8") as In:
	with open("public/third.txt", "w", encoding='utf8') as Out:
		read = In.read()
		for emoticon in emoticons:
			read = read.replace(emoticon, "")
		Out.write(read)

## Load Comment
with open("public/third.txt", "r", encoding='utf8') as f:
	for line in iter(lambda: f.readline(), ''):
		score = line[0]
		line = line[1:].replace("\n", "")
		if score == '2' or score == '0':
			comment = {"score": score, "text": line}
			comments.append(comment)

countV = CountVectorizer(tokenizer=tokenizer_morphs)
tfidf = TfidfVectorizer(tokenizer=tokenizer_morphs)


y_train = np.array([ d["score"] for d in comments[200:]])
x_train = np.array([ d["text"] for d in comments[200:]])
x_test = np.array([ d["text"] for d in comments[:200]])
y_test = np.array([ d["score"] for d in comments[:200]])


knn = Pipeline([('vect', countV), ('knn', KNeighborsClassifier(n_neighbors=5))])

param_range = [3, 5, 7, 9]
param_grid = [
	{'knn__n_neighbors': param_range}]

gs = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=1)
gs = gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

#report(x_train,y_train,x_test,y_test,tokenizer_morphs,"morphs")

