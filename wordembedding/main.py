from konlpy.tag import Twitter
from sklearn.naive_bayes import MultinomialNB #Navie Bayes
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from time import time
import pickle

twitter= Twitter()


def summary(model, x_train, y_train, x_test, y_test, name):
	start = time()
	model.fit(x_train, y_train)
	end = time()

	data = {}
	filename = 'saved_model/'+name+'.sav'
	pickle.dump(model, open(filename, 'wb'))

	print('Time: %.2fs' % (end-start))

	y_pred = model.predict(x_train)
	data["Train"] = accuracy_score(y_train, y_pred)
	print("Train 정확도: {:.3f}".format(data["Train"]))
	print(confusion_matrix(y_train, y_pred))
	print(classification_report(y_train, y_pred, target_names=['class 0', 'class 2']))

	y_pred = model.predict(x_test)
	data["Test"] = accuracy_score(y_test, y_pred)
	print("Test 정확도: {:.3f}".format(data["Test"]))
	print(confusion_matrix(y_test, y_pred))
	print(classification_report(y_test, y_pred, target_names=['class 0', 'class 2']))

	return data


def tokenizer_morphs(doc):
	return twitter.morphs(doc)


def tokenizer_twitter_noun(doc):
	return twitter.nouns(doc)


def tokenizer_twitter_pos(doc):
	return twitter.pos(doc, norm=True, stem=True)


def report(x_train, y_train, x_test, y_test, token, name):
	countV = CountVectorizer(tokenizer=token)
	tfidf = TfidfVectorizer(tokenizer=token)

	print("===========Navie Bayes===========")
	multi_nbc = Pipeline([('vect', countV), ('nbc', MultinomialNB())])
	summary(multi_nbc, x_train, y_train, x_test, y_test, "nav_count_"+name)

	multi_nbc = Pipeline([('vect', tfidf), ('nbc', MultinomialNB())])
	summary(multi_nbc, x_train, y_train, x_test, y_test, "nav_tfidf_" + name)

	print("=============SGD=================")
	sgd_clf = Pipeline([('vect', countV), ('sgd', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])
	summary(sgd_clf, x_train, y_train, x_test, y_test, "sgd_count_" + name)

	sgd_clf = Pipeline([('vect', tfidf), ('sgd', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42))])
	summary(sgd_clf, x_train, y_train, x_test, y_test, "sgd_tfidf_" + name)

	print("===========KNN===========")
	knn = Pipeline([('vect', countV), ('knn', KNeighborsClassifier(n_neighbors=5))])
	summary(knn, x_train, y_train, x_test, y_test, "knn_count_" + name)

	knn = Pipeline([('vect', tfidf), ('knn', KNeighborsClassifier(n_neighbors=5))])
	summary(knn, x_train, y_train, x_test, y_test, "knn_tfidf_" + name)

	"""
	print("===========SVM===========")
	svm = Pipeline([('vect', countV), ('svc',svm.SVC(gamma="auto"))])
	summary(svm,x_train,y_train,x_test,y_test,"svm")

	svm = Pipeline([('vect', tfidf), ('svc',svm.SVC(gamma="auto"))])
	summary(svm,x_train,y_train,x_test,y_test,"svm")
	"""

	print("=======Random Forest=====")
	rf = Pipeline([('vect', countV), ('rf', RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456))])
	summary(rf, x_train, y_train, x_test, y_test, "rf_count_"+ name)

	rf = Pipeline([('vect', tfidf), ('rf', RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456))])
	summary(rf, x_train, y_train, x_test, y_test, "rf_tfidf_"+ name)

emoticons = ["!","@","#","$","%","^","&","*","(",")","-","=","_","+","~",",",".","?","/",">","<","\t"]
comments = []

with open("result.txt", "r") as In:
	with open("public/first.txt", "w") as Out:
		read = In.read()
		for emoticon in emoticons:
			read=read.replace(emoticon,"")
		Out.write(read)

## Load Comment
with open("public/first.txt", "r") as f:
	for line in iter(lambda: f.readline(), ''):
		score = line[0]
		line = line[1:].replace("\n", "")
		if score == '2' or score == '0':
			comment = {"score": score, "text": line}
			comments.append(comment)

countV = CountVectorizer(tokenizer=tokenizer_morphs)
tfidf = TfidfVectorizer(tokenizer=tokenizer_morphs)


y_train = [d["score"] for d in comments[200:]]
x_train = [d["text"] for d in comments[200:]]
x_test = [d["text"] for d in comments[:200]]
y_test = [d["score"] for d in comments[:200]]

print(len(y_train))

report(x_train, y_train, x_test, y_test, tokenizer_morphs, "morphs")
report(x_train, y_train, x_test, y_test, tokenizer_twitter_noun, "noun")
report(x_train, y_train, x_test, y_test, tokenizer_twitter_pos, "pos")

