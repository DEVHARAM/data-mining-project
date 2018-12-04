import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from konlpy.tag import Twitter
from sklearn.metrics import accuracy_score
twitter= Twitter()

def tokenizer_morphs(doc):
    return twitter.morphs(doc)

array = []
with open("wordembedding/public/first.txt","r") as f:
	for line in iter(lambda:f.readline(),""):
		if(line[0] =='1' or line[0] == '2'):
			array.append(line)

data = {"score":[],"token":[]}

for index,value in enumerate(array):
	data["score"].append(value[0])
	data["token"].append(value[1:])

x_data = np.array( [value for value in data["token"] ] )
y_data = np.array( [value for value in data["score"] ] )

vect = TfidfVectorizer(tokenizer=tokenizer_morphs)

vect.fit(x_data,y_data)

x_data = vect.transform(x_data)


x_train = (x_data[100:])
x_test = (x_data[:100])
y_train = (y_data[100:])
y_test = (y_data[:100])

print("===============Multinomial==========")
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

result = accuracy_score(y_test, y_pred)
print("Test 정확도: {:.3f}".format(result))

print("==============Bernoulli=============")
model = BernoulliNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
result = accuracy_score(y_test, y_pred)
print("Test 정확도: {:.3f}".format(result))
"""
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [20,40,60,80,100,None]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
			   'max_features': max_features,
			   'max_depth': max_depth,
			   'min_samples_split': min_samples_split,
			   'min_samples_leaf': min_samples_leaf,
			   'bootstrap': bootstrap}
print(random_grid)
gs = GridSearchCV(estimator=rf, param_grid=random_grid, scoring='accuracy', cv=5, n_jobs=16)
gs = gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)

with open("log/rf_cv.txt","w") as f:
	f.write(gs.best_score_)
	f.write(gs.best_params_)
"""

