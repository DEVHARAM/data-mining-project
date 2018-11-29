import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.svm import SVC
import sys
sys.path.append('module')
import numpy as np
import ffp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB #Navie Bayes
from konlpy.tag import Twitter

test=[]
test.append("진짜 예쁘다")

model=pickle.load(open('save_model.sav','rb'))
pred=model.predict(test)
print(pred)

