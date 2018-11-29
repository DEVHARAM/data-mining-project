from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB #Navie Bayes
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle
twitter=Twitter()
def tokenizer_morphs(doc):
     return twitter.morphs(doc)

test=["진짜 예쁘다"]

filename="save_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
pred=loaded_model.predict(test)
print(pred)
