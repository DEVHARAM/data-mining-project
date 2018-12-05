from collections import namedtuple
import pandas as pd
import multiprocessing
from time import time
from sklearn.linear_model import LogisticRegression
import pickle

cores = multiprocessing.cpu_count()

comments = []
TaggedDocument = namedtuple('TaggedDocument', 'words tags')

df_train = pd.read_csv("saved_model/df_train.csv", index_col=0)
df_test = pd.read_csv("saved_model/df_test.csv", index_col=0)

tagged_train_docs = [TaggedDocument(d, c)
                     for d, c in df_train[['token_review', 'score']].values]
tagged_test_docs = [TaggedDocument(d, c)
                    for d, c in df_test[['X_test_tokkended', 'y_test']].values]
print(tagged_train_docs, len(tagged_train_docs))
filename = 'saved_model/d2v.sav'
doc_vectorizer = pickle.load(open(filename, 'rb'))

X_train = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
y_train = [doc.tags for doc in tagged_train_docs]
X_test = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
y_test = [doc.tags for doc in tagged_test_docs]
print(len(X_train), len(y_train))

#####################################################################################################
## logistic regression

clf = LogisticRegression(random_state=1)
start = time()
clf.fit(X_train, y_train)
end = time()
print('Time: {:f}s'.format(end-start))

y_pred = clf.score(X_test, y_test)
print("====Logistic Regression====")
print("테스트 정확도: {:.2f}%".format(y_pred*100))
