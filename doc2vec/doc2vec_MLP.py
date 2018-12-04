from collections import namedtuple
import pandas as pd
import multiprocessing
from time import time
from sklearn.neural_network import MLPClassifier
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

filename = 'saved_model/d2v.sav'
doc_vectorizer = pickle.load(open(filename, 'rb'))

X_train = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
y_train = [doc.tags for doc in tagged_train_docs]
X_test = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
y_test = [doc.tags for doc in tagged_test_docs]
print(len(X_train), len(y_train))

## sklearn neural network

mlp_clf = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=10,
    alpha=1e-4,
    solver='sgd',
    verbose=10,
    tol=1e-4,
    random_state=1,
    learning_rate_init=.1
)
start = time()
mlp_clf.fit(X_train, y_train)
end = time()
print('Time: {:f}s'.format(end-start))

y_pred = mlp_clf.score(X_test, y_test)
print("테스트 정확도: {:.2f}".format(y_pred*100))
