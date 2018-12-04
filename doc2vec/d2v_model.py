from collections import namedtuple
import pandas as pd
from konlpy.tag import Twitter
import multiprocessing
from gensim.models import Doc2Vec
from time import time
from sklearn.linear_model import LogisticRegression
import pickle

cores = multiprocessing.cpu_count()
twitter = Twitter()


def tokenizer_morphs(doc):
    return twitter.morphs(doc)


emoticons = ["!","@","#","$","%","^","&","*","(",")","-","=","_","+","~",",",".","?","/",">","<","\t"]

comments = []

with open("simple.txt", "r", encoding='UTF8') as In:
    with open("public/first.txt", "w", encoding='UTF8') as Out:
        read = In.read()
        for emoticon in emoticons:
            read = read.replace(emoticon, "")
        Out.write(read)
# Load Comment
with open("public/first.txt", "r", encoding='UTF8') as f:
    for line in iter(lambda: f.readline(), ''):
        score = line[0]
        line = line[1:].replace("\n", "")
        if score == '2' or score == '0':
            comment = {"score": score, "text": line}
            comments.append(comment)

df_train = pd.DataFrame({
    "score": [d["score"] for d in comments[200:]],
    "text": [d["text"] for d in comments[200:]],
})
df_test = pd.DataFrame({
    "X_test": [d["text"] for d in comments[:200]],
    "y_test": [d["score"] for d in comments[:200]],
})
df_train['token_review'] = df_train['text'].apply(tokenizer_morphs)
df_test['X_test_tokkended'] = df_test['X_test'].apply(tokenizer_morphs)

df_train.to_csv("saved_model/df_train.csv", mode='w', encoding='UTF8')
df_test.to_csv("saved_model/df_test.csv", mode='w', encoding='UTF-8')

TaggedDocument = namedtuple('TaggedDocument', 'words tags')

tagged_train_docs = [TaggedDocument(d, c)
                     for d, c in df_train[['token_review', 'score']].values]
tagged_test_docs = [TaggedDocument(d, c)
                    for d, c in df_test[['X_test_tokkended', 'y_test']].values]

print(len(tagged_test_docs), len(tagged_train_docs))

doc_vectorizer = Doc2Vec(
    dm=0,            # PV-DBOW / default 1
    dbow_words=1,    # w2v simultaneous with DBOW d2v / default 0
    window=8,        # distance between the predicted word and context words
    vector_size=40,  # vector size
    alpha=0.025,     # learning-rate
    seed=1234,
    min_count=20,    # ignore with freq lower
    min_alpha=0.025, # min learning-rate
    workers=cores,   # multi cpu
    hs=1,            # hierarchical softmax / default 0
    negative=10,     # negative sampling / default 5
)

doc_vectorizer.build_vocab(tagged_train_docs)
print(str(doc_vectorizer))
start = time()
for epoch in range(10):
    doc_vectorizer.train(tagged_train_docs, total_examples=doc_vectorizer.corpus_count, epochs=doc_vectorizer.iter)
    doc_vectorizer.alpha -= 0.002  # decrease the learning rate
    doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate, no decay
end = time()
print("During Time: {}".format(end-start))

print(tagged_train_docs)
filename = 'saved_model/d2v.sav'
pickle.dump(doc_vectorizer, open(filename, 'wb'))

print("model has saved!")

