#-*-coding: utf-8 -*-

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
sentences = [['나는','한국인','입니다'],
['당신은','한국인','입니까?'],
['나는','한국에서','잘','지내요'],
['한국','노래를','잘','좋아해요']]
# train model
model = Word2Vec(sentences, min_count=1)

# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
words=list(model.wv.vocab)
print(words)

pca = PCA(n_components=2)
result = pca.fit_transform(X) #벡터화 하는 부분
print(result)


