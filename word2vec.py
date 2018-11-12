#-*-coding: utf-8 -*-

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import matplotlib.font_manager as fm
import matplotlib
# define training data

path = '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'
fontprop = fm.FontProperties(fname=path, size=18)

matplotlib.rc('font',family='HYsanB')
sentences=[]
f=open('test.txt',mode='rt', encoding='utf-8')
while True:
    line=f.readline()
    if not line:break
    raw=line.split(',')
    print(raw)
    sentences.append(raw)
f.close()

# train model
model = Word2Vec(sentences, min_count=1)

# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
words=list(model.wv.vocab)
print(words)

print(model.most_similar(u'한국'))

pca = PCA(n_components=2)
result = pca.fit_transform(X) #벡터화 하는 부분
print(result)

# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]),fontproperties=fontprop)
pyplot.show()
