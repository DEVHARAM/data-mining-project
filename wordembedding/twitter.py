from konlpy.tag import Twitter
import matplotlib as mpl
import matplotlib.pylab as plt
#import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import nltk
from time import time
#한글 폰트
"""
font_path = '/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
mpl.rc('font', family=font_name)
"""



twitter= Twitter()

def tokenizer_morphs(doc):
	 return twitter.morphs(doc)

emoticons = ["!","@","#","$","%","^","&","*","(",")","-","=","_","+","~",",",".","?","/",">","<","\t"]

comments=[]

with open("simple.txt","r") as In:
	 with open("public/first.txt","w") as Out:
		  read = In.read()
		  for emoticon in emoticons:
				 read=read.replace(emoticon,"")
		  Out.write(read)

## Load Comment
with open("public/first.txt","r") as f:
	 for line in iter(lambda: f.readline(),''):
		  score = line[0]
		  line = line[1:].replace("\n","")
		  if score=='2' or score=='0' :
				 comment={"score": score,"text": line}
				 comments.append(comment)

## Delete duplicated comments
tokens = [i for d in comments for t in d["text"] for i in twitter.morphs(t)]
text = nltk.Text(tokens)
print("Number of tokens : "+ str(len(text.tokens)))

print("Number of deleted duplication tokens : "+ str(len(set(text.tokens))))

top10=text.vocab().most_common(10)

print("Top 10 of tokens : "+ str(top10))

# Print Graph
"""
x,y = zip(*top10)
plt.title("Token")
plt.plot(x,y)
plt.show()
"""

tfidf = TfidfVectorizer(tokenizer=tokenizer_morphs)

multi_nbc = Pipeline([('vect', tfidf), ('nbc', MultinomialNB())])

y_train = [ d["score"] for d in comments]
x_train = [ d["text"] for d in comments]

start = time()
multi_nbc.fit(x_train,y_train)
end = time()
print('Time: %.2fs' %(end-start))

y_pred = multi_nbc.predict(x_train)
print("트레인 정확도: {:.3f}".format(accuracy_score(y_train, y_pred)))
