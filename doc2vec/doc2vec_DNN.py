from collections import namedtuple
import pandas as pd
from konlpy.tag import Twitter
import multiprocessing
from gensim.models import Doc2Vec
from time import time
import numpy as np
import tensorflow as tf

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
    "score": [d["score"] for d in comments[100:]],
    "text": [d["text"] for d in comments[100:]],
})

df_test = pd.DataFrame({
    "X_test": [d["text"] for d in comments[:100]],
    "y_test": [d["score"] for d in comments[:100]],
})

df_train['token_review'] = df_train['text'].apply(tokenizer_morphs)
df_test['X_test_tokkended'] = df_test['X_test'].apply(tokenizer_morphs)
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
    doc_vectorizer.alpha -= 0.002 # decrease the learning rate
    doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate, no decay
end = time()
print("During Time: {}".format(end-start))

model_name = 'Doc2Vec(dbow+w,d40,n10,hs,w8,mc20,s0.001,t8).model'

doc_vectorizer.save(model_name)
#
doc_vectorizer = Doc2Vec.load(model_name)

# print(doc_vectorizer.wv.most_similar('가족'))

X_train = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
y_train = [doc.tags for doc in tagged_train_docs]
X_test = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
y_test = [doc.tags for doc in tagged_test_docs]
print(len(X_train), len(y_train))
####################################################################################################
y_train_np = np.asarray(y_train, dtype=int)
y_test_np = np.asarray(y_test, dtype=int)
X_train_np = np.asarray(X_train)
X_test_np = np.asarray(X_test)

y_train_np = np.eye(3)[y_train_np.reshape(-1)]
y_test_np = np.eye(3)[y_test_np.reshape(-1)]

# # Xavier_Initializer
xavier_init = tf.contrib.layers.xavier_initializer()
###################################################################################################
##deep neural learning

tf.reset_default_graph()

# hyper Parameter
learning_rate = 0.001
training_epochs = 50
batch_size = 20

# input layer
X = tf.placeholder(tf.float32, [None, 40])
Y = tf.placeholder(tf.float32, [None, 3])

# dropout
keep_prob = tf.placeholder(tf.float32)

# Hidden layers and Output layer
W1 = tf.get_variable("W1", shape=[40, 32], initializer=xavier_init)
b1 = tf.Variable(tf.random_normal([32]))
L1 = tf.nn.relu(tf.matmul(X, W1)+b1)
dropout1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[32, 32], initializer=xavier_init)
b2 = tf.Variable(tf.random_normal([32]))
L2 = tf.nn.relu(tf.matmul(dropout1, W2)+b2)
dropout2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[32, 32], initializer=xavier_init)
b3 = tf.Variable(tf.random_normal([32]))
L3 = tf.nn.relu(tf.matmul(dropout2, W3)+b3)
dropout3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[32, 32], initializer=xavier_init)
b4 = tf.Variable(tf.random_normal([32]))
L4 = tf.nn.relu(tf.matmul(dropout3, W4)+b4)
dropout4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[32, 3], initializer=xavier_init)
b5 = tf.Variable(tf.random_normal([3]))
hypothesis = tf.matmul(dropout4, W5)+b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train Model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(X_train_np) / batch_size)

    for i in range(0, len(X_train_np), batch_size):
        batch_xs = X_train_np[i:i+batch_size]
        batch_ys = y_train_np[i:i+batch_size]

        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '{:04d}'.format(epoch +1), 'cost =', '{:.9f}'.format(avg_cost))

print('Training Finished')

# Test Model and check Accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('테스트 정확도: ', sess.run(accuracy, feed_dict={X: X_test_np, Y: y_test_np, keep_prob: 1}))
