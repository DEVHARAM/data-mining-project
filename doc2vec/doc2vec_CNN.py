from collections import namedtuple
import pandas as pd
from konlpy.tag import Twitter
import multiprocessing
from gensim.models import Doc2Vec
from time import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.contrib.framework import arg_scope

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

#####################################################################################################
## CNN

y_train_np = np.asarray(y_train, dtype=int)
y_test_np = np.asarray(y_test, dtype=int)
X_train_np = np.asarray(X_train)
X_test_np = np.asarray(X_test)

y_train_np = np.eye(3)[y_train_np.reshape(-1)]
y_test_np = np.eye(3)[y_test_np.reshape(-1)]

# # Xavier_Initializer
xavier_init = tf.contrib.layers.xavier_initializer()

# hyper parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 10
keep_prob = 0.7

# Input Layer
X = tf.placeholder(tf.float32, [None, 40])
Y = tf.placeholder(tf.float32, [None, 3])
train_mode = tf.placeholder(tf.bool, name='train_mode')

# Layer output size
hidden_output_size = 40
final_ouput_size = 3

bn_params = {
    'is_training': train_mode,
    'decay':0.9,
    'updates_collections':None
}

with arg_scope(
        [fully_connected],
        activation_fn=tf.nn.relu,
        weights_initializer=xavier_init,
        biases_initializer=None,
        normalizer_fn=batch_norm,
        normalizer_params=bn_params):

    h1 = fully_connected(X, hidden_output_size, scope='h1')
    dropout1 = dropout(h1, keep_prob, is_training=train_mode)

    h2 = fully_connected(dropout1, hidden_output_size, scope='h2')
    dropout2 = dropout(h2, keep_prob, is_training=train_mode)

    h3 = fully_connected(dropout2, hidden_output_size, scope='h3')
    dropout3 = dropout(h3, keep_prob, is_training=train_mode)

    h4 = fully_connected(dropout3, hidden_output_size, scope='h4')
    dropout4 = dropout(h4, keep_prob, is_training=train_mode)

    hypothesis = fully_connected(dropout4, final_ouput_size, activation_fn=None, scope='hypothesis')

    # define Cost/Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(X_train_np) / batch_size)

        for i in range(0, len(X_train_np), batch_size):
            batch_xs = X_train_np[i:i + batch_size]
            batch_ys = y_train_np[i:i + batch_size]

            feed_dict_train = {X: batch_xs, Y: batch_ys, train_mode: True}
            feed_dict_cost = {X: batch_xs, Y: batch_ys, train_mode: False}

            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict_train)
            avg_cost += c / total_batch
        print("Epoch: {:4d} cost={:.9f}".format(epoch + 1, avg_cost))

    print("Learning Finished")

    # Test Model and Check Accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('Accuracy:', sess.run(accuracy, feed_dict={X: X_test_np, Y: y_test_np, train_mode: False}))
