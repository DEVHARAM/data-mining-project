from collections import namedtuple
import pandas as pd
import multiprocessing
from time import time
import numpy as np
import tensorflow as tf
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
print(len(X_test), len(X_train))
####################################################################################################
y_train_np = np.asarray(y_train, dtype=int)
y_test_np = np.asarray(y_test, dtype=int)
X_train_np = np.asarray(X_train)
X_test_np = np.asarray(X_test)

y_train_np = np.eye(2)[y_train_np.reshape(-1)]
y_test_np = np.eye(2)[y_test_np.reshape(-1)]

# # Xavier_Initializer
xavier_init = tf.contrib.layers.xavier_initializer()
###################################################################################################
##deep neural learning

tf.reset_default_graph()

# hyper Parameter
learning_rate = 0.001
training_epochs = 50
batch_size = 9

# input layer
X = tf.placeholder(tf.float32, [None, 50])
Y = tf.placeholder(tf.float32, [None, 2])

# dropout
keep_prob = tf.placeholder(tf.float32)

# Hidden layers and Output layer
W1 = tf.get_variable("W1", shape=[50, 32], initializer=xavier_init)
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

W5 = tf.get_variable("W5", shape=[32, 2], initializer=xavier_init)
b5 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(dropout4, W5)+b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train Model
start = time()
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(X_train_np) / batch_size)

    for i in range(0, len(X_train_np), batch_size):
        batch_xs = X_train_np[i:i+batch_size]
        batch_ys = y_train_np[i:i+batch_size]

        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '{:04d}'.format(epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

end = time()
print('Training Finished')
print('Time: {:f}s'.format(end-start))

# Test Model and check Accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("====Deep Neural Network====")
print("테스트 정확도: {:.2f}%".format((sess.run(accuracy,
                                          feed_dict={X: X_test_np, Y: y_test_np, keep_prob: 1})) * 100))
