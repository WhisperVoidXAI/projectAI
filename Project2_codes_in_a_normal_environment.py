import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
tf.compat.v1.disable_v2_behavior() 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print (len(x_train))
print (len(x_test))
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = [[i] for i in y_train]
y_test = [[i] for i in y_test]
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse_output=False)
enc.fit(y_train)
y_train =  enc.transform(y_train)
y_test = enc.transform(y_test)
n_input = 784 # input layer (28x28 pixels)
n_hidden1 = 512 # 1st hidden layer
n_hidden2 = 256 # 2nd hidden layer
n_hidden3 = 128 # 3rd hidden layer
n_output = 10 # output layer (0-9 digits)
learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5
X = tf.compat.v1.placeholder("float", [None, n_input])
Y = tf.compat.v1.placeholder("float", [None, n_output])
keep_prob = tf.compat.v1.placeholder(tf.float32)
weights = {
    'w1': tf.Variable(tf.random.truncated_normal([n_input,
    n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.random.truncated_normal([n_hidden1,
    n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.random.truncated_normal([n_hidden2,
    n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.random.truncated_normal([n_hidden3,
    n_output], stddev=0.1)),
}
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
 }
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.compat.v1.train.AdamOptimizer(1e4).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)
for i in range(n_iterations):
    startbatch = (i*batch_size) % len(x_train)
    endbatch = ((i+1)*batch_size) % len(x_train)
    batch_x = np.array(x_train[startbatch:endbatch])
    batch_x = batch_x.reshape(batch_size, -1)
    batch_y = y_train[startbatch:endbatch]  # <-- تم حذف .toarray()
    if batch_x.shape != (128, 784):
        continue
    sess.run(train_step, feed_dict={
        X: batch_x, Y: (batch_y), keep_prob: dropout})
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
        print(
            "Iteration",
            str(i),
            "\t| Loss =",
            str(minibatch_loss),
            "\t| Accuracy =",
            str(minibatch_accuracy))
x_test = x_test.reshape(-1,784)
test_accuracy = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)
img = np.array(Image.open(r"D:\MyDesktop\mnist_images\13_label6.png").convert('L')).ravel()
prediction = sess.run(tf.argmax(output_layer, 1), feed_dict={X: 
[img]})
print ("Prediction for test image:", np.squeeze(prediction)) 