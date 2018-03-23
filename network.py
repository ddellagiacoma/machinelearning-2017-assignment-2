import sys
import numpy as np
from numpy import argmax
import tensorflow as tf
from sklearn.model_selection import KFold

TRAIN_DATA = "data/train-data.csv"
TRAIN_TARGETS= "data/train-target.csv"
TEST_DATA = "data/test-data.csv"

def one_hot_encode(data):
	# define universe of possible input values
	alphabet = 'abcdefghijklmnopqrstuvwxyz'
	# define a mapping of chars to integers
	char_to_int = dict((c, i) for i, c in enumerate(alphabet))
	# integer encode input data
	integer_encoded = [char_to_int[char] for char in data]
	# one hot encode
	onehot_encoded = list()
	for value in integer_encoded:
		letter = [0 for _ in range(len(alphabet))]
		letter[value] = 1
		onehot_encoded.append(letter)
	return(np.array(onehot_encoded))

def one_hot_decode(data):
	# define universe of possible input values
	alphabet = 'abcdefghijklmnopqrstuvwxyz'
	# define a mapping of integers to chars
	int_to_char = dict((i, c) for i, c in enumerate(alphabet))
	# one hot encode
	onehot_decoded = list()
	for i in range(len(data)):
		onehot_decoded.append(int_to_char[argmax(data[i])])
	return(np.array(onehot_decoded))

def next_batch(num, data, labels):
	# Return a total of 'num' random samples and labels
	idx = np.arange(0 , len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[i] for i in idx]
	labels_shuffle = [labels[i] for i in idx]
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# Initializes a weight variable by sampling from a truncated normal distribution with standard deviation of 0.1
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# Initializes a bias variable with a constant value 0.1
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=False)

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':

	with open(TRAIN_DATA) as f:
		X_train = np.loadtxt(f, delimiter=',')
	f.close()

	with open(TRAIN_TARGETS) as f:
		y_labels = np.loadtxt(f, dtype='unicode_')
	f.close()
	# Transform the train labels in one-hot array
	y_train = one_hot_encode(y_labels)

	with open(TEST_DATA) as f:
		X_test = np.loadtxt(f, delimiter=',')
	f.close()

	x = tf.placeholder(tf.float32, [None, 128])
	y = tf.placeholder(tf.float32, [None, 26])

	# Declares the session
	sess = tf.InteractiveSession()

	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1, 16, 8, 1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([4 * 2 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 2 * 64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 26])
	b_fc2 = bias_variable([26])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	y_hat = tf.nn.softmax(y_conv)

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))

	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess.run(tf.global_variables_initializer())

	for i in range(2600):
		batch_xs, batch_ys = next_batch(50, X_train, y_train)
		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
			print('step {}, training accuracy {}'.format(i, train_accuracy))
		train_step.run(feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
	
	# K-Fold cross validation
	accuracy_values = []
	kf = KFold(n_splits=5)
	for train_idx, val_idx in kf.split(X_train, y_train):
		train_x = X_train[train_idx]
		train_y = y_train[train_idx]
		val_x = X_train[val_idx]
		val_y = y_train[val_idx]

		batch_xs, batch_ys = next_batch(100, train_x, train_y)
		accuracy_values.append(sess.run(accuracy, feed_dict={x: val_x, y: val_y, keep_prob: 1.0}))
	print(accuracy_values)

	y_pred = sess.run(y_hat, feed_dict={x: X_test, keep_prob: 1.0})

	np.savetxt('test-labels.txt', one_hot_decode(y_pred), delimiter=',', fmt='%c')
