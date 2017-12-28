import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"

data_folder = "data/"
train_file = data_folder + "train.csv"


test_file = data_folder + "test.csv"

data = pd.read_csv(train_file)

msk = np.random.rand(len(data)) < 0.8

train = data[msk]
test = data[~msk]

one_hot_of_winner = pd.get_dummies(train['Winner'])
one_hot_of_winner_test = pd.get_dummies(test['Winner'])

train.drop(['Winner'], axis=1, inplace=True)
test.drop(['Winner'], axis=1, inplace=True)
X_train = train.as_matrix()
Y_train = one_hot_of_winner.as_matrix()

X_test = test.as_matrix()
Y_test = one_hot_of_winner_test.as_matrix()


learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1

n_hidden_1 = 20 # 1st layer number of features
n_hidden_2 = 20 # 2nd layer number of features
n_input = 14 # Number of feature
n_classes = 2 # Number of classes to predict


# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
