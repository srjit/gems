import numpy as np
import tensorflow as tf


n_input = 14
n_classes = 2
epochs = 100


n_hidden_1 = 20
n_hidden_2 = 20

train_X = np.random.random((100, 14))
train_Y =  np.random.choice([0, 1], size=(100,2), p=[1./3, 2./3])
learning_rate = 0.01



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


output = multilayer_perceptron(x, weights, biases)

# get the index with highest argument (index that maximizes the output)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_count = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_count, "float"))


## has to be called after defining all tensors
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(epochs):
        acc, _, c = sess.run([accuracy, optimizer, cost], feed_dict={x:train_X, y:train_Y})
        print(acc)
