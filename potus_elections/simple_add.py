

__author__ = "Sreejith Sreekumar"
__email__ = "sreekumar.s@husky.neu.edu"
__version__ = "0.0.1"


x_tensor = tf.placeholder(tf.float64)
y_tensor = tf.placeholder(tf.float64)
add_out = tf.add(x_tensor, y_tensor)

with tf.Session() as sess:
    
    test_x = X_train[0][0]
    test_y = X_train[0][1]

    print(test_x)
    print(test_y)
    
    print(sess.run(add_out, feed_dict={x_tensor: (test_x), y_tensor:  (test_y)}))
