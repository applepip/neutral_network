import tensorflow as tf
import numpy as np

input_x1 = np.linspace(-1, 1, 100)
input_x2 = np.linspace(-1, 1, 100)
input_y = (input_x1**2 + input_x2 - 11)**2 + (input_x2 + input_x2**2 - 7)**2 + np.random.randn(input_x1.shape[0])*0.5

x = tf.placeholder(tf.float32, [1, 2])
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.zeros([2, 1]), name='weight')
b = tf.Variable(0.0, name='bias')

#model
o = tf.sigmoid(tf.matmul(x, w) + b)

loss = tf.square(o[0][0] - y)/2

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for (X1, X2, Y) in zip(input_x1, input_x2, input_y):
        _train_op, w_value, b_value = sess.run([train_op, w, b], feed_dict={x: [[X1, X2]], y: Y})
        print("w: {}, b: {}".format(w_value, b_value))