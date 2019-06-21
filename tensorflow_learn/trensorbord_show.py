import tensorflow as tf

# y = W*x + b

w = tf.Variable(2.0, dtype=tf.float32, name="Weight")
b = tf.Variable(1.0, dtype=tf.float32, name="Bias")

x = tf.placeholder(dtype=tf.float32,name ="Input")

with tf.name_scope("Output"):
    y = w * x +b

path = "logts/"

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init) #初始化变量
    writer = tf.summary.FileWriter(path, sess.graph)
    result = sess.run(y, {x:3.0})
    print(result)
