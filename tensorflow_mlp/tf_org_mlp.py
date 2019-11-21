'''
使用普通 TensorFlow 训练 DNN
'''

import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# 直接可以使用tensorflow的fully_connected或者tf.layers.dense()代替
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        # 保存权重矩阵的 W变量，它将是包含每个输入和每个神经元之间
        # 的所有连接权重的2D张量；因此，它的形状将是(n_inputs, n_neurons)

        # tf.truncated_normal函数从截断的正态分布中输出随机值初
        # 始化权重，即使用具有标准差为 2/√n 的截断的正态（高斯）
        # 分布初始化权重
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        # 初始化权重和bias
        W = tf.Variable(init, name="kernel")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")

        # 神经网络
        Z = tf.matmul(X, W) + b

        if activation is not None:
            return activation(Z)
        else:
            return Z


with tf.name_scope("dnn"):
    hidden_1 = neuron_layer(X, n_hidden1, name="hidden1",
                            activation=tf.nn.relu)
    hidden_2 = neuron_layer(hidden_1, n_hidden2, name="hidden2",
                            activation=tf.nn.relu)

    # logit是在通过softmax激活函数之前神经网络的输出
    logits = neuron_layer(hidden_1, n_outputs, name="outputs")

with tf.name_scope("loss"):

    # 计算logits和labels之间的稀疏softmax交叉熵，即根据“logit”变量计算交叉熵
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)

    # tf.reduce_mean数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# 评估模型
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 40
batch_size = 50

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

with tf.Session() as sess:
    init.run()

    wirter = tf.summary.FileWriter('logs/', sess.graph)

    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

    save_path = saver.save(sess, "tmp/my_model_final.ckpt")