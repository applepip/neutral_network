
import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #不指定GPU,强制使用CPU进行计算

config = tf.ConfigProto(device_count={'gpu': 1})  #强制使用CPU计算，否则导致显卡占用过高，让显卡温度过高，自动熄屏
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 程序最多只能占用指定gpu90%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)
import keras.backend.tensorflow_backend as KTF
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 1})))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LERAN_RATE = 0.1
REAL_PARAMS  = [1.2, 2.5]
INIT_PARAMS = [[5, 4],
               [5, 1],
               [2, 4.5]][1]  # 取2,4.5 这组数据

x = np.linspace(-1, 1, 200, dtype=np.float32)

y_fun = lambda a, b: np.sin(b * np.cos(a*x))
# 真实模型的tensorflow版预测模型
tf_y_fun = lambda a, b: tf.sin(b * tf.cos(a * x))

noise = np.random.randn(200) / 10
y = y_fun(*REAL_PARAMS) + noise

a, b = [tf.Variable(initial_value=p, dtype=tf.float32) for p in INIT_PARAMS]

pred = tf_y_fun(a, b)
mse = tf.reduce_mean(tf.square(y - pred))
train_op = tf.train.GradientDescentOptimizer(LERAN_RATE).minimize(mse)

a_list, b_list, cost_list = [], [], []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(10000):
        a_, b_, mse_ = sess.run([a, b, mse])
        a_list.append(a_);
        b_list.append(b_);
        cost_list.append(mse_)  # record parameter changes
        result, _ = sess.run([pred, train_op])  # training

# visualization codes:
print('a=', a_, 'b=', b_)
plt.figure(1)
plt.scatter(x, y, c='b')  # plot data
plt.xlabel('x')
#设置Y轴标签
plt.ylabel('y')
# 3D cost figure
fig = plt.figure(2);
ax = Axes3D(fig)
a3D, b3D = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))  # parameter space
cost3D = np.array([np.mean(np.square(y_fun(a_, b_) - y)) for a_, b_ in zip(a3D.flatten(), b3D.flatten())]).reshape(
    a3D.shape)
ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.5)
ax.scatter(a_list[0], b_list[0], zs=cost_list[0], s=300, c='r')  # initial parameter place
ax.set_xlabel('a');
ax.set_ylabel('b')
ax.set_zlabel('cost')
ax.plot(a_list, b_list, zs=cost_list, zdir='z', c='r', lw=3)  # plot 3D gradient descent
plt.show()