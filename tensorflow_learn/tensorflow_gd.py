import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#构建数据

#构建100个数据点

points_num = 100
vectors = []

#用Numpy的正太分布函数生成100个点
#这些点的（x，y）坐标值对应线性方程 y = 0.1*x + 0.2
#权重（Weight）0。1 偏差（Bias）0。2
for i in range(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 *x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1,y1])

x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

#展示所有的随机点

plt.plot(x_data, y_data, 'r*',label="Original data") #红色*点为原始数据
plt.title("Linear Regression using Gradient Descent")
plt.legend() #show lable
plt.show()

#构建线性回归模型

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = W * x_data + b

#定义loss function或cost function
#对Tensor的所有纬度计算sum(（（y-y_data）^2）)/N

loss=tf.reduce_mean(tf.square(y-y_data))

#使用梯度下降优化器优化loss

optimizer = tf.train.GradientDescentOptimizer(0.5) # 设置学习率
train = optimizer.minimize(loss)

#create tf session

sess = tf.Session()

#初始化变量

init = tf.global_variables_initializer()
sess.run(init)

#训练20步

for step in range(20):
    sess.run(train)
    print("step=%d, loss=%f,[Weight=%f Bias=%f]" \
          % (step,sess.run(loss),sess.run(W),sess.run(b)))

plt.plot(x_data, y_data, 'r*',label="Original data") #红色*点为原始数据
plt.title("Linear Regression using Gradient Descent")
plt.plot(x_data, sess.run(W) * x_data + sess.run(b),label="Fitted Line")
plt.legend() #show lable
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#关闭会话，否则会有资源泄露
sess.close()