# encoding:utf-8
# 解决中文编码的问题

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 只显示 warning 和 Error

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

class LeNet5():

    data = []

    #  ---卷积核设置---begin
    # 第一个卷积核的大小 5*5
    conv1_filter_size = 5
    # 第一个卷积的卷积核数目
    num_conv1_filter = 16

    # 第二个卷积核的大小 5*5
    conv2_filter_size = 5
    # 第一个卷积的卷积核数目
    num_conv2_filter = 36

    # 全连接层的神经元数目
    num_fc = 128
    #  ---卷积核设置---end

    # ---输入图片设置---begin
    # 图片长和宽为28个像素点
    img_size = 28
    # 图片扁平化处理,大小为28*28=784
    img_size_flat = img_size * img_size
    # 图片大小
    img_shape = (img_size, img_size)
    # ---输入图片设置---end

    def __init__(self):
        self.get_data()

    def get_data(self):
        '''
        获取mnist数据
        '''

        self.data = read_data_sets('data/MNIST/', one_hot=True)
        print("样本大小:")
        print("- 训练集大小:\t\t{}".format(len(self.data.train.labels)))
        print("- 测试集大小:\t\t{}".format(len(self.data.test.labels)))
        print("- 验证集大小:\t{}".format(len(self.data.validation.labels)))
        self.data.test.cls = np.argmax(self.data.test.labels, axis=1)

    def plot_imaege(self, images, cls_true, cls_pred = None):
        '''
        显示图片
        :param images: 需要显示的图片
        :param cls_true:
        :param cls_pred:
        :return:
        '''
        # 显示3*3个图片
        fig, axes = plt.subplot(3, 3)  #  fig为总画布，有figure就可以在上边
                                       # （或其中一个子网格/subplot上）作图
                                       #  axes为子区域（子网格）对象
        fig.subplots_adjust(hspace=0.3, wspace=0.3) # 调整子图边距和子图的间距，
                                                    # hspace：子图之间的空间的高度
                                                    # wspace：子图之间的空间的宽度

        for i, ax in enumerate(axes.flat):
            # Plot image.
            ax.imshow(images[i].reshape(self.img_shape), cmap='binary')

            # 显示真实值(True)和预测值(Pred)
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

            # 设置上面定义好的x轴显示内容
            ax.set_xlabel(xlabel)

            # 重新定义x轴和y轴的刻度
            # 先移除当前的x轴和y轴的刻度
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()



    def plot_9_test_images(self):
        '''
        显示测试集中的前9张图片和对应的真实标签
        :return:
        '''
        images = self.data.test.images[0:9]

        cls_true = self.data.test.cls[0:9]

        self.plot_imaege(images=images, cls_true=cls_true)


    def new_weight(self, shape):
        '''
        生成一个权重张量，权重张量维度为shape，该权重服从正态分布均值为mean，标准差为stddev
        :param shape: 权重张量维度
        :return:
        '''
        return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

    def new_biases(self, length):
        '''
        偏置生成函数,因为激活函数使用的是ReLU
        我们给偏置增加一些小的正值(0.05)避免死亡节点(dead neurons)
        :param length:
        :return:
        '''
        return tf.Variable(tf.constant(0.05, shape=[length]))



lenet5 = LeNet5()
