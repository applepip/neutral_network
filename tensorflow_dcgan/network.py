# -*- coding:UTF-8 -*-

"""
DCGAN 深层卷积的生成对抗网络
"""
import tensorflow as tf


# Hyperparameters 超参数
EPOCHES = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
BETA_1 = 0.5

# 定义判别器模型

def discriminator_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        64,  #64个filter，输出的深度是64
        (5, 5), #过滤器在二维的大小是（5 * 5）
        padding = 'same', #same表示输出的feature map的大小不变，因此需要在外围补零2圈
        input_shape=(64, 64, 3) #输入形状[64,64,3]。3表示RGB三原色
    ))
    model.add(tf.keras.layers.Activation("tanh")) #添加Tanh激活层
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2))) #池化层

    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation("tanh"))  # 添加Tanh激活层
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))  # 池化层

    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation("tanh"))  # 添加Tanh激活层
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))  # 池化层

    model.add(tf.keras.layers.Flatten()) #扁平化操作
    model.add(tf.keras.layers.Dense(1024)) #1024个神经元的全连接层
    model.add(tf.keras.layers.Activation("tanh"))  # 添加Tanh激活层

    model.add(tf.keras.layers.Dense(1))  # 1个神经元的全连接层
    model.add(tf.keras.layers.Activation("sigmoid"))  # 添加Sigmoid激活层

    return model


#定义生成器
#从随机数生成图片

def generator_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(input_dim=100, units=1024)) #输入维度是100，输出维度（神经元个数）是1024的全连接
    model.add(tf.keras.layers.Activation("tanh"))  # 添加Tanh激活层

    model.add(tf.keras.layers.Dense(128*8*8))  # 8192个神经元的全连接层
    model.add(tf.keras.layers.BatchNormalization())  #批量标准化
    model.add(tf.keras.layers.Activation("tanh"))  # 添加Tanh激活层
    model.add(tf.keras.layers.Reshape((8, 8, 128), input_shape=(128*8*8, ))) #该层将图像变成8*8的图片
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2))) #上采样后图像变成16*16像素
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))  # 添加Tanh激活层
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))  # 上采样后图像变成64*64像素
    model.add(tf.keras.layers.Conv2D(3, (5, 5), padding="same")) #3为RGB的3个通道
    model.add(tf.keras.layers.Activation("tanh"))  # 添加Tanh激活层

    return model

# 构造一个Sequential对象，包含一个生成器和一个判别器
# 输入-> 生成器 -> 判别器 -> 输出

def generator_containing_discriminator(generator, discriminator):
    model = tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable = False #初始时判别器是不可被训练的
    model.add(discriminator)


