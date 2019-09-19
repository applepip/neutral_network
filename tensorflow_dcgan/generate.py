# -*- coding:UTF8 -*-

"""
用DCGAN的生成器模型和训练得到的生成器参数文件来生成图片
"""

import numpy as np
from PIL import Image
import tensorflow as tf

from network import *

def generate():
    # 构造生成器
    g = generator_model()

    # 配置生成器
    g.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr = LEARNING_RATE, beta_1=BETA_1))

    #加载训练好的生成器参数
    g.load_weights("generator_weight")

    #连续型均匀分布的的随机数据（噪声）
    random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

    #用随机数据作为输入，用生成器生成图片
    images = g.predict(random_data, verbose=1)

    #用生成的图片数据生成PNG图片
    for i in range(BATCH_SIZE):
        image = images[i] * 127.5 +127.5
        Image.fromarray(image.astype(np.uint8)).save("image-%s.png" % i)