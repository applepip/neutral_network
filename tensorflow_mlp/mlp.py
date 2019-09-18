import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

learn_rate = 0.1
real_params = [1.2, 2.5]
init_params = [[5, 4],
               [5, 1],
               [2, 4.5]][2]  # 取2,4.5 这组数据

x = np.linspace(-1, 1, 200, dtype=np.float32)

y_fun = lambda a, b: np.sin(b*np.cos(a*x))
tf_y_fun = lambda a, b: tf.sin(b * tf.cos(a * x))

noise = np.random.randn(200) / 10
y = y_fun(*real_params) + noise