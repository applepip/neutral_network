import numpy as np
from plot_data_distribution import *

def gen_echo_data(datasize, batch_vertical_size, roll_steps = 3):
    '''
	生成二进制数据流x，x沿x轴的正方向滚动roll_steps位形成y
	'''
    x = np.array(np.random.choice(2, datasize, p=[0.5, 0.5])) #p=[0.5, 0.5] 取0,1的概率各为50%
    y = np.roll(x, roll_steps)
	y[0:echo_step] = 0  #roll后，roll_steps前的数据设为0

    # 显示25个数据样本
    # plot_data_distribution(x[0:25], y[0:25])

    x = x.reshape((batch_vertical_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_vertical_size, -1))

    return (x, y)
