import numpy as np
from plot_data_distribution import *

def gen_batch(datasize, batch_vertical_size):
    '''
    产生数据训练的batch
    :param raw_data: gen_data()函数生成的数据
    :param batch_vertical_size:
    :return:
    '''
    raw_x, raw_y = gen_data(datasize)
    data_length = len(raw_x)

    x = raw_x.reshape((batch_vertical_size, -1))  # The first index changing slowest, subseries as rows
    y = raw_y.reshape((batch_vertical_size, -1))

    return (x, y)


def gen_data(size=20):
    '''
    生成数据
    输入数据X：在时间t，Xt的值有50%的概率为1，50%的概率为0；
    输出数据Y：在实践t，Yt的值有50%的概率为1，50%的概率为0，除此之外，如果`Xt-3 == 1`，Yt为1的概率增加50%，
     如果`Xt-8 == 1`，则Yt为1的概率减少25%， 如果上述两个条件同时满足，则Yt为1的概率为75%。
    :param size: 数据量
    :return:
    '''

    x = np.array(np.random.choice(2, size=(size,)))  #自动生成一个[0,2）的一维数组形式, 长度为size
    print(x)
    y = []

    for i in range(size):
        threshold = 0.5
        if x[i-3] == 1:
            threshold += 0.5
        if x[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:  #通过np.random.rand()函数可以返回一个或一组服从“0~1”均匀分布的随机样本值
            y.append(0)  # 随机性大于为1的概率，则y=0
        else:
            y.append(1)  # 随机性小于为1的概率，则y=1

    y = np.array(y)

    # 显示25个数据样本
    plot_data_distribution(x[0:25], y[0:25])

    return x,y
