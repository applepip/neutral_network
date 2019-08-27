import matplotlib.pyplot as plt

def plot_data_distribution(x, y):
    '''
    画出数据的概率分布图
    :return:
    '''
    xt = [i for i in range(len(x))]

    # 设定x的点显示大小
    xs = 30 ** 1.5
    plt.scatter(xt, x, c='g', s=xs, marker='o', label="x")
    plt.xlabel('x time series')
    plt.scatter(xt, y, c='r', marker='^', label="y")
    plt.ylabel('value of y')
    plt.legend(loc='best')
    plt.title('Binary data(size of 20)')
    plt.show()
