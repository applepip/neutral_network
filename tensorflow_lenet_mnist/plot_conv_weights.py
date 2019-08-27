import numpy as np
import math
import matplotlib.pyplot as plt

# 卷积分析
# weights 的形状
# shape = [filter_size, filter_size, num_input_channels, num_filters]
def plot_conv_weights(session, weights, input_channel=0):

    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)

    # 获取当前filter个数
    num_filters = w.shape[3]

    # 画图的格子数量
    num_grids_max = math.ceil(math.sqrt(num_filters))
    num_grids_min = math.floor(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids_max, num_grids_min)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:, :, input_channel, i]
            map = ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
            cbar = plt.colorbar(mappable=map, ax=ax)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()







