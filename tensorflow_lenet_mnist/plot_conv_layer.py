import numpy as np
import math
import matplotlib.pyplot as plt

# 卷积分析
# feature maps 的形状
# shape = [num_img, con_img_size, con_img_size, num_feature_maps]

def plot_conv_layer(session, x, layer, image):
    feed_dict = {x: [image]}
    layer = session.run(layer, feed_dict=feed_dict)

    # 获取当前filter个数
    num_feature_maps = layer.shape[3]

    # 画图的格子数量
    num_grids_max = math.ceil(math.sqrt(num_feature_maps))
    num_grids_min = math.floor(math.sqrt(num_feature_maps))

    fig, axes = plt.subplots(num_grids_max, num_grids_min)

    for i, ax in enumerate(axes.flat):
        if i < num_feature_maps:
            img = layer[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


