# 图解图像内容
import matplotlib.pyplot as plt
import numpy as np

def plot_img_content(data):
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 30)
    width, height = data.shape

    imshow_data = np.random.rand(width, height)
    ax.imshow(imshow_data, cmap=plt.cm.Pastel1, interpolation='nearest')

    for x in range(0, height):
        for y in range(0, width):
            if (data[y][x] > 0):
                ax.text(x, y, round(data[y][x],2), va='center', ha='center', fontsize=10)

    plt.show()