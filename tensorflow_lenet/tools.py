
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from neutral_network.tensorflow_lenet.input_img_parameter import *

def plot_example_errors(data, cls_pred, correct):
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_confusion_matrix(data, cls_pred):
    cls_true = data.test.cls
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    tick_marks = np.arange(num_img_classes)
    plt.xticks(tick_marks, range(num_img_classes))
    plt.yticks(tick_marks, range(num_img_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_9_test_images(data):
    '''
    显示测试集中的前9张图片和对应的真实标签
    :return:
    '''
    images = data.test.images[0:9]

    cls_true = data.test.cls[0:9]

    plot_images(images=images, cls_true=cls_true)


def plot_images(images, img_shape, cls_true, cls_pred=None):
    '''
    显示图片
    :param images: 需要显示的图片
    :param cls_true:
    :param cls_pred:
    :return:
    '''
    # 显示3*3个图片
    fig, axes = plt.subplot(3, 3)  # fig为总画布，有figure就可以在上边
    # （或其中一个子网格/subplot上）作图
    #  axes为子区域（子网格）对象
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # 调整子图边距和子图的间距，
    # hspace：子图之间的空间的高度
    # wspace：子图之间的空间的宽度

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

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
