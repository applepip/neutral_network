import matplotlib.pyplot as plt
import numpy as np

from init_parameters import *

def plot_training_batches_results(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(batch_vertical_size):
        print(predictions_series)
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx]
        print(one_hot_output_series)
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
        print(single_output_series)

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()  # Clear axis即清除当前图形中的当前活动轴。其他轴不受影响。

        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)

        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)