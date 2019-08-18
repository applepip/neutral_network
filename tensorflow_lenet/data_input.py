import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

'''
获取mnist数据
'''
data = read_data_sets('data/MNIST/', one_hot=True)
print("样本大小:")
print("- 训练集大小:\t\t{}".format(len(data.train.labels)))
print("- 测试集大小:\t\t{}".format(len(data.test.labels)))
print("- 验证集大小:\t{}".format(len(data.validation.labels)))

# argmax会根据axis取值的不同返回每行或者每列最大值的索引, labels是one-hot编码，取出后就是完整的数字标签了
data.test.cls = np.argmax(data.test.labels, axis=1)


