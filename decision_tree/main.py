from xigua.DTree import *
import csv
import numpy as np
import xigua.Tree_plot as Tree_plot

def load_data(filename):
    ''' 加载文本文件中的数据.
    '''
    csv_file = csv.reader(open(filename, 'r', encoding='utf-8'))
    dataset = []
    for line in csv_file:
        dataset.append(line)

    return dataset


data = load_data('data/xigua2.csv')
data = np.array(data)
lables = data[0,1:-1]
data_train = data[1:, 1:]
target = data[1:, -1]

lables = lables.tolist()
data_train = data_train.tolist()

dt = DTree()
tree = dt.create_division_tree(data_train, lables)


print(tree)

Tree_plot.createPlot(tree)
