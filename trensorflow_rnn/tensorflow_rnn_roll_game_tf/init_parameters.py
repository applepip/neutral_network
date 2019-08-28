
import tensorflow as tf

# 在三个位置同时开启训练，所以在前向传播时需要保存三个状态。
# 我们在参数定义时就已经考虑到这一点了，故将init_state设置为3
batch_vertical_size = 5

state_size = 15

init_states = tf.placeholder(tf.float32, [batch_vertical_size, state_size])

# RNN这种网络能记忆输入数据信息，在若干时间步后将其回传，现在设截断反传长度为3。
truncated_backprop_length = 15

# 分类结果种类
num_classes = 2

learning_rate = 0.2

# 训练递归次数
num_epochs = 3

# 回滚次数
roll_steps = 3

# 生成数据的数据量
total_series_length = 50000

# 将数据分成batch[batch_vertical_size,truncated_backprop_length]的数量
num_batches = total_series_length//batch_vertical_size//truncated_backprop_length