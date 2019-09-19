from init_parameters import *

# 模型定义

# 将x，y按照batch[batch_vertical_size, truncated_backprop_length]的训练形式定义
batchX_placeholder = tf.placeholder(tf.float32, [batch_vertical_size, truncated_backprop_length], name='input_placeholder')
batchY_placeholder = tf.placeholder(tf.int32, [batch_vertical_size, truncated_backprop_length], name='labels_placeholder')

# cell当前状态
cell_state = tf.placeholder(tf.float32, [batch_vertical_size, state_size])
# cell当前状态下的输出信息
hidden_state = tf.placeholder(tf.float32, [batch_vertical_size, state_size])
init_states = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)


# 以相邻的时间步分割批数据[batch_vertical_size, truncated_backprop_length]
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

def rnn_cell():
    '''
    前向传播
    :return:states_ouputs[输出层],current_states[cell状态，包含lstm的cell_state, hidden_state]
    '''
    cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
    states_ouputs, current_states = tf.nn.static_rnn(cell, inputs_series, initial_state=init_states)

    return states_ouputs, current_states


def rnn_softmax(states_ouputs):
    '''
    定义softmax层
    :param states_ouputs: states_ouputs[输出层]
    :return:total_loss[损失量], train_step[梯度下降优化过程], predictions_series[预测值]
    '''
    with tf.variable_scope('softmax'):
        weights = tf.get_variable('weights', [state_size, num_classes], dtype=tf.float32)
        biases = tf.get_variable('biases', [1, num_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    # 注意，这里要将num_steps个输出全部分别进行计算其输出，然后使用softmax预测
    logits_series = [tf.matmul(state, weights) + biases for state in states_ouputs]
    predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

    # losses and train_step
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for \
              logits, labels in zip(logits_series, labels_series)]

    total_loss = tf.reduce_mean(losses)

    # TensorFlow将自动执行反向传播函数：对每批数据执行一次计算图，并逐步更新网络权重。
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

    return total_loss, train_step, predictions_series


