from init_parameters import *

# 将x，y按照batch[batch_vertical_size, truncated_backprop_length]的训练形式定义
batchX_placeholder = tf.placeholder(tf.float32, [batch_vertical_size, truncated_backprop_length], name='input_placeholder')
batchY_placeholder = tf.placeholder(tf.int32, [batch_vertical_size, truncated_backprop_length], name='labels_placeholder')


# 以相邻的时间步分割批数据[batch_vertical_size, truncated_backprop_length]
columnsinputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

#定义rnn_cell的权重参数
with tf.variable_scope('rnn_cell'):
    """由于tf.Variable() 每次都在创建新对象，所有reuse=True 和它并没有什么关系。
    对于get_variable()，来说，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，
    就创建一个新的。"""
    weights = tf.get_variable('weights', [state_size + 1, state_size], dtype=tf.float32)
    biases = tf.get_variable('biases',  [1, state_size], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

def rnn_cell(current_input, current_state):
    with tf.variable_scope('rnn_cell', reuse=True):
        weights = tf.get_variable('weights', [state_size + 1, state_size], dtype=tf.float32)
        biases = tf.get_variable('biases', [1, state_size], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    # 定义rnn_cell具体的操作，这里使用的是最简单的rnn，不是LSTM
    input_and_state_concatenated = tf.concat([current_input, current_state], 1) # Increasing number of columns

    next_state = tf.matmul(input_and_state_concatenated, weights) + biases

    return tf.tanh(next_state) # Broadcasted addition

# 前向传播
current_states  = init_states
states_ouputs = []

#滑动steps次，即将一个序列输入RNN模型
for current_input in columnsinputs_series:
    current_input = tf.reshape(current_input, [batch_vertical_size, 1])
    next_states = rnn_cell(current_input, current_states)
    states_ouputs.append(next_states)
    current_states = next_states

final_state = states_ouputs[-1]

#定义softmax层
with tf.variable_scope('softmax'):
    weights = tf.get_variable('weights', [state_size, num_classes], dtype=tf.float32)
    biases = tf.get_variable('biases', [1, num_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

#注意，这里要将num_steps个输出全部分别进行计算其输出，然后使用softmax预测
logits_series = [tf.matmul(state, weights) + biases for state in states_ouputs]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]


#losses and train_step
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits, labels=labels) for \
          logits, labels in zip(logits_series, labels_series)]

total_loss = tf.reduce_mean(losses)

#TensorFlow将自动执行反向传播函数：对每批数据执行一次计算图，并逐步更新网络权重。
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)