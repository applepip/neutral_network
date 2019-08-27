from function_train import *

# 建立模型
states_ouputs, current_states = rnn_cell()
total_loss, train_step, predictions_series = rnn_softmax(states_ouputs)

# 运行训练
train_network(total_loss, train_step, current_states, predictions_series)