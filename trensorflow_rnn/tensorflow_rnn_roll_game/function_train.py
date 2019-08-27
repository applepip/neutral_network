from data import *
from model_rnn import *

from plot_training_batches_results import *

def train_network(total_loss, train_step, current_states, predictions_series):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_losses = []

        for epoch_idx in range(num_epochs):
            x, y = gen_echo_data(total_series_length, batch_vertical_size, roll_steps)

            _current_state = np.zeros((batch_vertical_size, state_size))

            print("New data, epoch", epoch_idx)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * truncated_backprop_length
                end_idx = start_idx + truncated_backprop_length

                batchX = x[:, start_idx:end_idx]
                batchY = y[:, start_idx:end_idx]

                _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                    [total_loss, train_step, current_states, predictions_series],
                    feed_dict={
                        batchX_placeholder: batchX,
                        batchY_placeholder: batchY,
                        init_states: _current_state
                    })

                training_losses.append(_total_loss)

                if batch_idx % 100 == 0:
                    print("Step", batch_idx, "Loss", _total_loss)
                    plot_training_batches_results(training_losses, _predictions_series, batchX, batchY, epoch_idx, batch_idx)
