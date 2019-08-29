import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 只显示 warning 和 Error

import time
from datetime import timedelta

from model import *
from data_input import *

from tools import *
from plot_conv_weights import *
from plot_conv_layer import *

# tensorboard 位置
path = "log/"

saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    train_batch_size = 64

    total_iterations = 0

    writer = tf.summary.FileWriter(path, session.graph)


    def optimize(num_iterations):
        global total_iterations
        start_time = time.time()

        for i in range(total_iterations,
                       total_iterations + num_iterations):
            x_batch, y_true_batch = data.train.next_batch(train_batch_size)
            # x_batch的维度是（64,28*28*1）即一次训练64张图片
            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            session.run(optimizer, feed_dict=feed_dict_train)

            if i % 100 == 0:
                acc = session.run(accuracy, feed_dict=feed_dict_train)
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
                print(msg.format(i + 1, acc))

        total_iterations += num_iterations

        end_time = time.time()
        time_dif = end_time - start_time
        print("The ", total_iterations, " times optimize and the time usage is: " + str(timedelta(seconds=int(round(time_dif)))))


    def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
        '''
        使用测试集测试模型精度
        :param show_example_errors:
        :param show_confusion_matrix:
        :return:
        '''

        #每次选取256组测试图片进行测试
        test_batch_size = 256

        num_img_test = len(data.test.images)   #测试集大小
        cls_pred = np.zeros(shape=num_img_test, dtype=np.int)   #初始化预测值

        i = 0
        while i < num_img_test:
            j = min(i+test_batch_size, num_img_test)
            images = data.test.images[i:j, :]
            labels = data.test.labels[i:j, :]
            feed_dict = {x: images,
                         y_true: labels}
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
            i = j

        cls_true = data.test.cls   #测试集的真实值标签
        correct = (cls_true == cls_pred)
        correct_sum = correct.sum()

        acc = float(correct_sum) / num_img_test
        msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
        print(msg.format(acc, correct_sum, num_img_test))
        if show_example_errors:
            print("Example errors:")
            plot_example_errors(cls_pred=cls_pred, correct=correct)
        if show_confusion_matrix:
            print("Confusion Matrix:")
            plot_confusion_matrix(cls_pred=cls_pred)


    # 1次迭代后的性能
    optimize(num_iterations=1)
    print_test_accuracy()

    #100次迭代后的性能
    optimize(num_iterations=99)
    print_test_accuracy()

    #1000次迭代后的性能
    optimize(num_iterations=900)
    print_test_accuracy()

    print_test_accuracy(show_example_errors=True)

    #10000次迭代后的性能
    optimize(num_iterations=9000) # We performed 1000 iterations above.
    print_test_accuracy(show_example_errors=True,
                        show_confusion_matrix=True)

    saver.save(session, 'checkpoint_dir_10000/my_model')

    image = data.test.images[0]
    plot_image(image)

    # 输出第一层卷积核的内容
    plot_conv_weights(session, weights=weights_conv1)

    # 输出第一层卷后的特征图
    plot_conv_layer(session, x, layer=layer_conv1, image=image)

    # 输出第二层卷积核的内容
    plot_conv_weights(session, weights=weights_conv2)

    # 输出第二层卷后的特征图
    plot_conv_layer(session, x, layer=layer_conv2, image=image)

    session.close()
