import tensorflow as tf

from data_input import *
from model import *
from tools import *

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('checkpoint_dir')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise FileNotFoundError("未保存任何模型")

    imgs = data.test.images[0:9]
    y_correct = data.test.cls[0:9]

    flatten_imgs = [np.reshape(img, img_size_flat) for img in imgs]

    predictions = [sess.run(y_pred_cls, feed_dict={x: [flatten_img]}) for flatten_img in flatten_imgs]

    print(predictions)

    plot_result_prediction(imgs, y_correct, predictions)