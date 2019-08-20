
import tensorflow as tf

def new_weights(shape):
    '''
    生成一个权重张量，权重张量维度为shape，该权重服从正态分布均值为mean，标准差为stddev
    :param shape: 权重张量维度
    :return:
    '''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    '''
    偏置生成函数,因为激活函数使用的是ReLU
    我们给偏置增加一些小的正值(0.05)避免死亡节点(dead neurons)
    :param length:
    :return:
    '''
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(img_inputs,
                   num_img_channels,
                   filter_size,
                   num_filters,
                   use_pooling=True,
                   padding='SAME'):
    '''
    生成卷积层
    :param img_inputs: 上一层输出作为输入
    :param num_img_channels: 值为1,因为是黑白图片
    :param filter_size: 当前使用卷积的大小
    :param num_filters: 当前使用卷积的数量
    :param use_pooling: 使用2*2的max-pooling
    :return: 卷积层，该卷积层的权重
    '''
    shape = [filter_size, filter_size, num_img_channels, num_filters]

    weights = new_weights(shape)
    biases = new_biases(num_filters)

    print('con_weight', weights)
    print('con_biases', biases)

    layer = tf.nn.conv2d(input=img_inputs,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding=padding)

    print('con_layer', layer)

    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding=padding)

        print('pooling', layer)

    layer = tf.nn.relu(layer)

    return layer, weights


def flatten_layer(layer):
    '''
    将特征图进行展开
    :param layer: 卷积层返回的特征图
    :return: 展开的卷积层，该卷积层的特征总数量
    '''
    layer_shape = layer.get_shape()

    # layer_shape内容是 [num_images, img_height, img_width, num_channels]
    # 该层特征的总数量是 img_height * img_width * num_channels

    num_features = layer_shape[1:4].num_elements()

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    layer_flat = tf.reshape(layer, [-1, num_features])  # -1表示需要计算的纬度

    return layer_flat, num_features


def new_fc_layer(inputs,
                 num_inputs,
                 num_outputs,
                 use_relu=True):
    '''
    生成全连接层
    :param inputs: 上一层卷积层输出
    :param num_inputs: 上一层卷积层输出的个数
    :param num_outputs: 全连接层输出的个数
    :param use_relu: 是否用relu函数
    :return: 全连接层输出
    '''

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    print('fc_weight', weights)
    print('fc_biases', biases)

    layer = tf.matmul(inputs, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)
    return layer