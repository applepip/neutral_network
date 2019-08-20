from cnn import *
from input_img_parameter import *

#  ---卷积核设置---begin
# 第一个卷积核的大小 5*5
conv1_filter_size = 5
# 第一个卷积的卷积核数目
num_conv1_filter = 6

# 第二个卷积核的大小 5*5
conv2_filter_size = 5
# 第一个卷积的卷积核数目
num_conv2_filter = 16

# 全连接层的神经元数目
fc1_size = 120

fc2_size = 84
#  ---卷积核设置---end


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

# 图片输入层32*32，1个channel的黑白图片
x_image = tf.reshape(x, [-1, img_size, img_size, num_img_channels])  # 初始图片

print(x_image)

y_true = tf.placeholder(tf.float32, shape=[None, num_img_classes], name='y_true')

y_true_cls = tf.argmax(y_true, axis=1)

# 第一次卷积，输入5*5的filter，共6个
layer_conv1, weights_conv1 = \
    new_conv_layer(img_inputs=x_image,
                       num_img_channels=num_img_channels,
                       filter_size=conv1_filter_size,
                       num_filters=num_conv1_filter,
                       use_pooling=True)

print(layer_conv1)

# 第二次卷积，输入5*5的filter，共16个
layer_conv2, weights_conv2 = \
    new_conv_layer(img_inputs=layer_conv1,
                       num_img_channels=num_conv1_filter,
                       filter_size=conv2_filter_size,
                       num_filters=num_conv2_filter,
                       use_pooling=True)

print(layer_conv2)

# 特征展开
layer_flat, num_features = flatten_layer(layer_conv2)

print(layer_flat,num_features)

layer_fc1 = new_fc_layer(inputs=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc1_size,
                             use_relu=True)

print(layer_fc1)

layer_fc2 = new_fc_layer(inputs=layer_fc1,
                             num_inputs=fc1_size,
                             num_outputs=fc2_size,
                             use_relu=False)

print(layer_fc2)

layer_outputs = new_fc_layer(inputs=layer_fc2,
                             num_inputs=fc2_size,
                             num_outputs=num_img_classes,
                             use_relu=False)

print(layer_outputs)



# 用softmax计算预测值
y_pred = tf.nn.softmax(layer_outputs)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_outputs,
                                                           labels=y_true)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


