from frcnn_lab import *

'''
rpn(region proposal networks)
'''

def rpn_layer(base_layers, num_anchors):
    '''
    通过设计卷积神经网络来提取候选区域（RPN）
        Step1: Pass through the feature map from base layer to a 3x3 512 channels convolutional layer
                        Keep the padding 'same' to preserve the feature map's size
        Step2: Pass the step1 to two (1,1) convolutional layer to replace the fully connected layer
                    classification layer: num_anchors (9 in here) channels for 0, 1 sigmoid activation output
                    regression layer: num_anchors*4 (36 in here) channels for computing the regression of bboxes with linear activation

    :param base_layers: 输入为vgg16模型
    :param num_anchors: 这里设使用9个anchor
    :return:
    '''

    x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(
        base_layers)

    x_class = tf.keras.layers.Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = tf.keras.layers.Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]