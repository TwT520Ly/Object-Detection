"""
    @Author: Tan.wt
    @Time: 2018/05/25
    @File: darknet53.py
    @License: Apache License
"""

from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from keras.layers import add, Activation, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2

def conv2d_unit(x, filter_num, kernel_size, strides=1):
    '''
    使用LeakyRelu和BatchNormalization构建卷积单元

    :param x: input tensor for conv layer
    :param filter_num: the dimensionality of the output space
    :param kernel_size: size of conv kernel, (x, y) -> (width, height)
    :param strides: specifying the strides of the convlution alon the width and height
    :return: tensor
    '''
    x = Conv2D(filters=filter_num,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               activation='linear',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def residual_block(inputs, filter_num):
    '''
    创建残差模块

    :param inputs:
    :param filter_num:
    :return: tensor
    '''
    x = conv2d_unit(inputs, filter_num, (1, 1))
    x = conv2d_unit(inputs, 2*filter_num, (3, 3))
    x = add([inputs, x])
    x = Activation('linear')(x)

    return x

def stack_residual_block(input, filter_num, n):
    if n >= 1:
        x = residual_block(input, filter_num)
        for i in range(n-1):
            x = residual_block(x, filter_num)
    else:
        x = input

    return x


def darknet_base(inputs):
    '''
    创建Darknet-53基本结构

    :param inputs: Image data
    :return: Feature map
    '''
    x = conv2d_unit(inputs, 32, (3, 3))
    x = conv2d_unit(x, 64, (3, 3), strides=2)
    x = stack_residual_block(x, 32, n=1)

    x = conv2d_unit(x, 64, (3, 3), strides=2)
    x = stack_residual_block(x, 64, n=2)

    x = conv2d_unit(x, 128, (3, 3), strides=2)
    x = stack_residual_block(x, 128, n=8)

    x = conv2d_unit(x, 512, (3, 3), strides=2)
    x = stack_residual_block(x, 256, n=8)

    x = conv2d_unit(x, 1024, (3, 3), strides=2)
    x = stack_residual_block(x, 512, n=4)

    return x

def darknet() -> Model:
    '''
    创建Darknet-53分类器，分类结果为1000类

    :return:  keras.models.Model
    '''
    inputs = Input(shape=(416, 416, 3))
    x = darknet_base(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='softmax')(x)

    model = Model(inputs, x)

    return model

if __name__ == '__main__':
    model = darknet()
    print(model.summary())