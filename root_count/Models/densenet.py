from __future__ import absolute_import
from __future__ import print_function
from keras.layers import *
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import parameters as para
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.engine.topology import get_source_inputs


def dense_block(x, blocks, name):
    for i in range(blocks):
        name_str = name + '_block' + str(i + 1)
        x = conv_block(x, 32, name=name_str)
    return x


def transition_block(x, reduction, name):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)

    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)

    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks, input_shape=(para.img_rows, para.img_cols, para.channels), classes=para.count_classes):

    img_input = Input(shape=input_shape)
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='fc_layer')(x)

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(img_input, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(img_input, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(img_input, x, name='densenet201')
    else:
        model = Model(img_input, x, name='densenet')

    return model


if __name__ == '__main__':
    model = DenseNet([6, 12, 24, 16])
    model.summary()
    #model = DenseNet([6, 12, 32, 32])
    #model.summary()
    #model = DenseNet([6, 12, 48, 32])
    #model.summary()