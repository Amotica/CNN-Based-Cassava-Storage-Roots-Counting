from keras.models import Model
import parameters as para
from keras.layers import *
import tensorflow as tf


def bilinear_upsampling(stride, **kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (stride * input_shape[1], stride * input_shape[2])
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)


def bicubic_upsampling(stride, **kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (stride * input_shape[1], stride * input_shape[2])
        return tf.image.resize_bicubic(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)


def nearest_upsampling(stride, **kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (stride * input_shape[1], stride * input_shape[2])
        return tf.image.resize_nearest_neighbor(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)


def unet_lite(input_shape=(para.img_rows, para.img_cols, para.channels), alpha=1.0, gf=64):
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = SeparableConv2D(int(filters*alpha), kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        #u = UpSampling2D(size=2)(layer_input)
        u = nearest_upsampling(stride=2)(layer_input)
        u = SeparableConv2D(int(filters*alpha), kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=input_shape)

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, gf * 4)
    u2 = deconv2d(u1, d2, gf * 2)
    u3 = deconv2d(u2, d1, gf)

    u4 = UpSampling2D(size=2)(u3)

    model = Model(d0, u4)
    rows = model.output_shape[1]
    cols = model.output_shape[2]

    u5 = Conv2D(para.seg_classes, (1, 1), padding="valid")(u4)
    u5 = Reshape((input_shape[0] * input_shape[1], para.seg_classes))(u5)  # *****************
    seg_output = Activation("softmax", name='segmentModel')(u5)

    model = Model(d0, seg_output)

    return model, rows, cols



def unet_lite_aug(input_shape=(para.img_rows, para.img_cols, para.channels), alpha=1.0, gf=64):
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = SeparableConv2D(int(filters*alpha), kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        #u = UpSampling2D(size=2)(layer_input)
        u = nearest_upsampling(stride=2)(layer_input)
        u = SeparableConv2D(int(filters*alpha), kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=input_shape)

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, gf * 4)
    u2 = deconv2d(u1, d2, gf * 2)
    u3 = deconv2d(u2, d1, gf)

    u4 = UpSampling2D(size=2)(u3)

    model = Model(d0, u4)
    rows = model.output_shape[1]
    cols = model.output_shape[2]

    u5 = Conv2D(para.channels, (1, 1), padding="valid")(u4)
    seg_output = Activation("sigmoid", name='segmentModel')(u5)

    model = Model(d0, seg_output)

    return model, rows, cols


if __name__ == '__main__':
    m, rows, cols = unet_lite_aug(alpha=0.25)
    print(m.summary())