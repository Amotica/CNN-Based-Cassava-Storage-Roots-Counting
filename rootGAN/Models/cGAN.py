from __future__ import print_function, division

from keras.layers import Input, Dropout, Concatenate, Cropping2D, Activation, Add

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Lambda
from keras.models import Model
import parameters as para
from keras.applications import VGG19, VGG16, InceptionResNetV2
import tensorflow as tf
from keras import backend as K
import os


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


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


def conv2d(layer_input, filters, f_size=4, bn=True):
    """Layers used during downsampling"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    return d


def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
    """Layers used during upsampling"""
    #u = UpSampling2D(size=2)(layer_input)
    u = nearest_upsampling(stride=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    u = BatchNormalization(momentum=0.8)(u)

    ch, cw = get_crop_shape(u, skip_input)
    crop_u = Cropping2D(cropping=(ch, cw))(u)

    u = Concatenate()([crop_u, skip_input])

    return u


def d_layer(layer_input, filters, f_size=4, bn=True):
    """Discriminator layer"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    if bn:
        d = BatchNormalization(momentum=0.8)(d)
    return d


def generativeModel(input_shape=(para.img_rows, para.img_cols, para.channels), gf=64):
    """U-Net Generator"""

    d0 = Input(shape=input_shape)

    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)
    d5 = conv2d(d4, gf * 8)
    d6 = conv2d(d5, gf * 8)
    d7 = conv2d(d6, gf * 8)
    d8 = conv2d(d7, gf * 8)
    d9 = conv2d(d8, gf * 8)
    d10 = conv2d(d9, gf * 8)

    # Upsampling
    u1 = deconv2d(d10, d9, gf * 8)
    u2 = deconv2d(u1, d8, gf * 8)
    u3 = deconv2d(u2, d7, gf * 8)
    u4 = deconv2d(u3, d6, gf * 8)
    u5 = deconv2d(u4, d5, gf * 8)
    u6 = deconv2d(u5, d4, gf * 8)
    u7 = deconv2d(u6, d3, gf * 4)
    u8 = deconv2d(u7, d2, gf * 2)
    u9 = deconv2d(u8, d1, gf)

    #u10 = UpSampling2D(size=2)(u9)
    u10 = nearest_upsampling(stride=2)(u9)
    output_img = Conv2D(para.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u10)

    model = Model(d0, output_img)

    model_folder = para.home_dir + para.model_type + '/Models/'
    model_file = model_folder + 'weights_gen.h5'
    if os.path.exists(model_file):
        model.load_weights(model_file)

    return model


def generativeModel2(input_shape=(para.img_rows, para.img_cols, para.channels), gf=64):
    """U-Net Generator"""

    d0 = Input(shape=input_shape)

    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)
    d5 = conv2d(d4, gf * 8)
    d6 = conv2d(d5, gf * 8)
    d7 = conv2d(d6, gf * 8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf * 8)
    u2 = deconv2d(u1, d5, gf * 8)
    u3 = deconv2d(u2, d4, gf * 8)
    u4 = deconv2d(u3, d3, gf * 4)
    u5 = deconv2d(u4, d2, gf * 2)
    u6 = deconv2d(u5, d1, gf)

    #u7 = UpSampling2D(size=2)(u6)
    u7 = nearest_upsampling(stride=2)(u6)
    output_img = Conv2D(para.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    model = Model(d0, output_img)

    model_folder = para.home_dir + 'Models/' + para.model_type + '/'
    model_file = model_folder + 'weights_gen.h5'
    if os.path.exists(model_file):
        model.load_weights(model_file)

    return model


def discriminativeModel(input_shape=(para.img_rows, para.img_cols, para.channels), df=64):
    img_A = Input(shape=input_shape)
    img_B = Input(shape=input_shape)

    # Concatenate image and conditioning image by channels to produce input

    combined_imgs = Concatenate(axis=-1)([img_A, img_B])
    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    model = Model([img_A, img_B], validity)

    rows = model.output_shape[1]
    cols = model.output_shape[2]

    model_folder = para.home_dir + 'Models/' + para.model_type + '/'
    model_file = model_folder + 'weights_dis.h5'
    if os.path.exists(model_file):
        model.load_weights(model_file)

    return model, rows, cols


def build_vgg19(input_shape=(para.img_rows, para.img_cols, para.channels)):
    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    vgg = VGG19(weights="imagenet")

    # Set outputs to outputs of last conv. layer in block 3
    vgg.outputs = [vgg.layers[5].output]

    img = Input(shape=input_shape)

    # Extract image features
    img_features = vgg(img)
    model = Model(img, img_features)

    return model


def build_vgg16(input_shape=(para.img_rows, para.img_cols, para.channels)):
    """
    Builds a pre-trained VGG19 model that outputs image features extracted at the
    third block of the model
    """
    vgg = VGG16(weights="imagenet")

    # Set outputs to outputs of last conv. layer in block 3
    # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
    vgg.outputs = [vgg.layers[5].output]

    img = Input(shape=input_shape)

    # Extract image features
    img_features = vgg(img)
    model = Model(img, img_features)

    return model


if __name__ == "__main__":
    # use bi-linear / bi-cubic / nearest-neighbour up-sampling
    model, rows, cols = discriminativeModel()
    print(rows, cols)
    model.summary()
