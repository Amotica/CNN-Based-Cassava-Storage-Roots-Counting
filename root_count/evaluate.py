from keras.callbacks import *
import matplotlib.pyplot as plt
import parameters as para
from Models import smallVGGnet, densenet, densenet_lite
import numpy as np
from utils.memory_usage import memory_usage
import dataset_utils
from keras.utils import np_utils
import tensorflow as tf
import os
from keras import optimizers
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    if para.model_type == "smallVGGnet":
        print('Initialising smallVGGnet ...')
        model = smallVGGnet.smallVGGNet()
    if para.model_type == "densenet_121":
        print('Initialising Densenet 121 ...')
        model = densenet.DenseNet([6, 12, 24, 16])
    if para.model_type == "densenet_lite":
        print('Initialising Densenet Lite ...')
        model = densenet_lite.DenseNet([3, 6, 12, 8])

    #   Compile the model using Adam optimizer
    #   ====================================
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    #   print memory requirement by this model
    #   =====================================================
    gigabytes = memory_usage(model, batch_size=para.batch_size)
    print('Model will need ', gigabytes, 'GB of Memory')

    # Load the model weights
    model.load_weights(para.misc_dir + "/weights.h5")

    # Training Set
    test_images, test_labels, image_names = dataset_utils.image_class_label_processor(para.test, para.img_rows, para.img_cols)
    test_images = test_images.astype(np.float32)
    test_images = test_images / 255.0
    test_labels = np_utils.to_categorical(test_labels, para.count_classes)

    pred_labels = model.predict(test_images, batch_size=para.batch_size)

    for p_label, t_labels, img_names in zip(pred_labels, test_labels, image_names):
        t_labels = np.argmax(t_labels, axis=-1)

        top_3_idx = np.argsort(p_label)[-3:]
        top_3_values = [round(p_label[i] * 100) for i in top_3_idx]

        #p_label = np.argmax(p_label, axis=-1)
        print(img_names, t_labels, top_3_idx[2], top_3_idx[1], top_3_idx[0])



