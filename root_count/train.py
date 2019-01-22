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

    #   Save the model after every epoch. NB: Save the best only
    #   ========================================================
    # Create folder if not exists
    models_folder = para.home_dir + 'misc/' + para.dataset + '/' + para.model_type
    if not os.path.exists(models_folder):
        os.makedirs(models_folder, mode=0o777)

    models_log_folder = para.home_dir + 'misc/' + para.dataset + '/' + para.model_type + '/tensorboard_log'
    if not os.path.exists(models_log_folder):
        os.makedirs(models_log_folder, mode=0o777)

    #   Set up callbacks to be assigned to the fitting stage
    check_pt_file = models_folder + '/weights.h5'
    model_checkpoint = ModelCheckpoint(check_pt_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', cooldown=0, min_lr=0)
    tensorboard_log = TensorBoard(log_dir=models_log_folder + '/', histogram_freq=0, write_graph=True, write_images=True)


    #   Compile the model using Adam optimizer
    #   ====================================
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    #   Compile the model using SGD optimizer
    #   =====================================
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])
    #print(model.summary())

    #   print memory requirement by this model
    #   =====================================================
    gigabytes = memory_usage(model, batch_size=para.batch_size)
    print('Model will need ', gigabytes, 'GB of Memory')


    #   Load the training dataset
    #   =========================
    if not para.data_gen:
        # Training Set
        train_images, train_labels, img_names = dataset_utils.image_class_label_processor(para.train, para.img_rows, para.img_cols)
        train_images = train_images.astype(np.float32)
        train_images = train_images / 255.0
        train_labels = np_utils.to_categorical(train_labels, para.count_classes)

        # Validation Set
        val_images, val_labels, _ = dataset_utils.image_class_label_processor(para.val, para.img_rows, para.img_cols)
        val_images = val_images.astype(np.float32)
        val_images = val_images / 255.0
        val_labels = np_utils.to_categorical(val_labels, para.count_classes)

        history = model.fit(
            train_images,
            train_labels,
            callbacks=[model_checkpoint, tensorboard_log, reduce_lr],
            batch_size=para.batch_size,
            epochs=para.num_epoch,
            verbose=1,
            validation_data=(val_images, val_labels),
            shuffle=True
        )

    else:
        # Training Set
        train_images, train_labels, _ = dataset_utils.image_class_label_processor(para.train, para.img_rows, para.img_cols)
        steps_per_epoch = len(train_images) // para.batch_size
        train_images = train_images.astype(np.float32)
        train_images = train_images / 255.0

        train_labels = np_utils.to_categorical(train_labels, para.count_classes)

        # Validation Set
        val_images, val_labels, _ = dataset_utils.image_class_label_processor(para.val, para.img_rows, para.img_cols)
        val_images = val_images.astype(np.float32)
        val_images = val_images / 255.0

        val_labels = np_utils.to_categorical(val_labels, para.count_classes)

        # construct the image generator for data augmentation
        image_generator = ImageDataGenerator(
            rescale=1. / 255.0,
            zoom_range=0.2,
            brightness_range=[0.2, 1.0],
            rotation_range=10,
            #horizontal_flip=True
        )

        history = model.fit_generator(
            image_generator.flow(train_images, train_labels, batch_size=para.batch_size),
            epochs=para.num_epoch,
            validation_data=(val_images, val_labels),
            steps_per_epoch=steps_per_epoch,
            shuffle=True,
            callbacks=[model_checkpoint, tensorboard_log, reduce_lr],
            verbose=1
        )


    #   summarize history for accuracy
    #   ==============================
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='upper left')
    plt.savefig(para.home_dir + 'misc/' + para.dataset + '/' + para.model_type + '/accuracy.png')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    #   summarize history for accuracy
    #   ==============================
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    plt.savefig(para.home_dir + 'misc/' + para.dataset + '/' + para.model_type + '/loss.png')
    plt.show()