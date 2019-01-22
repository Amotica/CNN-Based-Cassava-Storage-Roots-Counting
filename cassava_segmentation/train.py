from keras.callbacks import *
import matplotlib.pyplot as plt
import parameters as para
from Models import segnet, segnet_lite, unet_lite
import numpy as np
from utils.memory_usage import memory_usage
import dataset_utils
from keras.utils import np_utils
import tensorflow as tf
import os


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    # ==================
    # Call model
    # ==================
    output_rows = 0
    output_cols = 0

    if para.model_type == "segnet_lite":
        print('Initialising Segnet Lite...')
        model, output_rows, output_cols = segnet_lite.SegNet()

    if para.model_type == "segnet":
        print('Initialising SegNet...')
        model, output_rows, output_cols = segnet.SegNet()

    if para.model_type == "unet_lite":
        print('Initialising Unet Lite...')
        model, output_rows, output_cols = unet_lite.unet_lite()

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
    model_checkpoint = ModelCheckpoint(check_pt_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=10, verbose=1, mode='auto',
                                      epsilon=0.0001, cooldown=0, min_lr=0)
    tensorboard_log = TensorBoard(log_dir=models_log_folder + '/', histogram_freq=0,
                                  write_graph=True, write_images=True)

    #   Compile the model using Adam optimizer
    #   =====================================
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    print(model.summary())

    #   print memory requirement by this model
    #   =====================================================
    gigabytes = memory_usage(model, batch_size=para.batch_size)
    print('Model will need ', gigabytes, 'GB of Memory')

    #   Load the training dataset
    #   =========================
    train_images, train_labels = dataset_utils.image_mask_processor2(para.train, para.trainannot, para.img_rows, para.img_cols)
    val_images, val_labels = dataset_utils.image_mask_processor2(para.val, para.valannot, para.img_rows, para.img_cols)

    train_labels = np_utils.to_categorical(train_labels, para.seg_classes)
    train_labels = np.reshape(train_labels, (len(train_images), para.img_rows * para.img_cols, para.seg_classes))
    steps_per_epoch = len(train_images) / para.batch_size
    train_images = train_images.astype(np.float32)
    train_images = train_images / 255.0

    val_labels = np_utils.to_categorical(val_labels, para.seg_classes)
    val_labels = np.reshape(val_labels, (len(val_labels), para.img_rows * para.img_cols, para.seg_classes))
    #steps_per_epoch = len(train_images) / para.batch_size
    val_images = val_images.astype(np.float32)
    val_images = val_images / 255.0

    history = model.fit(train_images,
                        train_labels,
                        callbacks=[model_checkpoint, tensorboard_log, reduce_lr],
                        batch_size=para.batch_size,
                        epochs=para.num_epoch,
                        verbose=1,
                        class_weight=para.class_weighting,
                        validation_data=(val_images, val_labels),
                        shuffle=True
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
