from keras.callbacks import *
from Models import xception
import parameters as para
import data_loader
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # create model
    model = xception.xception(input_shape=(para.img_rows, para.img_cols, para.channels), classes=para.num_classes)

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
    tensorboard_log = TensorBoard(log_dir=models_log_folder + '/', histogram_freq=0, write_graph=True,
                                  write_images=True)

    #   Compile the model using Adam optimizer
    #   =====================================
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    print(model.summary())

#   Load the training dataset
    #   =========================
    train_generator, validation_generator = data_loader.train_validation_images(batch_size=para.batch_size,
                                                                                train_path=para.train,
                                                                                validation_path=para.val)

    history = model.fit_generator(
        train_generator,
        epochs=para.num_epoch,
        validation_data=validation_generator,
        shuffle=True,
        callbacks = [model_checkpoint, tensorboard_log, reduce_lr]
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


