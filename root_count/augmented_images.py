import parameters as para
from Models import unet_lite, densenet
import dataset_utils
import os


def save_generated_augmented_images(epochs=5, dataset=para.dataset):
    for cls in range(para.count_classes):
        train_path = para.home_dir + dataset + "/train/" + str(cls) + "/"
        train_mask_path = para.home_dir + dataset + "/trainannot_rgb/" + str(cls) + "/"

        train_path_save_to = para.home_dir + dataset + "/train/" + str(cls) + "_/"
        if not os.path.exists(train_path_save_to):
            os.makedirs(train_path_save_to, mode=0o777)

        train_mask_path_save_to = para.home_dir + dataset + "/trainannot_rgb/" + str(cls) + "_/"
        if not os.path.exists(train_mask_path_save_to):
            os.makedirs(train_mask_path_save_to, mode=0o777)

        val_path = para.home_dir + dataset + "/train/" + str(cls) + "/"
        val_mask_path = para.home_dir + dataset + "/trainannot_rgb/" + str(cls) + "/"

        model, output_rows, output_cols = unet_lite.unet_lite_aug()

        #   Compile the model using Adam optimizer
        #   =====================================
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        print(model.summary())

        #images, masks = dataset_utils.get_images_masks(train_path, train_mask_path, para.img_rows, para.img_cols, cls=cls)

        train_generator = dataset_utils.train_image_mask_processor(train_path, train_mask_path, train_path_save_to,
                                                                   train_mask_path_save_to)
        val_generator = dataset_utils.val_image_mask_processor(val_path, val_mask_path)

        print(dataset_utils.count_samples_aug(train_path, cls=0))

        steps_per_epoch = int(dataset_utils.count_samples_aug(train_path, cls=0) / para.batch_size)
        validation_steps = int(dataset_utils.count_samples_aug(val_path, cls=0) / para.batch_size)

        #young = 36 / old = 40
        st = 40 * epochs
        ct = dataset_utils.count_samples_aug(train_path, cls=0)
        
        epochs_ = int(st / ct)

        history = model.fit_generator(
            train_generator,
            epochs=epochs_,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            shuffle=True
        )


if __name__ == '__main__':
    save_generated_augmented_images(epochs=5)