import cv2
import glob
import parameters as para
from keras.preprocessing.image import img_to_array
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.utils import np_utils

#   =====================================
#   Counting functions
#   =====================================

def image_class_label_processor(image_dir, img_rows, img_cols, classes=para.count_classes):
    img = []
    lbl = []
    img_names = []

    for cls in range(classes):
        image_paths = glob.glob(image_dir + str(cls) + "/" + "*.jpg") + glob.glob(image_dir + str(cls) + "/" + "*.png") + glob.glob(
            image_dir + str(cls) + "/" + "*.jpeg")
        image_paths.sort()

        for img_path in image_paths:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (img_cols, img_rows), interpolation=cv2.INTER_NEAREST)

            image = img_to_array(image)
            img.append(image)

            img_names.append(os.path.basename(img_path))
            lbl.append(cls)

    return np.array(img), np.array(lbl), np.array(img_names)


def image_class_label_processor_2(image_dir, img_rows, img_cols, classes=para.count_classes, t_classes=para.type_classes):
    img = []
    lbl = []
    img_names = []
    for t_cls in range(t_classes):
        for cls in range(classes):
            image_paths = glob.glob(image_dir + str(t_cls) + "/" + str(cls) + "/" + "*.jpg") + glob.glob(
                image_dir + str(t_cls) + "/" + str(cls) + "/" + "*.png") + glob.glob(
                image_dir + str(t_cls) + "/" + str(cls) + "/" + "*.jpeg")
            image_paths.sort()

            for img_path in image_paths:
                image = cv2.imread(img_path)
                image = cv2.resize(image, (img_cols, img_rows), interpolation=cv2.INTER_NEAREST)

                image = img_to_array(image)
                img.append(image)

                img_names.append(os.path.basename(img_path))
                lbl.append(t_cls)

    return np.array(img), np.array(lbl), np.array(img_names)


#   =========================================
#   Augmentation Functions
#   =========================================
def count_samples_aug(image_dir, cls=0):
    image_paths = glob.glob(image_dir + str(cls) + "/" + "*.jpg") + glob.glob(
        image_dir + str(cls) + "/" + "*.png") + glob.glob(
        image_dir + str(cls) + "/" + "*.jpeg")
    return len(image_paths)


def train_image_mask_processor(train_path, train_mask_path, img_save_to, msk_save_to):
    # we create two instances with the same arguments
    data_gen_args_image = dict(
        zoom_range=0.2,
        brightness_range=[0.2, 1.0],
        rotation_range=10,
        horizontal_flip=True
    )

    data_gen_args_mask = dict(
        zoom_range=0.2,
        rotation_range=10,
        horizontal_flip=True
    )

    image_datagen = ImageDataGenerator(**data_gen_args_image)
    mask_datagen = ImageDataGenerator(**data_gen_args_mask)

    # Provide the same seed and keyword arguments to the fit and flow methods
    train_seed = 1

    image_generator = image_datagen.flow_from_directory(
        train_path, # this is the target directory
        class_mode=None,
        target_size=(para.img_rows, para.img_cols),
        batch_size=para.batch_size,
        seed=train_seed,
        shuffle=False
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        target_size=(para.img_rows, para.img_cols),
        batch_size=para.batch_size,
        class_mode=None,
        seed=train_seed,
        shuffle=False
    )

    # combine generators into one which yields image and masks
    f = 0
    for img, msk in zip(image_generator, mask_generator):
        # save the images and mask
        ind = image_generator.batch_index
        #reshape image and mask
        dis_img = img.reshape((para.img_rows, para.img_cols, para.channels))
        dis_img = np.array(dis_img).astype('uint8')
        dis_img = dis_img[..., ::-1]

        dis_msk = msk.reshape((para.img_rows, para.img_cols, para.channels))
        dis_msk = np.array(dis_msk).astype('uint8')
        dis_msk = dis_msk[..., ::-1]

        cv2.imwrite(img_save_to + str(ind) + "_" + str(f) + "_aug.png", dis_img)
        cv2.imwrite(msk_save_to + str(ind) + "_" + str(f) + "_aug.png", dis_msk)
        f+=1
        yield img, msk


def val_image_mask_processor(train_path, train_mask_path):
    # we create two instances with the same arguments
    data_gen_args_image = dict(
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #zoom_range=0.2
    )

    data_gen_args_mask = dict(
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #zoom_range=0.2
    )

    image_datagen = ImageDataGenerator(**data_gen_args_image)
    mask_datagen = ImageDataGenerator(**data_gen_args_mask)

    # Provide the same seed and keyword arguments to the fit and flow methods
    train_seed = 2

    image_generator = image_datagen.flow_from_directory(
        train_path, # this is the target directory
        class_mode=None,
        target_size=(para.img_rows, para.img_cols),
        batch_size=para.batch_size,
        seed=train_seed
    )

    mask_generator = mask_datagen.flow_from_directory(
        train_mask_path,
        target_size=(para.img_rows, para.img_cols),
        batch_size=para.batch_size,
        class_mode=None,
        seed=train_seed
    )

    # combine generators into one which yields image and masks

    for img, msk in zip(image_generator, mask_generator):
        yield img, msk


def change_storage_color(images_path="leaf/testannot/", target_path="leaf/testannot_new/"):
    image_paths = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(
        images_path + "*.jpeg") + glob.glob(images_path + "*.PNG") + glob.glob(images_path + "*.JPG") + glob.glob(
        images_path + "*.JPEG")

    if not os.path.exists(target_path):
        os.makedirs(target_path, mode=0o777)

    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        filename = target_path + base_name
        print(filename)

        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, 0)
        formed_image = np.dstack((thresh, thresh, thresh))
        formed_image[np.where((formed_image == [255, 255, 255]).all(axis=2))] = [50, 100, 200]
        cv2.imshow("image", formed_image)
        cv2.imwrite(filename, formed_image)
        cv2.waitKey(10)


def rgb_pixels_to_binary(images_path="leaf/testannot/", target_path="leaf/testannot_new/"):
    image_paths = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(
        images_path + "*.jpeg") + glob.glob(images_path + "*.PNG") + glob.glob(images_path + "*.JPG") + glob.glob(images_path + "*.JPEG")

    if not os.path.exists(target_path):
        os.makedirs(target_path, mode=0o777)

    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        filename = target_path + base_name
        print(filename)
        image = cv2.imread(img_path)

        image[np.where((image == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        image[np.where((image == [50, 100, 200]).all(axis=2))] = [1, 1, 1]
        image[np.where((image == [0, 255, 255]).all(axis=2))] = [2, 2, 2]

        cv2.imshow("image", image)
        cv2.imwrite(filename, image)
        cv2.waitKey(10)



if __name__ == '__main__':
    images_path = "cassava_dataset_synthetic/old_cassava_real_synthetic_aug/train/"
    image_paths = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(
        images_path + "*.jpeg") + glob.glob(images_path + "*.PNG") + glob.glob(images_path + "*.JPG") + glob.glob(
        images_path + "*.JPEG")
    print(len(image_paths))

    #image = cv2.imread("./cassava_dataset_sythetic/old_cassava_real_synthetic_aug/trainannot_rgb_new/0/11_135_aug.png")
    #unique_pixels = np.vstack({tuple(r) for r in image.reshape(-1, 3)})
    #print(unique_pixels)

    #for cls in range(8):
        #change_storage_color(images_path="./cassava_dataset_synthetic/young_cassava_aug/trainannot_rgb/" + str(cls) + "/",
                            #target_path="./cassava_dataset_synthetic/young_cassava_aug/trainannot_rgb_new/" + str(cls) + "/")

    #for cls in range(8):
            #rgb_pixels_to_binary(images_path="./cassava_dataset_synthetic/young_cassava_aug/trainannot_rgb/"+ str(cls) + "/",
                                 #target_path="./cassava_dataset_synthetic/young_cassava_aug/trainannot_rgb_new/"+ str(cls) + "/")