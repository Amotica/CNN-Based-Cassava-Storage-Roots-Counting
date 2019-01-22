import cv2
import os
import parameters as para
from keras.utils import np_utils
import glob
import ntpath
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img


def get_fore_back_train_test_annot(trainval, trainvalannot, annot_all):
    trainval_paths = glob.glob(trainval + "0/" + "*.jpg") + glob.glob(trainval + "0/" + "*.png") + glob.glob(
        trainval + "0/" + "*.jpeg")

    for trainval_path in trainval_paths:
        annot_all_path = annot_all + "0/" + os.path.basename(trainval_path)
        mask = cv2.imread(annot_all_path)
        cv2.imwrite(trainvalannot + "0/" + os.path.basename(trainval_path), mask)


def split_train_test(image_dir, mask_dir, mask_multi_dir, train=0.80):
    image_paths = glob.glob(image_dir + "0/" + "*.jpg") + glob.glob(image_dir + "0/" + "*.png") + glob.glob(
        image_dir + "0/" + "*.jpeg")
    np.random.shuffle(image_paths)
    train_value = int(round(len(image_paths) * train))
    train, test = image_paths[:train_value], image_paths[train_value:]

    #Save the train data
    for train_path in train:
        train_msk_path = mask_dir + "0/" + os.path.basename(train_path)
        train_msk_multi_path = mask_multi_dir + "0/" + os.path.basename(train_path)
        print(para.trainval + "0/" + os.path.basename(train_path))
        image = cv2.imread(train_path)
        mask = cv2.imread(train_msk_path)
        mask_multi = cv2.imread(train_msk_multi_path)

        if not os.path.exists(para.trainval + "0/"):
            os.makedirs(para.trainval + "0/", mode=0o777)
        if not os.path.exists(para.trainvalannot + "0/"):
            os.makedirs(para.trainvalannot + "0/", mode=0o777)
        if not os.path.exists(para.trainvalannot_multi + "0/"):
            os.makedirs(para.trainvalannot_multi + "0/", mode=0o777)

        cv2.imwrite(para.trainval + "0/" + os.path.basename(train_path), image)
        cv2.imwrite(para.trainvalannot + "0/" + os.path.basename(train_path), mask)
        cv2.imwrite(para.trainvalannot_multi + "0/" + os.path.basename(train_path), mask_multi)

    # Save the test data
    for test_path in test:
        test_msk_path = mask_dir + "0/" + os.path.basename(test_path)
        test_msk_multi_path = mask_multi_dir + "0/" + os.path.basename(test_path)
        print(para.test_data + "0/" + os.path.basename(test_path))
        image = cv2.imread(test_path)
        mask = cv2.imread(test_msk_path)
        mask_multi = cv2.imread(test_msk_multi_path)
        if not os.path.exists(para.test_data + "0/"):
            os.makedirs(para.test_data + "0/", mode=0o777)
        if not os.path.exists(para.test_data_annot + "0/"):
            os.makedirs(para.test_data_annot + "0/", mode=0o777)
        if not os.path.exists(para.test_data_annot_multi + "0/"):
            os.makedirs(para.test_data_annot_multi + "0/", mode=0o777)
        cv2.imwrite(para.test_data + "0/" + os.path.basename(test_path), image)
        cv2.imwrite(para.test_data_annot + "0/" + os.path.basename(test_path), mask)
        cv2.imwrite(para.test_data_annot_multi + "0/" + os.path.basename(test_path), mask_multi)

    return train, test


def rgb_pixels_to_binary(images_path="leaf/testannot/", target_path="leaf/testannot_new/"):
    image_paths = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(
        images_path + "*.jpeg")

    if not os.path.exists(target_path):
        os.makedirs(target_path, mode=0o777)

    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    for img_path in image_paths:
        base_name = ntpath.basename(img_path)
        filename = target_path + base_name
        print(filename)
        image = cv2.imread(img_path)

        image[np.where((image == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        image[np.where((image == [50, 100, 200]).all(axis=2))] = [1, 1, 1]
        image[np.where((image == [0, 255, 255]).all(axis=2))] = [2, 2, 2]
        #image[np.where((image == [0, 0, 204]).all(axis=2))] = [1, 1, 1]

        cv2.imshow("image", image)
        cv2.imwrite(filename, image)
        cv2.waitKey(10)


def image_mask_processor(image_dir, mask_dir, img_rows, img_cols):
    image_paths = glob.glob(image_dir + "0/" + "*.jpg") + glob.glob(image_dir + "0/" + "*.png") + glob.glob(
        image_dir + "0/" + "*.jpeg")
    image_paths.sort()
    mask_paths = glob.glob(mask_dir + "0/" + "*.jpg") + glob.glob(mask_dir + "0/" + "*.png") + glob.glob(
        mask_dir + "0/" + "*.jpeg")
    mask_paths.sort()
    img = []
    msk = []
    for img_path, msk_path in zip(image_paths, mask_paths):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (img_cols, img_rows), interpolation=cv2.INTER_NEAREST)
        img.append(image)

        mask = cv2.imread(msk_path)
        mask = cv2.resize(mask, (img_cols, img_rows), interpolation=cv2.INTER_NEAREST)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        msk.append(mask)

    return np.array(img), np.array(msk)


def image_mask_processor2(image_dir, mask_dir, img_rows, img_cols, classes=para.count_classes):
    img = []
    msk = []
    for cls in range(classes):
        image_paths = glob.glob(image_dir + str(cls) + "/" + "*.jpg") + glob.glob(image_dir + str(cls) + "/" + "*.png") + glob.glob(
            image_dir + str(cls) + "/" + "*.jpeg")
        image_paths.sort()
        mask_paths = glob.glob(mask_dir + str(cls) + "/" + "*.jpg") + glob.glob(mask_dir + str(cls) + "/" + "*.png") + glob.glob(
            mask_dir + str(cls) + "/" + "*.jpeg")
        mask_paths.sort()

        for img_path, msk_path in zip(image_paths, mask_paths):
            image = cv2.imread(img_path)
            image = cv2.resize(image, (img_cols, img_rows), interpolation=cv2.INTER_NEAREST)
            img.append(image)

            mask = cv2.imread(msk_path)
            mask = cv2.resize(mask, (img_cols, img_rows), interpolation=cv2.INTER_NEAREST)
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            msk.append(mask)

    return np.array(img), np.array(msk)


def prepare_evaluation_images(image_dir, mask_dir, img_rows, img_cols, img_rows_out, img_cols_out):
    image_paths = glob.glob(image_dir + "0/" + "*.jpg") + glob.glob(image_dir + "0/" + "*.png") + glob.glob(
        image_dir + "0/" + "*.jpeg")
    image_paths.sort()
    mask_paths = glob.glob(mask_dir + "0/" + "*.jpg") + glob.glob(mask_dir + "0/" + "*.png") + glob.glob(
        mask_dir + "0/" + "*.jpeg")
    mask_paths.sort()

    img = []
    msk = []
    img_names = []
    for img_path, msk_path in zip(image_paths, mask_paths):
        #   Open and resize the image to the input shape and
        #   prepare it for the classifier and the mask to the output shape
        img_names.append(ntpath.basename(img_path))
        #   Image
        image = cv2.imread(img_path)
        image = cv2.resize(image, (img_cols, img_rows), interpolation=cv2.INTER_NEAREST)
        image = np.array(image).astype('uint8')
        image = image.astype(np.float32)
        image = image / 255.0
        img.append(image)

        #   Mask
        mask = cv2.imread(msk_path)
        mask = cv2.resize(mask, (img_cols_out, img_rows_out), interpolation=cv2.INTER_NEAREST)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = np.array(mask).astype('uint8')
        mask = np_utils.to_categorical(mask, num_classes=para.seg_classes)
        msk.append(mask)

    return np.array(img), np.array(msk), np.array(img_names)


def visualise_mask(msk, dataset=para.dataset):
    #   convert prediction to same channel as ground truth mask
    #print(msk.shape)
    #pred_mask = msk
    pred_mask = np.zeros((msk.shape[0], msk.shape[1], 3))

    pred_mask[:, :, 0] = msk
    pred_mask[:, :, 1] = msk
    pred_mask[:, :, 2] = msk

    # CamVid Classes = 12
    if dataset == "CamVid":
        pred_mask[np.where((pred_mask == [0, 0, 0]).all(axis=2))] = [128, 128, 128]
        pred_mask[np.where((pred_mask == [1, 1, 1]).all(axis=2))] = [128, 0, 0]
        pred_mask[np.where((pred_mask == [2, 2, 2]).all(axis=2))] = [192, 192, 128]

        pred_mask[np.where((pred_mask == [3, 3, 3]).all(axis=2))] = [128, 64, 128]
        pred_mask[np.where((pred_mask == [4, 4, 4]).all(axis=2))] = [60, 40, 222]
        pred_mask[np.where((pred_mask == [5, 5, 5]).all(axis=2))] = [128, 128, 0]

        pred_mask[np.where((pred_mask == [6, 6, 6]).all(axis=2))] = [192, 128, 128]
        pred_mask[np.where((pred_mask == [7, 7, 7]).all(axis=2))] = [64, 64, 128]
        pred_mask[np.where((pred_mask == [8, 8, 8]).all(axis=2))] = [64, 0, 128]

        pred_mask[np.where((pred_mask == [9, 9, 9]).all(axis=2))] = [64, 64, 0]
        pred_mask[np.where((pred_mask == [10, 10, 10]).all(axis=2))] = [0, 128, 192]
        pred_mask[np.where((pred_mask == [11, 11, 11]).all(axis=2))] = [0, 0, 0]

    if dataset == "flowers_multi_class":
        pred_mask[np.where((pred_mask == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        pred_mask[np.where((pred_mask == [1, 1, 1]).all(axis=2))] = [128, 0, 0]
        pred_mask[np.where((pred_mask == [2, 2, 2]).all(axis=2))] = [192, 192, 128]

        pred_mask[np.where((pred_mask == [3, 3, 3]).all(axis=2))] = [128, 64, 128]
        pred_mask[np.where((pred_mask == [4, 4, 4]).all(axis=2))] = [60, 40, 222]
        pred_mask[np.where((pred_mask == [5, 5, 5]).all(axis=2))] = [128, 128, 0]

        pred_mask[np.where((pred_mask == [6, 6, 6]).all(axis=2))] = [192, 128, 128]
        pred_mask[np.where((pred_mask == [7, 7, 7]).all(axis=2))] = [64, 64, 128]
        pred_mask[np.where((pred_mask == [8, 8, 8]).all(axis=2))] = [64, 0, 128]

        pred_mask[np.where((pred_mask == [9, 9, 9]).all(axis=2))] = [64, 64, 0]
        pred_mask[np.where((pred_mask == [10, 10, 10]).all(axis=2))] = [0, 128, 192]
        pred_mask[np.where((pred_mask == [11, 11, 11]).all(axis=2))] = [0, 0, 128]

        pred_mask[np.where((pred_mask == [12, 12, 12]).all(axis=2))] = [64, 128, 0]

    if dataset == "flowers_fore_back":
        pred_mask[np.where((pred_mask == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        pred_mask[np.where((pred_mask == [1, 1, 1]).all(axis=2))] = [255, 0, 255]

    if dataset == "leaf":
        pred_mask[np.where((pred_mask == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        pred_mask[np.where((pred_mask == [1, 1, 1]).all(axis=2))] = [255, 0, 255]
        pred_mask[np.where((pred_mask == [2, 2, 2]).all(axis=2))] = [255, 255, 0]

    if dataset == "cassava_fore_back" or "cassava_fore_back_640x480" or "cassava_fore_back_480x640":
        pred_mask[np.where((pred_mask == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        pred_mask[np.where((pred_mask == [1, 1, 1]).all(axis=2))] = [50, 100, 200]
    if dataset == "cassava_multi_class" or "cassava_multi_class_640x480" or "cassava_multi_class_480x640":
        pred_mask[np.where((pred_mask == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
        pred_mask[np.where((pred_mask == [1, 1, 1]).all(axis=2))] = [50, 100, 200]
        pred_mask[np.where((pred_mask == [2, 2, 2]).all(axis=2))] = [0, 255, 255]

    return np.array(pred_mask)


def class_weights(images_path):
    n_class = para.seg_classes
    n_cls_pixels = np.zeros((n_class,))
    n_img_pixels = np.zeros((n_class,))

    image_paths = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(
        images_path + "*.jpeg")
    for img_path in image_paths:
        label = cv2.imread(img_path)
        # resize the images
        label = cv2.resize(label, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
        for cls_i in np.unique(label):
            print(cls_i, img_path)
            n_cls_pixels[cls_i] += np.sum(label == cls_i)
            n_img_pixels[cls_i] += label.size

    freq = n_cls_pixels / n_img_pixels
    median_freq = np.median(freq)

    print(median_freq / freq)


def train_image_mask_generator(image_dir, mask_dir):
    image_datagen = ImageDataGenerator(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    mask_datagen = ImageDataGenerator(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    count = 10
    image_gen = image_datagen.flow_from_directory(
        image_dir,
        target_size=(para.img_rows, para.img_cols),
        batch_size=para.batch_size,
        class_mode=None,
        seed=seed
    )

    mask_gen = mask_datagen.flow_from_directory(
        mask_dir,
        target_size=(para.img_rows, para.img_cols),
        batch_size=para.batch_size,
        class_mode=None,
        seed=seed
    )

    # let's create infinite flow of images

    train_generator = zip(image_gen, mask_gen)

    while True:
        image_gen.next()
        image_gen.next()


if __name__ == '__main__':
    # 1. split the dataset.
    #split_train_test("./cassava_fore_back/train_val_all/", "./cassava_fore_back/train_valannot_all/", "./cassava_fore_back/train_val_final_all/", train=0.80)
    # 2. convert rgb annotations to binary annotations
    #rgb_pixels_to_binary(images_path="./cassava_multi_class/trainvalannot_rgb/0/",
                         #target_path="./cassava_multi_class/trainvalannot_new/0/")
    #rgb_pixels_to_binary(images_path="./cassava_multi_class/testannot_rgb/0/",
                        #target_path="./cassava_multi_class/testannot_new/0/")
    # 3. Get the class weights
    #class_weights("./cassava_fore_back/trainvalannot/0/")
    class_weights("./cassava_multi_class_480x640/trainvalannot/0/")
    # Copy the trainval and test folder from multiclass. then select the correct annotations from fore_back folder
    #get_fore_back_train_test_annot("./cassava_fore_back/trainval/", "./cassava_fore_back/trainvalannot/",
                                   #"./cassava_fore_back/annot_all/")
   #get_fore_back_train_test_annot("./cassava_fore_back/test/", "./cassava_fore_back/testannot/",
                                  #"./cassava_fore_back/annot_all/")

    #train_image_mask_generator("cassava_fore_back/trainval/", "cassava_fore_back/trainvalannot/")

    #image = cv2.imread("./cassava_fore_back/all_data/train_val_final/121.png")
    #unique_pixels = np.vstack({tuple(r) for r in image.reshape(-1, 3)})
    #print(unique_pixels)
