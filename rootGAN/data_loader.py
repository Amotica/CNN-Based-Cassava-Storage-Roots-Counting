import glob, os
import numpy as np
from scipy.misc import imread, imresize
import parameters as para
from keras.preprocessing.image import ImageDataGenerator
import cv2
import csv
from keras.utils import np_utils


img_res = (para.img_rows, para.img_cols)


def load_batch(image_dir, mask_dir, batch_size=6, is_testing=False):

    image_paths = glob.glob(image_dir + "0/" + "*.jpg") + glob.glob(image_dir + "0/" + "*.png") + glob.glob(
        image_dir + "0/" + "*.jpeg")
    image_paths.sort()
    mask_paths = glob.glob(mask_dir + "0/" + "*.jpg") + glob.glob(mask_dir + "0/" + "*.png") + glob.glob(
        mask_dir + "0/" + "*.jpeg")
    mask_paths.sort()

    n_batches = int(len(image_paths) / batch_size)
    total_samples = len(image_paths)

    index = np.arange(total_samples-1)
    np.random.shuffle(index)

    for i in range(n_batches - 1):

        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_images, batch_masks=[], []
        for b_i in batch_index:
            batch_images.append(image_paths[b_i])
            batch_masks.append(mask_paths[b_i])

        imgs, msks = [], []
        for img, msk in zip(batch_images, batch_masks):
            img = imread(img)
            msk = imread(msk)

            img = imresize(img, img_res)
            msk = imresize(msk, img_res)

            if not is_testing and np.random.random() > 0.5:
                img = np.fliplr(img)
                msk = np.fliplr(msk)

            imgs.append(img)
            msks.append(msk)

        imgs = np.array(imgs) / 127.5 - 1.
        msks = np.array(msks) / 127.5 - 1.

        yield imgs, msks, n_batches


def load_data(image_dir, mask_dir, batch_size=1, is_testing=True):
    image_paths = glob.glob(image_dir + "0/" + "*.jpg") + glob.glob(image_dir + "0/" + "*.png") + glob.glob(
        image_dir + "0/" + "*.jpeg")
    image_paths.sort()
    mask_paths = glob.glob(mask_dir + "0/" + "*.jpg") + glob.glob(mask_dir + "0/" + "*.png") + glob.glob(
        mask_dir + "0/" + "*.jpeg")
    mask_paths.sort()

    n_batches = int(len(image_paths) / batch_size)
    total_samples = len(image_paths)

    index = np.arange(total_samples - 1)
    np.random.shuffle(index)

    for i in range(n_batches - 1):

        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_images, batch_masks=[], []
        for b_i in batch_index:
            batch_images.append(image_paths[b_i])
            batch_masks.append(mask_paths[b_i])

        imgs, msks = [], []
        for img, msk in zip(batch_images, batch_masks):
            img = imread(img)
            msk = imread(msk)

            img = imresize(img, img_res)
            msk = imresize(msk, img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img = np.fliplr(img)
                msk = np.fliplr(msk)

            imgs.append(img)
            msks.append(msk)

        imgs = np.array(imgs) / 127.5 - 1.
        msks = np.array(msks) / 127.5 - 1.

        return imgs, msks


def load_synthetic_mask(synthetic_mask_dir, is_testing=True, cls=0):
    synthetic_mask_paths = glob.glob(synthetic_mask_dir + str(cls) + "/" + "*.jpg") + glob.glob(synthetic_mask_dir + str(cls) + "/" + "*.png") + glob.glob(
        synthetic_mask_dir + str(cls) + "/" + "*.jpeg")
    synthetic_mask_paths.sort()

    msks = []
    for msk in synthetic_mask_paths:
        msk = imread(msk)

        msk = imresize(msk, img_res)

        # If training => do random flip
        if not is_testing and np.random.random() < 0.5:
            msk = np.fliplr(msk)

        msks.append(msk)

    msks = np.array(msks) / 127.5 - 1.

    return msks


def load_data_synthetic(msk):
    msks = []
    msk = imresize(msk, img_res)
    msks.append(msk)

    msks = np.array(msks) / 127.5 - 1.

    return msks


def train_validation_images(batch_size=para.batch_size, train_path="data/train", validation_path="data/validation"):

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True
    )

    # this is the augmentation configuration we will use for validation:
    validation_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=(para.img_rows, para.img_cols),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical' # since we use crossentropy loss, we need categorical labels
    )

    # this is a similar generator, for validation data
    validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=(para.img_rows, para.img_cols),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator


def test_images(test_path="data/test", classes=para.num_classes):
    print(test_path)
    imgs = []
    class_labels = []
    for cls in range(classes):
        image_paths = glob.glob(test_path + str(cls) + "/" + "*.jpg") + glob.glob(test_path + str(cls) + "/" + "*.png") + glob.glob(
            test_path + str(cls) + "/" + "*.jpeg") + glob.glob(test_path + str(cls) + "/" + "*.PNG")
        image_paths.sort()
        print(len(image_paths))
        for img in image_paths:
            image = imread(img)

            image = cv2.resize(image, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
            image = image / 255.
            imgs.append(image)
            cls_label = np_utils.to_categorical(cls, num_classes=para.num_classes)
            class_labels.append(cls_label)

    return np.array(imgs), np.array(class_labels)


def train_validation_test_splits(image_dir, classes=para.num_classes, train_ratio=0.8):
        for cls in range(classes):

            # make a folder for each class : train/validate/test
            train_folder = image_dir + "train/" + str(cls) + "/"
            if not os.path.exists(train_folder):
                os.makedirs(train_folder, mode=0o777)

            validation_folder = image_dir + "val/" + str(cls) + "/"
            if not os.path.exists(validation_folder):
                os.makedirs(validation_folder, mode=0o777)

            test_folder = image_dir + "test/" + str(cls) + "/"
            if not os.path.exists(test_folder):
                os.makedirs(test_folder, mode=0o777)

            trainannot_folder = image_dir + "trainannot/" + str(cls) + "/"
            if not os.path.exists(trainannot_folder):
                os.makedirs(trainannot_folder, mode=0o777)

            valannot_folder = image_dir + "valannot/" + str(cls) + "/"
            if not os.path.exists(valannot_folder):
                os.makedirs(valannot_folder, mode=0o777)

            testannot_folder = image_dir + "testannot/" + str(cls) + "/"
            if not os.path.exists(testannot_folder):
                os.makedirs(testannot_folder, mode=0o777)

            image_paths = glob.glob(image_dir + str(cls) + "/" + "*_syn.jpg") + glob.glob(
                image_dir + str(cls) + "/" + "*_syn.png") + glob.glob(
                image_dir + str(cls) + "/" + "*_syn.jpeg")

            # randomly shuffle the arrray
            np.random.shuffle(image_paths)

            # get the total samples in array
            total_samples = len(image_paths)

            # calculate the total samples for each split
            num_trainval = int(total_samples * train_ratio)
            trainval_image_paths = image_paths[0:num_trainval]
            test_image_paths = image_paths[num_trainval:total_samples]

            # for validation and train splits
            total_trainval = len(trainval_image_paths)
            # randomly shuffle the arrray
            np.random.shuffle(trainval_image_paths)
            num_train = int(total_trainval * train_ratio)
            train_image_paths = image_paths[0:num_train]
            val_image_paths = image_paths[num_train:total_trainval]

            # Now save the train images
            for train_path in train_image_paths:
                print(train_path)
                train_image = cv2.imread(train_path)
                train_image_file = train_folder + os.path.basename(train_path)
                cv2.imwrite(train_image_file, train_image)
                # get anotation image and save them too
                # get anotation image and save them too
                trainannot_path = os.path.dirname(train_path) + "/" + os.path.splitext(os.path.basename(train_path))[0][
                                                                      :-4] + "_mask.png"
                trainannot_image = cv2.imread(trainannot_path)
                trainannot_image_file = trainannot_folder + os.path.basename(trainannot_path)
                cv2.imwrite(trainannot_image_file, trainannot_image)

            # Now save the validation images
            for val_path in val_image_paths:
                print(val_path)
                val_image = cv2.imread(val_path)
                val_image_file = validation_folder + os.path.basename(val_path)
                cv2.imwrite(val_image_file, val_image)
                # get anotation image and save them too
                valannot_path = os.path.dirname(val_path) + "/" + os.path.splitext(os.path.basename(val_path))[0][
                                                                      :-4] + "_mask.png"
                valannot_image = cv2.imread(valannot_path)
                valannot_image_file = valannot_folder + os.path.basename(valannot_path)
                cv2.imwrite(valannot_image_file, valannot_image)

            # Now save the test images
            for test_path in test_image_paths:
                print(test_path)
                test_image = cv2.imread(test_path)
                test_image_file = test_folder + os.path.basename(test_path)
                cv2.imwrite(test_image_file, test_image)
                # get anotation image and save them too
                testannot_path = os.path.dirname(test_path) + "/" + os.path.splitext(os.path.basename(test_path))[0][
                                                                  :-4] + "_mask.png"
                testannot_image = cv2.imread(testannot_path)
                testannot_image_file = testannot_folder + os.path.basename(testannot_path)
                cv2.imwrite(testannot_image_file, testannot_image)


def train_val_test_real_into_class(csv_file="cassava_real/count.csv"):
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row['filename'], row['storage_roots'])

            # check is exist in test set. If exists then save in appropriate class in test set
            test_path = "cassava_real/test/"
            testannot_path = "cassava_real/testannot_rgb/"
            test_file = test_path + "_0/" + row['filename']
            testannot_file = testannot_path + "_0/" + row['filename']
            if os.path.isfile(test_file):
                #create folder number of storage roots and save file in that folder
                test_folder = test_path + str(row['storage_roots']) + "/"
                if not os.path.exists(test_folder):
                    os.makedirs(test_folder, mode=0o777)
                # now save the file in this created folder
                test_image = cv2.imread(test_file)
                test_image_file = test_folder + row['filename']
                cv2.imwrite(test_image_file, test_image)

                testannot_folder = testannot_path + str(row['storage_roots']) + "/"
                if not os.path.exists(testannot_folder):
                    os.makedirs(testannot_folder, mode=0o777)
                # now save the file in this created folder
                testannot_image = cv2.imread(testannot_file)
                testannot_image_file = testannot_folder + row['filename']
                cv2.imwrite(testannot_image_file, testannot_image)

            # check is exist in test set. If exists then save in appropriate class in test set
            trainval_path = "cassava_real/trainval/"
            trainvalannot_path = "cassava_real/trainvalannot_rgb/"
            trainval_file = trainval_path + "_0/" + row['filename']
            trainvalannot_file = trainvalannot_path + "_0/" + row['filename']
            if os.path.isfile(trainval_file):
                # create folder number of storage roots and save file in that folder
                trainval_folder = trainval_path + str(row['storage_roots']) + "/"
                if not os.path.exists(trainval_folder):
                    os.makedirs(trainval_folder, mode=0o777)
                # now save the file in this created folder
                trainval_image = cv2.imread(trainval_file)
                trainval_image_file = trainval_folder + row['filename']
                cv2.imwrite(trainval_image_file, trainval_image)

                trainvalannot_folder = trainvalannot_path + str(row['storage_roots']) + "/"
                if not os.path.exists(trainvalannot_folder):
                    os.makedirs(trainvalannot_folder, mode=0o777)
                # now save the file in this created folder
                trainvalannot_image = cv2.imread(trainvalannot_file)
                trainvalannot_image_file = trainvalannot_folder + row['filename']
                cv2.imwrite(trainvalannot_image_file, trainvalannot_image)

            # check is exist in train set. If exists then save in appropriate class in train set


def train_val_split(image_dir, classes=para.num_classes, train_ratio=0.8):
    for cls in range(classes):

        # make a folder for each class : train/validate/test
        train_folder = image_dir + "train/" + str(cls) + "/"
        if not os.path.exists(train_folder):
            os.makedirs(train_folder, mode=0o777)

        validation_folder = image_dir + "val/" + str(cls) + "/"
        if not os.path.exists(validation_folder):
            os.makedirs(validation_folder, mode=0o777)

        trainannot_folder = image_dir + "trainannot/" + str(cls) + "/"
        if not os.path.exists(trainannot_folder):
            os.makedirs(trainannot_folder, mode=0o777)

        valannot_folder = image_dir + "valannot/" + str(cls) + "/"
        if not os.path.exists(valannot_folder):
            os.makedirs(valannot_folder, mode=0o777)

        image_paths = glob.glob(image_dir + str(cls) + "/" + "*.jpg") + glob.glob(
            image_dir + str(cls) + "/" + "*.png") + glob.glob(
            image_dir + str(cls) + "/" + "*.jpeg")

        # randomly shuffle the arrray
        np.random.shuffle(image_paths)

        # get the total samples in array
        total_samples = len(image_paths)

        # calculate the total samples for each split
        num_train = int(total_samples * train_ratio)
        train_image_paths = image_paths[0:num_train]
        val_image_paths = image_paths[num_train:total_samples]

        # Now save the train images
        for train_path in train_image_paths:
            print(train_path)
            train_image = cv2.imread(train_path)
            train_image_file = train_folder + os.path.basename(train_path)
            cv2.imwrite(train_image_file, train_image)
            # get anotation image and save them too
            trainannot_path = image_dir[:-1] + "annot/" + str(cls) + "/" + os.path.basename(train_path)
            trainannot_image = cv2.imread(trainannot_path)
            trainannot_image_file = trainannot_folder + os.path.basename(trainannot_path)
            cv2.imwrite(trainannot_image_file, trainannot_image)

        for valid_path in val_image_paths:
            print(valid_path)
            val_image = cv2.imread(valid_path)
            val_image_file = validation_folder + os.path.basename(valid_path)
            cv2.imwrite(val_image_file, val_image)
            # get anotation image and save them too
            valannot_path = image_dir[:-1] + "annot/" + str(cls) + "/" + os.path.basename(valid_path)
            valannot_image = cv2.imread(valannot_path)
            valannot_image_file = valannot_folder + os.path.basename(valannot_path)
            cv2.imwrite(valannot_image_file, valannot_image)


def copy_images(_from="cassava_synthetic/", _to="900/", split_type="test", split_annot_type="testannot", classes=para.num_classes):
    for cls in range(classes):
        # Now save the train images
        image_paths = glob.glob(_from + split_type + "/" + str(cls) + "/" + "*_syn.jpg") + glob.glob(
            _from + split_type + "/" + str(cls) + "/" + "*_syn.png") + glob.glob(
            _from + split_type + "/" + str(cls) + "/" + "*_syn.jpeg")

        for paths in image_paths:
            print(paths)
            _to_image_file = _to + split_type + "/" + str(cls) + "/" + os.path.basename(paths)
            image_file = _to + str(cls) + "/" + os.path.basename(paths)
            image = cv2.imread(image_file)
            cv2.imwrite(_to_image_file, image)

            # get anotation image and save them too
            annot_file = os.path.dirname(image_file) + "/" + os.path.splitext(os.path.basename(image_file))[0][
                                                                  :-4] + "_mask.png"
            _to_annot_file = _to + split_annot_type + "/" + str(cls) + "/" + os.path.basename(annot_file)

            annot_image = cv2.imread(annot_file)
            cv2.imwrite(_to_annot_file, annot_image)

if __name__ == '__main__':
    train_validation_test_splits(image_dir="900/", classes=para.num_classes, train_ratio=0.8)
    #train_val_test_real_into_class()
    #train_val_split(image_dir="cassava_real/trainval/", classes=para.num_classes, train_ratio=0.8)





