import cv2, os
import numpy as np
import glob
import parameters as para
from Models import segnet, segnet_lite, unet_lite, densenet, densenet_lite
import dataset_utils
from keras.preprocessing.image import img_to_array

#============================
# Parameters initialisation
#============================

contour_dist_thresh = 2000

# Ground truth images and mask folder
image_dir = para.test
annot_dir = para.testannot_rgb


def skeleton(img):

    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    # this is a skeleton with little flesh
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    skel = cv2.dilate(skel, element)

    return skel


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def closest_contour(cnt, contours, i, dist_thresh=100):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    order_box = order_points(box)
    x_bl, y_bl = order_box[3]
    small_ind = -1
    small_dist = 999999999
    for j, cnt_ in enumerate(contours):
        if j != i:
            rect_ = cv2.minAreaRect(cnt_)
            box_ = cv2.boxPoints(rect_)
            order_box_ = order_points(box_)
            x_tr, y_tr = order_box_[1]
            dist = ((x_bl - x_tr) * (x_bl - x_tr) ) + ((y_bl - y_tr) * (y_bl - y_tr))
            if dist < small_dist:
                small_dist = dist
                small_ind = j

    if small_dist < dist_thresh:
        return small_ind, small_dist
    else:
        return -1, -1


def root_count_opencv(img, predMask, img_name):
    #img = cv2.imread(img_path)
    img = cv2.resize(img, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
    predMask[np.where((predMask == [50, 100, 200]).all(axis=2))] = [255, 255, 255]
    gray = cv2.cvtColor(predMask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)

    '''Get the skeleton of the storage roots'''
    thresh = skeleton(thresh)

    '''Get the contours of the skeletal storage roots'''
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(predMask, contours, -1, (0, 0, 255), 1)

    alpha = 0.5
    added_image = cv2.addWeighted(predMask, alpha, img, 1 - alpha, 0)

    '''Filter and remove very tiny contours. they any not be complete roots'''
    # ========================================================================
    new_contours = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 200.0:
            rect = cv2.minAreaRect(cnt)
            new_contours.append(cnt)

    contours = new_contours
    # ===========================================

    '''Merge contours based of bottom left point and top right point. they any not be complete roots'''
    # ================================================================================================
    new_contours = []
    merged = [0] * len(contours)
    for i, cnt in enumerate(contours):
        indx, small_dist = closest_contour(cnt, contours, i, dist_thresh=contour_dist_thresh)
        if indx != -1:
            contours[indx] = np.vstack((contours[indx], contours[i]))
            merged[i] = 1
    # delete the merged elements
    for i in range(len(merged)):
        if merged[i] == 0:
            new_contours.append(contours[i])

    contours = new_contours
    # ===========================================

    '''Print number of storage roots and lengths of each roots'''
    # ============================================================
    new_contours = np.array(new_contours)
    font_small = cv2.FONT_HERSHEY_PLAIN
    # print("Number of storage roots: ", len(new_contours))
    start_x = 10
    start_y =25
    #start_y = np.array(predMask).shape[0] - 250
    cv2.putText(added_image, "Image: " + img_name , (start_x, start_y), font_small, 1,
                (255, 0, 255), 1, cv2.LINE_AA)
    start_y += 20
    cv2.putText(added_image, "==============" , (start_x, start_y), font_small, 1,
                (255, 0, 255), 1, cv2.LINE_AA)

    start_y += 25
    cv2.putText(added_image, "Opencv Roots count: " + str(len(new_contours)), (start_x, start_y), font_small, 1,
                (255, 0, 255), 1, cv2.LINE_AA)

    return added_image, len(new_contours), start_y


def root_count_cnn(roots_count_image, predicted_roots, gt_roots, start_y):
    # ============================================================
    # Print the number of predicted and ground truth storage roots
    # ============================================================
    font_small = cv2.FONT_HERSHEY_PLAIN
    # print("Number of storage roots: ", len(new_contours))
    start_x = 10
    start_y += 25
    cv2.putText(roots_count_image, "CNN Predicted Roots:  " + str(predicted_roots + 1), (start_x, start_y), font_small, 1,
                (255, 0, 255), 1, cv2.LINE_AA)

    start_y += 25
    cv2.putText(roots_count_image, "Ground Truth Roots:  " + str(gt_roots + 1), (start_x, start_y), font_small, 1,
                (255, 0, 255), 1, cv2.LINE_AA)

    return roots_count_image


def preprocess_image(image):
    # prepare the rgb image for prediction
    image = cv2.resize(image, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
    #image = np.array(image).astype('uint8')
    image = image.astype(np.float32)
    image = image / 255.0
    image = image.reshape((-1, para.img_rows, para.img_cols, 3))
    return image

def count_image_processor(image):
    img = []
    image = cv2.resize(image, (para.img_cols_count, para.img_rows_count), interpolation=cv2.INTER_NEAREST)
    image = img_to_array(image)
    img.append(image)
    return np.array(img)


def predicted_mask2RGB(p_mask):
    prediction = np.argmax(p_mask, axis=-1).reshape((para.img_rows, para.img_cols))
    predMask = dataset_utils.visualise_mask(prediction)
    predMask = np.uint8(predMask)
    return predMask


if __name__ == '__main__':

    # ==============================================
    # Load the Segmentation CNN Model and compile it
    # ==============================================
    output_rows = 0
    output_cols = 0

    if para.model_type == "segnet_lite":
        print('Initialising Segnet Lite...')
        model_seg, output_rows, output_cols = segnet_lite.SegNet()

    if para.model_type == "segnet":
        print('Initialising SegNet...')
        model_seg, output_rows, output_cols = segnet.SegNet()

    if para.model_type == "unet_lite":
        print('Initialising Unet Lite...')
        model_seg, output_rows, output_cols = unet_lite.unet_lite()

    model_seg.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    print("CNN segmentation model sucessfully compiled...")

    # load the segmentation  CNN model
    model_seg.load_weights(para.misc_dir_eval_seg + "/weights.h5")
    print("CNN segmentation model sucessfully compiled...")

    # =============================================
    # Load the Root Count CNN Model and compile it
    # =============================================
    if para.model_type_count == "densenet_121":
        print('Initialising Densenet 121 ...')
        model_count = densenet.DenseNet([6, 12, 24, 16])
    if para.model_type_count == "densenet_lite":
        print('Initialising Densenet Lite ...')
        model_count = densenet_lite.DenseNet([3, 6, 12, 8])

    model_count.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    print("CNN root count model sucessfully compiled...")

    # load the roots count CNN model
    model_count.load_weights(para.misc_dir_eval_count + "/weights.h5")
    print("CNN root count model weights sucessfully loaded...")

    cv2.namedWindow('Storage Root Count', cv2.WND_PROP_AUTOSIZE)

    # Open the csv file
    roots_path = para.misc_dir_eval_count + "/results"
    if not os.path.exists(roots_path):
        os.makedirs(roots_path, mode=0o777)

    print(roots_path)

    f = open(roots_path + "/" + 'roots.csv', 'w')
    f.write("image_name" + "\n")
    f.write("gt_root" + "\n")
    f.write("opencv_roots" + "\n")
    f.write("cnn_roots" + "\n")

    # loop through the class of images
    for cls in range(para.count_classes):

        image_paths = glob.glob(image_dir + str(cls) + "/" + "*.jpg") + glob.glob(image_dir + str(cls) + "/" + "*.png") + glob.glob(
            image_dir + str(cls) + "/" + "*.jpeg")
        annot_paths = glob.glob(annot_dir + str(cls) + "/" + "*.jpg") + glob.glob(annot_dir + str(cls) + "/" + "*.png") + glob.glob(
            annot_dir + str(cls) + "/" + "*.jpeg")

        for img_path, annot_path in zip(image_paths, annot_paths):

            # get the image and the GT annotation
            img = cv2.imread(img_path)
            img = cv2.resize(img, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)

            img_annot = cv2.imread(annot_path)
            img_annot = cv2.resize(img_annot, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
            #img_annot[np.where((img_annot == [50, 100, 200]).all(axis=2))] = [255, 255, 255]

            # predict the annotation from the image using the segmentation CNN
            preprocesssed_image = preprocess_image(img)
            predicted_mask = model_seg.predict(preprocesssed_image)
            predicted_mask = predicted_mask2RGB(predicted_mask)

            # get the count of storage roots from the predicted annotation using root_count_opencv
            img_name = os.path.basename(img_path)
            roots_count_image, opencv_roots, start_y = root_count_opencv(img, predicted_mask, img_name)

            # get the count of storage roots from the image using the root_count_cnn
            prep_count_image = count_image_processor(img)
            prep_count_image = prep_count_image.astype(np.float32)
            prep_count_image = prep_count_image / 255.0
            predicted_roots = model_count.predict(prep_count_image, batch_size=para.batch_size)
            predicted_roots = np.argmax(predicted_roots, axis=-1)[0]

            # annotate the predicted roots count and ground truth count on image before displaying
            roots_count_image = root_count_cnn(roots_count_image, predicted_roots, gt_roots=cls, start_y=start_y)

            # save the images with the annotated classes for opencv, gt and the cnn
            roots_image_path = para.misc_dir_eval_count + "/results/" + str(cls)
            if not os.path.exists(roots_image_path):
                os.makedirs(roots_image_path, mode=0o777)
            cv2.imwrite(roots_image_path + "/" + img_name, roots_count_image)
            print(roots_image_path)

            # show the results and wait for some time before moving to the next image. this is for visualisation
            cv2.imshow('Storage Root Count', roots_count_image)
            cv2.waitKey(1)

            # save the predictions in a csv file.
            f.write(img_name + "," + str(cls + 1) + "," + str(opencv_roots) + "," + str(predicted_roots + 1) + "\n")


    f.close()