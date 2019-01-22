import cv2, os
import numpy as np
import glob
import parameters as para
from Models import segnet_lite, densenet_lite, densenet_lite_type
import dataset_utils
from keras.preprocessing.image import img_to_array
import datetime

#============================
# Parameters initialisation
#============================

contour_dist_thresh = 2000

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


def opencv_roots(img, predMask):
    # img = cv2.imread(img_path)
    #img = cv2.resize(img, (para.img_cols_seg_full, para.img_rows_seg_full), interpolation=cv2.INTER_NEAREST)
    predMask = cv2.resize(predMask, (para.img_cols_seg_full, para.img_rows_seg_full), interpolation=cv2.INTER_NEAREST)
    predMask[np.where((predMask == [50, 100, 200]).all(axis=2))] = [255, 255, 255]
    gray = cv2.cvtColor(predMask, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray, 127, 255, 0)

    '''Get the skeleton of the storage roots'''
    thresh = skeleton(gray)

    '''Get the contours of the skeletal storage roots'''
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(predMask, contours, -1, (0, 0, 255), 1)

    #print(predMask.shape)
    #print(img.shape)

    alpha = 0.4
    #added_image = cv2.addWeighted(img, alpha, predMask, 1 - alpha, 0)
    added_image = cv2.addWeighted(img, 1, predMask, alpha, 0)

    '''Filter and remove very tiny contours. they any not be complete roots'''
    # ========================================================================
    new_contours = []
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 200.0:
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

    return len(new_contours), added_image



def stats_on_root_image(image, stats, test):
    # ============================================================
    # Print the number of predicted and ground truth storage roots
    # ============================================================
    font_small = cv2.FONT_HERSHEY_PLAIN
    # print("Number of storage roots: ", len(new_contours))
    start_x = 10
    start_y = 170
    cv2.putText(image, "Image:  " + str(stats[0]), (start_x, start_y), font_small, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if test == "age":
        start_y += 35
        cv2.putText(image, "Actual Age:  " + str(stats[1]), (start_x, start_y), font_small, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)

        start_y += 25
        cv2.putText(image, "Predicted Age:  " + str(stats[2]), (start_x, start_y), font_small, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)
    elif test == "count":
        start_y += 35
        cv2.putText(image, "GT Root Count:  " + str(stats[3]), (start_x, start_y), font_small, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)

        start_y += 25
        cv2.putText(image, "Pred. Root Count:  " + str(stats[4]), (start_x, start_y), font_small, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)

        start_y += 25
        cv2.putText(image, "Seg. Root Count:  " + str(stats[7]), (start_x, start_y), font_small, 1, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return image


def seg_image_preprocessor(image):
    # prepare the rgb image for prediction
    image = cv2.resize(image, (para.img_cols_seg, para.img_rows_seg), interpolation=cv2.INTER_NEAREST)
    image = image.astype(np.float32)
    image = image / 255.0
    image = image.reshape((-1, para.img_rows_seg, para.img_cols_seg, 3))
    return image


def image_processor(image):
    img = []
    image = cv2.resize(image, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
    image = img_to_array(image)
    img.append(image)
    return np.array(img)


def predicted_mask2RGB(p_mask):
    prediction = np.argmax(p_mask, axis=-1).reshape((para.img_rows_seg, para.img_cols_seg))
    predMask = dataset_utils.visualise_mask(prediction)
    predMask = np.uint8(predMask)
    return predMask


#home_dir = "/home/pszjka/CNN/cassava_seg/"
home_dir = ""
seg_model_dir = home_dir + "misc/segnet_lite/"
count_model_dir = home_dir + "misc/densenet_lite/"
type_model_dir = home_dir + "misc/densenet_lite_type/"
# Ground truth images and mask folder
image_dir = home_dir + "test_data/test/"
annot_dir = home_dir + "test_data/testannot/"

test = "count" # age / count

misc_dir = home_dir + "misc/"

pred_cassava_type = ""
actual_cassava_type = ""
actual_roots = 0
top_1 = 0
top_2 = 0
top_3 = 0


if __name__ == '__main__':

    # ==============================================
    # Load the Segmentation CNN Model and compile it
    # ==============================================
    output_rows = 0
    output_cols = 0

    start_time_load_models = datetime.datetime.now()

    # load the segmentation  CNN model for older plants
    model_seg_old, output_rows, output_cols = segnet_lite.SegNet()
    model_seg_old.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model_seg_old.load_weights(seg_model_dir + "old_cassava_weights.h5")
    print("CNN segmentation model weights for older plants successfully loaded...")

    # load the segmentation  CNN model for younger plants
    model_seg_young, output_rows, output_cols = segnet_lite.SegNet()
    model_seg_young.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model_seg_young.load_weights(seg_model_dir + "young_cassava_weights.h5")
    print("CNN segmentation model weights for younger plants successfully loaded...")

    # =============================================
    # Load the Root type CNN Model and compile it
    # =============================================
    # load the root type CNN model for all plants
    model_root_type = densenet_lite_type.DenseNet([3, 6, 12, 8])
    model_root_type.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model_root_type.load_weights(type_model_dir + "weights.h5")
    print("CNN root type model weights for older plants successfully loaded...")

    # =============================================
    # Load the Root Count CNN Model and compile it
    # =============================================
    # load the root count CNN model for older plants
    model_count_old = densenet_lite.DenseNet([3, 6, 12, 8], classes=para.count_classes_old)
    model_count_old.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model_count_old.load_weights(count_model_dir + "old_cassava_weights.h5")
    print("CNN root count model weights for older plants successfully loaded...")

    # load the root count CNN model for younger plants
    model_count_young = densenet_lite.DenseNet([3, 6, 12, 8], classes=para.count_classes_young)
    model_count_young.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    model_count_young.load_weights(count_model_dir + "young_cassava_weights.h5")
    print("CNN root count model weights for younger plants successfully loaded...")

    elapsed_time_load_model = datetime.datetime.now() - start_time_load_models
    print("5 CNN models loaded loaded successfully in ", elapsed_time_load_model)


    # Open the csv file
    roots_path = misc_dir + "results"
    if not os.path.exists(roots_path):
        os.makedirs(roots_path, mode=0o777)

    f = open(roots_path + "/" + 'roots.csv', 'w')
    f.write("image_name" + "," + "actual_type" + "," + "predicted_type" + "," + "actual_roots" + "," + "top_1" + "," + "top_2" + "," + "top_3" + "," + "opencv_roots" + "\n")

    for t_cls in range(para.type_classes):

        if t_cls == 0:
            actual_cassava_type = ">= 2.5 Months"
        else:
            actual_cassava_type = "< 2.5 Months"

        for cls in range(para.count_classes_young):

            image_paths = glob.glob(image_dir + str(t_cls) + "/" + str(cls) + "/" + "*.jpg") + glob.glob(
                image_dir + str(t_cls) + "/" + str(cls) + "/" + "*.png") + glob.glob(
                image_dir + str(t_cls) + "/" + str(cls) + "/" + "*.jpeg")

            for img_path in image_paths:

                # get the image and the GT annotation
                img = cv2.imread(img_path)
                img = cv2.resize(img, (para.img_cols_seg_full, para.img_rows_seg_full),
                                      interpolation=cv2.INTER_NEAREST)

                # pre-processed image for segmentation CNN
                prep_seg_image = seg_image_preprocessor(img)

                # pre-processed image for the type and count CNN models
                processed_image = image_processor(img)
                processed_image = processed_image.astype(np.float32)
                processed_image = processed_image / 255.0

                # Get the cassava plant type 0 = old and 1 = young plant
                predicted_type = model_root_type.predict(processed_image, batch_size=para.batch_size)
                predicted_type = np.argmax(predicted_type, axis=-1)[0]

                if predicted_type == 0:
                    pred_cassava_type = ">= 2.5 Months"

                    # predict the storage root mask using the root segementation CNN model
                    predicted_mask = model_seg_old.predict(prep_seg_image)
                    predicted_mask = predicted_mask2RGB(predicted_mask)

                    opencv_storage_roots, added_image = opencv_roots(img, predicted_mask)

                    # Predict the number of storage roots using the root count CNN model
                    predicted_roots = model_count_old.predict(processed_image, batch_size=para.batch_size)
                    top_3_idx = np.argsort(predicted_roots[0])[-3:]
                    # top_3_values = [round(predicted_roots[i] * 100) for i in top_3_idx]
                    # for the old increase the count by 1. This is because the 0 class belongs to 1 storage root,
                    # the 1 class = 2 storage roots
                    actual_roots = cls + 1
                    top_1 = top_3_idx[2] + 1
                    top_2 = top_3_idx[1] + 1
                    top_3 = top_3_idx[0] + 1

                else:
                    pred_cassava_type = "< 2.5 Months"

                    # predict the storage root mask using the root segementation CNN model
                    predicted_mask = model_seg_young.predict(prep_seg_image)
                    predicted_mask = predicted_mask2RGB(predicted_mask)

                    opencv_storage_roots, added_image = opencv_roots(img, predicted_mask)

                    # Predict the number of storage roots using the root count CNN model
                    predicted_roots = model_count_young.predict(processed_image, batch_size=para.batch_size)
                    top_3_idx = np.argsort(predicted_roots[0])[-3:]
                    # top_3_values = [round(predicted_roots[i] * 100) for i in top_3_idx]
                    actual_roots = cls
                    top_1 = top_3_idx[2]
                    top_2 = top_3_idx[1]
                    top_3 = top_3_idx[0]

                print("Image: ", os.path.basename(img_path) )

                print("Actual Age: ", actual_cassava_type)
                print("Predicted Age: ", pred_cassava_type)


                print("Ground Truth Root Count: ", actual_roots)
                print("Predicted Root Count: ", top_1)
                #print("Top 2 CNN Roots: ", top_2)
                #print("Top 3 CNN Roots: ", top_3)
                print("Segmentation Root Count: ", opencv_storage_roots)
                print("\n\n")

                stats = []
                stats.append(os.path.basename(img_path))
                stats.append(actual_cassava_type)
                stats.append(pred_cassava_type)
                stats.append(actual_roots)
                stats.append(top_1)
                stats.append(top_2)
                stats.append(top_3)
                stats.append(opencv_storage_roots)

                f.write(
                    stats[0] + "," + stats[1] + "," + stats[2] + "," + str(stats[3]) + "," + str(stats[4]) + "," + str(
                        stats[5]) + "," + str(stats[6]) + "," + str(stats[7]) + "\n")

                added_image = stats_on_root_image(added_image, stats, test=test)

                # save the images with the annotated classes for opencv, gt and the cnn
                roots_image_path = roots_path + "/" + str(t_cls) + "/" + str(cls)
                if not os.path.exists(roots_image_path):
                    os.makedirs(roots_image_path, mode=0o777)
                cv2.imwrite(roots_image_path + "/" + os.path.basename(img_path), added_image)

                cv2.imshow("image", added_image)
                cv2.waitKey(100)

    f.close()