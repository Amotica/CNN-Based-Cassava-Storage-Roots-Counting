import cv2
import numpy as np
import glob
import parameters as para


#============================
# Parameters initialisation
#============================

contour_dist_thresh = 2000

# Ground truth images and mask folder
image_dir = "./cassava_fore_back_640x480/test/"
annot_dir = para.misc_dir + "/annot/"


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


if __name__ == '__main__':


    image_paths = glob.glob(image_dir + "0/" + "*.jpg") + glob.glob(image_dir + "0/" + "*.png") + glob.glob(
        image_dir + "0/" + "*.jpeg")
    annot_paths = glob.glob(annot_dir + "0/" + "*.jpg") + glob.glob(annot_dir + "0/" + "*.png") + glob.glob(
        annot_dir + "0/" + "*.jpeg")
    #print(annot_paths)
    for img_path, annot_path in zip(image_paths, annot_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
        #print(img_path)
        print(annot_path)
        img_annot = cv2.imread(annot_path)
        img_annot[np.where((img_annot == [50, 100, 200]).all(axis=2))] = [255, 255, 255]

        #element = cv2.getStructuringElement(cv2.MORPH_CROSS, (11, 11))
        #img_annot = cv2.dilate(img_annot, element)

        gray = cv2.cvtColor(img_annot, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)

        '''Get the skeleton of the storage roots'''
        thresh = skeleton(thresh)

        '''Get the contours of the skeletal storage roots'''
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_annot, contours, -1, (0, 0, 255), 1)

        #print(img_annot.shape)
        #print(img.shape)
        alpha = 0.5
        added_image = cv2.addWeighted(img_annot, alpha, img, 1 - alpha, 0)

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
        #print("Number of storage roots: ", len(new_contours))
        start_x = 10
        start_y = np.array(img_annot).shape[0] - 170
        cv2.putText(added_image, "# of Storage Roots = " + str(len(new_contours)), (start_x, start_y), font_small, 1, (255, 0, 255), 1, cv2.LINE_AA)
        # Print the lengths of storage roots in pixels
        for i, cnt in enumerate(new_contours):
            rect = cv2.minAreaRect(cnt)

            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            #cv2.drawContours(added_image, [box], 0, (0, 0, 255), 2)

            start_y += 15
            perimeter = cv2.arcLength(cnt, True)
            root_len = perimeter/2
            # get centriod of countour
            x, y = np.int0(rect[0])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(added_image, str(i), (int(x-10), int(y)), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(added_image, "Length of Root #" + str(i) + " = " + str(int(round(root_len))) + " Pixels", (start_x, start_y), font_small, 1,
                        (255, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow('edge2', added_image)
        cv2.waitKey(5000)
