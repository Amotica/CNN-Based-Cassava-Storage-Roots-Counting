import numpy as np
import parameters as para
import cv2, glob, os


contour_dist_thresh = 2000
c_size = 200.0

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


def get_unique_paths(left_paths, right_paths, ln, rn):
    left_paths = np.random.choice(left_paths, ln)
    right_paths = np.random.choice(right_paths, rn)
    return left_paths, right_paths


def get_unique_path(paths, ln=1):
    left_paths = np.random.choice(paths, ln)
    return left_paths


def get_random_roots(cls):
    clss = int(cls+1)
    a = range(0, clss)
    num = np.random.choice(a, 1)
    return num[0]


def translate_image(image, x, y, img_cols, img_rows):
    M = np.float32([[1, 0, x], [0, 1, y]])
    dst = cv2.warpAffine(image, M, (img_cols, img_rows))
    return dst

def translate_image_folder(image, x, y, img_cols, img_rows, folder):
    y_coeff = [140, 150, 160, 170, 160, 150, 140]
    x = (170 + ((folder + 1) * 20)) - x
    y = y_coeff[folder] - y
    M = np.float32([[1, 0, x], [0, 1, y]])
    dst = cv2.warpAffine(image, M, (img_cols, img_rows))
    return dst


def rotate_mask_center(img, img_cols, img_rows, w):
    if w >= 200:
        ang=20
    elif w >= 180:
        ang = 15
    elif w >= 150:
        ang = 10
    elif w >= 120:
        ang = 5
    else:
        ang = 0

    M = cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), ang, 1)
    dst = cv2.warpAffine(img, M, (img_cols, img_rows))
    return dst


def rotate_mask_anti_clockwise(img, img_cols, img_rows, w):
    if w >= 200:
        ang=15
    elif w >= 180:
        ang = 10
    elif w >= 150:
        ang = 5
    elif w >= 120:
        ang = 0
    elif w >= 90:
        ang = -5
    elif w >= 60:
        ang = -10
    else:
        ang = -15

    M = cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), ang, 1)
    dst = cv2.warpAffine(img, M, (img_cols, img_rows))
    return dst


def rotate_mask_clockwise(img, img_cols, img_rows, w):
    if w >= 200:
        ang=-15
    elif w >= 180:
        ang = -10
    elif w >= 150:
        ang = -5
    elif w >= 120:
        ang = 0
    elif w >= 90:
        ang = 5
    elif w >= 60:
        ang = 10
    else:
        ang = 15

    M = cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), ang, 1)
    dst = cv2.warpAffine(img, M, (img_cols, img_rows))
    return dst


def get_contour_extremes(contours):
    #extLeft = tuple(contours[contours[:, :, 0].argmin()][0])
    #extRight = tuple(contours[contours[:, :, 0].argmax()][0])
    extTop = tuple(contours[contours[:, :, 1].argmin()][0])
    extBot = tuple(contours[contours[:, :, 1].argmax()][0])
    return extTop, extBot


def get_folder(contours):
    x, y, w, h = cv2.boundingRect(contours)
    start = int(para.img_cols/10)
    if int(x) <= start*2:
        return 0, w
    elif int(x) <= start*3:
        return 1, w
    elif int(x) <= start*4:
        return 2, w
    elif int(x) <= start*5:
        return 3, w
    elif int(x) <= start*6:
        return 4, w
    elif int(x) <= start*7:
        return 5, w
    else:
        return 6, w


def get_contour(msk):
    msk[np.where((msk == [50, 100, 200]).all(axis=2))] = [255, 255, 255]
    gray = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def additional_single_root(blank_msk, folder=0, id=0):

    msk_contour = get_contour(blank_msk)
    for cont in msk_contour:
        extTop, extBot = get_contour_extremes(cont)
        x = extTop[0]
        y = extTop[1]
        blank_msk = translate_image_folder(blank_msk, x, y, para.img_cols, para.img_rows, folder)
        blank_msk[np.where((blank_msk == [255, 255, 255]).all(axis=2))] = [50, 100, 200]

        print("Generating image: ", "'cassava_gan/single_masks/temp/" + str(id) + ".png'")
        cv2.imwrite("cassava_gan/single_masks/temp/" + str(id) + ".png", blank_msk)
        cv2.imshow("additional mask", blank_msk)
        cv2.waitKey(1000)




def single_root_mask(train_dir=para.trainvalannot):
    mask_paths = glob.glob(train_dir + "0/" + "*.jpg") + glob.glob(train_dir + "0/" + "*.png") + glob.glob(
        train_dir + "0/" + "*.jpeg")

    c_num = 0
    l_num = 0
    r_num = 0
    f = open("cassava_gan/single_masks/" + 'widths.csv', 'w')
    for mask_path in mask_paths:
        msk = cv2.imread(mask_path)
        msk = cv2.resize(msk, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)

        msk[np.where((msk == [50, 100, 200]).all(axis=2))] = [255, 255, 255]

        gray = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)

        #element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        #thresh = cv2.erode(thresh, element)

        '''Get the contours of the skeletal storage roots'''
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        '''Filter and remove very tiny contours. they any not be complete roots'''
        # ========================================================================
        new_contours = []
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > c_size:
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

        for i, cnt in enumerate(contours):
            blank_msk = np.zeros((para.img_rows, para.img_cols, 3), np.uint8)
            cv2.drawContours(blank_msk, [cnt], -1, (50, 100, 200), -1)

            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            if rect[2] < -45.0:
                angle += 90

            if int(angle) < 0:
                # save as right - this is negative
                folder, w = get_folder(cnt)
                f.write(str(w) + "\n")

                # rotate mask as required. this depends on the w value
                blank_msk = rotate_mask_clockwise(blank_msk, para.img_cols, para.img_rows, w)

                # Translate the image to the correct x and y postions based on the top extreme value
                extTop, extBot = get_contour_extremes(cnt)
                x = extTop[0]
                y = extTop[1]
                blank_msk = translate_image_folder(blank_msk, x, y, para.img_cols, para.img_rows, folder)

                print("Generating image: ", "'cassava_gan/single_masks/right/" + str(folder) + "/" + str(r_num) + ".png'" + " - ", w)
                cv2.imwrite("cassava_gan/single_masks/right/" + str(folder) + "/" + str(r_num) + ".png", blank_msk)
                r_num += 1

            elif int(angle) > 0:
                # save as left - this is positive
                folder, w = get_folder(cnt)
                f.write(str(w) + "\n")

                # rotate mask as required. this depends on the w value
                blank_msk = rotate_mask_anti_clockwise(blank_msk, para.img_cols, para.img_rows, w)

                # Translate the image to the correct x and y postions based on the top extreme value
                extTop, extBot = get_contour_extremes(cnt)
                x = extTop[0]
                y = extTop[1]
                blank_msk = translate_image_folder(blank_msk, x, y, para.img_cols, para.img_rows, folder)

                print("Generating image: ", "'cassava_gan/single_masks/left/" + str(folder) + "/" + str(l_num) + ".png'" + " - ", w)
                cv2.imwrite("cassava_gan/single_masks/left/" + str(folder) + "/" + str(l_num) + ".png", blank_msk)
                l_num += 1
            else:
                # save as center - this is zero
                folder, w = get_folder(cnt)
                f.write(str(w) + "\n")

                # rotate mask as required. this depends on the w value
                blank_msk = rotate_mask_center(blank_msk, para.img_cols, para.img_rows, w)

                # Translate the image to the correct x and y postions based on the top extreme value
                extTop, extBot = get_contour_extremes(cnt)
                x = extTop[0]
                y = extTop[1]
                blank_msk = translate_image_folder(blank_msk, x, y, para.img_cols, para.img_rows, folder)

                print("Generating image: ", "'cassava_gan/single_masks/center/" + str(folder) + "/" + str(c_num) + ".png'" + " - ", w)
                cv2.imwrite("cassava_gan/single_masks/center/" + str(folder) + "/" + str(c_num) + ".png", blank_msk)
                c_num += 1
    f.close()


def entire_root_mask(single_msk_path="cassava_gan/single_masks/", total_roots=7, samples=500, cls=5):

    for i in range(samples - 1):
        left_roots = get_random_roots(cls)
        center_roots = get_random_roots(cls-left_roots)
        right_roots = cls-(left_roots+center_roots)

        # get the starting folder this should be random
        start_folder = get_random_roots(total_roots-cls)

        left_paths_ = []
        for lft in range(left_roots):
            paths = glob.glob(single_msk_path + "left/" + str(start_folder) + "/" + "*.jpg") + glob.glob(
                single_msk_path + "left/" + str(start_folder) + "/" + "*.png") + glob.glob(
                single_msk_path + "left/" + str(start_folder) + "/" + "*.jpeg")
            #get unique path from the selected left folder
            left_paths_.append(get_unique_path(paths, ln=1)[0])
            start_folder += 1

        center_paths_ = []
        for cen in range(center_roots):
            paths = glob.glob(single_msk_path + "center/" + str(start_folder) + "/" + "*.jpg") + glob.glob(
                single_msk_path + "center/" + str(start_folder) + "/" + "*.png") + glob.glob(
                single_msk_path + "center/" + str(start_folder) + "/" + "*.jpeg")
            # get unique path from the selected center folder
            center_paths_.append(get_unique_path(paths, ln=1)[0])
            start_folder += 1

        right_paths_ = []
        for rgt in range(right_roots):
            paths = glob.glob(single_msk_path + "right/" + str(start_folder) + "/" + "*.jpg") + glob.glob(
                single_msk_path + "right/" + str(start_folder) + "/" + "*.png") + glob.glob(
                single_msk_path + "right/" + str(start_folder) + "/" + "*.jpeg")
            # get unique path from the selected right folder
            right_paths_.append(get_unique_path(paths, ln=1)[0])
            start_folder += 1

        #x_pos = 50
        #y_pos = 25
        # create new blank image
        synthetic_mask = np.zeros((para.img_rows, para.img_cols, 3), np.uint8)

        for lpaths in left_paths_:
            msk_left = cv2.imread(lpaths)
            msk_left = cv2.resize(msk_left, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
            msk_left[np.where((msk_left == [50, 100, 200]).all(axis=2))] = [255, 255, 255]
            gray = cv2.cvtColor(msk_left, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #x_p, y_p, w, h = get_countour_pos(contours[0], x_pos, y_pos)
            #x_p = 0
            # Translate the image to correct position
            #msk_left = translate_image(msk_left, x_p, y_p, para.img_cols, para.img_rows)
            gray = cv2.cvtColor(msk_left, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(synthetic_mask, contours, -1, (50, 100, 200), -1)

        for rpaths in right_paths_:
            msk_right = cv2.imread(rpaths)
            msk_right = cv2.resize(msk_right, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
            msk_right[np.where((msk_right == [50, 100, 200]).all(axis=2))] = [255, 255, 255]
            gray = cv2.cvtColor(msk_right, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #x_p, y_p, w, h = get_countour_pos(contours[0], x_pos, y_pos)
            #x_p = 0
            #msk_right = translate_image(msk_right, x_p, y_p, para.img_cols, para.img_rows)
            gray = cv2.cvtColor(msk_right, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #print(rpaths)
            cv2.drawContours(synthetic_mask, contours, -1, (50, 100, 200), -1)

        for cpaths in center_paths_:
            msk_center = cv2.imread(cpaths)
            msk_center = cv2.resize(msk_center, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
            msk_center[np.where((msk_center == [50, 100, 200]).all(axis=2))] = [255, 255, 255]
            gray = cv2.cvtColor(msk_center, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #x_p, y_p, w, h = get_countour_pos(contours[0], x_pos, y_pos)
            #x_p = 0
            #msk_center = translate_image(msk_center, x_p, y_p, para.img_cols, para.img_rows)
            gray = cv2.cvtColor(msk_center, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #print(cpaths)
            cv2.drawContours(synthetic_mask, contours, -1, (50, 100, 200), -1)

        path_to_save = para.home_dir + para.dataset + "/synthetic_train/" + str(cls - 1)
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save, mode=0o777)

        print("Saving Sample ", i)
        file_to_save = path_to_save + "/" + str(i) + "_syn.png"
        cv2.imwrite(file_to_save, synthetic_mask)
        #cv2.imshow("synthetic_mask2", synthetic_mask)
        #cv2.waitKey(500)


if __name__ == '__main__':
    #single_root_mask()

    entire_root_mask(samples=200, cls=6)

    '''
    single_msk_path = "cassava_gan/single_masks/"
    start_folder = 3
    paths = glob.glob(single_msk_path + "left/" + str(start_folder) + "/" + "*.jpg") + glob.glob(
        single_msk_path + "left/" + str(start_folder) + "/" + "*.png") + glob.glob(
        single_msk_path + "left/" + str(start_folder) + "/" + "*.jpeg")
    
    for i, pths in enumerate(paths):
        blank_msk = cv2.imread(pths)
        additional_single_root(blank_msk, folder=6, id=i)
    '''
