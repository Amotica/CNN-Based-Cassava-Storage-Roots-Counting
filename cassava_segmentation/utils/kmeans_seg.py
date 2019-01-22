import numpy as np
import cv2, os
import glob

background_colour = [0, 0, 0]
fibrous_colour = [0, 255, 255]
storage_roots = [50, 100, 200]
pencil_roots = [50, 100, 200]


def closet_color_to_black(stack_of_centers):
    '''find the closest colour to black from the K-Means labels'''
    closest = float("inf")
    closest_index = 0
    for i, labels in enumerate(stack_of_centers):
        diff = sum([abs(component1-component2) for component1, component2 in zip(background_colour, labels)])
        if closest > diff:
            closest_index = i
            closest = diff
    '''change center colours: closest to background (black) and others to foreground (root - yellow)'''
    for i, labels in enumerate(stack_of_centers):
        if i == closest_index:
            stack_of_centers[i] = background_colour
        else:
            stack_of_centers[i] = fibrous_colour
    return stack_of_centers


def kmeans_segment_roots(clusters=2,  data_path="data/imgs/", seg_data_path="data/kmeans/"):
    '''Define K Means Criteria and the number of clusters K'''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = clusters
    '''Define the data folders'''
    data_path = data_path
    seg_data_path = seg_data_path

    images = os.listdir(data_path)
    for img in images:
        print("Segmenting root", img, 'into', K, 'clusters')
        img_file = os.path.join(data_path, img)
        seg_image_file = os.path.join(seg_data_path, img)
        '''read image to segment'''
        img = cv2.imread(img_file)
        Z = img.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        '''perform the k-mean clustering clustering roots and background'''
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        ''''''
        center = closet_color_to_black(center)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape(img.shape)
        '''Save the segmented images'''
        cv2.imwrite(seg_image_file, res2)

    print("Segmentation of roots completed successfully")


def kmeans_segment_roots_3(clusters=2, data_path="data/imgs/", seg_data_path="data/kmeans/"):
    '''Define K Means Criteria and the number of clusters K'''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = clusters
    '''Define the data folders'''
    data_path = data_path
    seg_data_path = seg_data_path

    images = os.listdir(data_path)
    for img in images:
        print("Segmenting root", img, 'into', K, 'clusters')
        img_file = os.path.join(data_path, img)
        seg_image_file = os.path.join(seg_data_path, img)
        '''read image to segment'''
        img = cv2.imread(img_file)
        Z = img.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        '''perform the k-mean clustering clustering roots and background'''
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        ''''''
        #center = closet_color_to_black(center)
        print(center)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape(img.shape)
        '''Save the segmented images'''
        cv2.imwrite(seg_image_file, res2)

    print("Segmentation of roots completed successfully")


def colour_threshold(data_path="data/imgs/", seg_data_path="data/kmeans/"):
    '''Define the data folders'''
    data_path = data_path
    seg_data_path = seg_data_path

    lower_color_bounds = np.array([82, 97, 119])
    upper_color_bounds = np.array([5, 10, 15])

    images = os.listdir(data_path)
    for img in images:
        print("Segmenting root", img)
        img_file = os.path.join(data_path, img)
        seg_image_file = os.path.join(seg_data_path, img)
        '''read image to segment'''
        img = cv2.imread(img_file)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray_img, 0, 255, cv.THRESH_BINARY)

        # print(np.unique(image, axis=2))
        unique_pixels = np.vstack({tuple(r) for r in gray_img.reshape(-1, 1)})
        print(unique_pixels)



        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(img, lower_color_bounds, upper_color_bounds)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img = img & mask_rgb
        cv2.imshow("image", img)
        cv2.waitKey(100)


    print("Segmentation of roots completed successfully")


def annotate_storage_roots(seg_data_path="data/kmeans/", gt_path="data/annotations/", final_anno_path="data/annotation_final/"):
    '''img_roots and img_storage are of the same dimensions'''
    seg_path = seg_data_path
    gt_seg_path = gt_path
    images = os.listdir(seg_path)
    for k, img in enumerate(images):
        print(img)
        seg_file = os.path.join(seg_path, img)
        gt_file = os.path.join(gt_seg_path, img)
        final_anno_file = os.path.join(final_anno_path, img)
        seg_img = cv2.imread(seg_file)
        gt_img = cv2.imread(gt_file)
        #   storage roots
        seg_img[np.where((gt_img == storage_roots).all(axis=2))] = storage_roots
        #   Pencil roots
        #seg_img[np.where((gt_img == pencil_roots).all(axis=2))] = pencil_roots
        cv2.imwrite(final_anno_file, seg_img)
    print('annotating ground truth completed successfully')
    return True


def change_colour_in_image(seg_data_path="../cassava_fore_back/valannot/0/", gt_path="data/annotations_original/", final_anno_path="../cassava_fore_back/valannot_final/0/"):
    '''img_roots and img_storage are of the same dimensions'''
    change_colour = [0, 255, 255]
    #change_colour = [50, 100, 200]
    seg_path = seg_data_path
    #gt_seg_path = gt_path
    images = os.listdir(seg_path)
    print(images)
    for k, img in enumerate(images):
        print(img)
        seg_file = os.path.join(seg_path, img)
        #gt_file = os.path.join(gt_seg_path, img)
        final_anno_file = os.path.join(final_anno_path, img)
        seg_img = cv2.imread(seg_file)
        #gt_img = cv2.imread(gt_file)
        #   storage roots
        seg_img[np.where((seg_img == change_colour).all(axis=2))] = storage_roots
        cv2.imwrite(final_anno_file, seg_img)
    print('Chaning color: ', change_colour, 'completed successfully')
    return True


def get_storage_roots_from_mask(seg_data_path="data/segments/", roots_path="data/images/", final_storage_root_path="data/cassava_storage/"):
    seg_path = seg_data_path
    cassava_roots_path = roots_path
    images = os.listdir(cassava_roots_path)
    for k, s_roots in enumerate(images):
        s_roots_file = os.path.join(cassava_roots_path, s_roots)
        print(s_roots_file)
        seg_file = os.path.join(seg_path, s_roots)
        seg_file = seg_file[:-4] + ".png"
        print(seg_file)
        s_roots_img = cv2.imread(s_roots_file)
        seg_img = cv2.imread(seg_file, 0)
        print(len(seg_img))
        storage_root = cv2.bitwise_and(s_roots_img, s_roots_img, mask=seg_img)
        # Save the storage roots
        final_filename = os.path.join(final_storage_root_path, s_roots)
        cv2.imwrite(final_filename, storage_root)


def closetRGB(rgb_pixels):
    closest = False
    colourset = [0, 0, 0]
    d = np.sqrt(
        ((colourset[0] - rgb_pixels[0]) * (colourset[0] - rgb_pixels[0]))
        + ((colourset[1] - rgb_pixels[1]) * (colourset[0] - rgb_pixels[0]))
        + ((colourset[2] - rgb_pixels[2]) * (colourset[2] - rgb_pixels[2]))
    )

    print(d)
    #if d < [0]:
       #closest = True

    return closest


def replace_closest_2_black(final_storage_root_path="data/cassava_storage/"):
    image_paths = glob.glob(final_storage_root_path + "*.jpg") + glob.glob(final_storage_root_path + "*.png") + glob.glob(
        final_storage_root_path + "*.jpeg")

    for imgs in image_paths:
        image = cv2.imread(imgs)

        for rgb_pixels in image:
            res = closetRGB(rgb_pixels)


        #seg_image = cv2.imread(seg_data_path + base_finename)
        #finale_mask = np.zeros((seg_image.shape[0], seg_image.shape[1], 3))

        #finale_mask[np.where((seg_image == [255, 255, 255]).all(axis=2))] = [128, 128, 128]


        #pred_mask[np.where((pred_mask == [1, 1, 1]).all(axis=2))] = [128, 0, 0]
        #pred_mask[np.where((pred_mask == [2, 2, 2]).all(axis=2))] = [192, 192, 128]


if __name__ == "__main__":
    #image = cv2.imread("cassava/trainannot/411.png")
    #unique_pixels = np.vstack({tuple(r) for r in image.reshape(-1, 3)})
    #print(unique_pixels)

    '''0. OPTIONAL: change the storage colour to desire colour if biologist label them a different colour'''
    #change_colour_in_image(seg_data_path="../data2/Photo 6 Mayo 10 harvest/harvest camera 2/Output_Images/", final_anno_path="../data2/Photo 6 Mayo 10 harvest/harvest camera 2/Output_Images_final/")
    '''1. Segment background using k-means'''
    #kmeans_segment_roots_3(clusters=2, data_path="../data2/Photo 6 Mayo 10 harvest/harvest camera 2/", seg_data_path="../data2/Photo 6 Mayo 10 harvest/harvest camera 2/kmeans/")
    '''2. final k-means - this replaces root with yellow and background with black.'''
    #kmeans_segment_roots(clusters=2,  data_path="../cassava_fore_back/all_data/train_val_kmeans/", seg_data_path="../cassava_fore_back/all_data/train_val_kmeans_final/")
    '''3. Append the Ground truth annotation to the kmeans segmented data and store results in the folder "anno_gt" '''
    annotate_storage_roots(seg_data_path="../cassava_fore_back/all_data/train_val_kmeans/",
                           gt_path="../cassava_fore_back/all_data/train_valannot/0/",
                           final_anno_path="../cassava_fore_back/all_data/train_val_final/")

    #get_storage_roots_from_mask(seg_data_path="data/Output_Images/", roots_path="data/NIkon_D7200/",
                                #final_storage_root_path="data/final_images/")

    #replace_closest_2_black(final_storage_root_path="data/final_images/")

