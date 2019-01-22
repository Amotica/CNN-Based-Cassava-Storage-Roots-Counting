import numpy as np
import cv2, os
import glob

background_colour = [0, 0, 0]
fibrous_colour = [0, 255, 255]
storage_roots = [50, 100, 200]
pencil_roots = [50, 100, 200]

change_colour = [0, 255, 255]


def change_colour_in_image(seg_data_path="cassava_fore_back/trainannot/0/", final_anno_path="cassava_fore_back/trainannot_final/"):
    '''img_roots and img_storage are of the same dimensions'''
    seg_path = seg_data_path
    images = os.listdir(seg_path)
    for k, img in enumerate(images):
        print(img)
        seg_file = os.path.join(seg_path, img)
        final_anno_file = os.path.join(final_anno_path, img)
        seg_img = cv2.imread(seg_file)
        #   storage roots
        seg_img[np.where((seg_img == change_colour).all(axis=2))] = storage_roots
        cv2.imwrite(final_anno_file, seg_img)
    print('Chaning color: ', change_colour, 'completed successfully')
    return True


if __name__ == "__main__":
    change_colour_in_image(seg_data_path="../cassava_fore_back/trainannot/0/", final_anno_path="../cassava_fore_back/trainannot_final/")