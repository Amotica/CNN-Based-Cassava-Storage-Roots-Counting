import pyrealsense2 as rs
import cv2
import numpy as np
import parameters as para
import dataset_utils
from Models import segnet_depool
from custom_metrics import *

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def depth2D_to_point3D(depth_intrinsic, raw_depth, rgb_image, depth_scale):
    rows, cols = np.array(raw_depth).shape
    rgb_image = rgb_image.reshape((rows, cols, 3))
    points3D = []
    rgb_values = []
    for r in range(rows):
        for c in range(cols):
            depth_pixel = [r, c]
            rgb_value = rgb_image[r, c] # use this for the ply file
            raw_depth_value = raw_depth[r, c]
            #print(raw_depth_value)
            if raw_depth_value > 0:
                pt3D = rs.rs2_deproject_pixel_to_point(depth_intrinsic, depth_pixel, raw_depth_value * depth_scale)
                points3D.append(pt3D)
                rgb_values.append(rgb_value)
    return np.array(points3D), np.array(rgb_values)


def read_images(rgb="captured_images/rgb_image/1.png", depth="captured_images/depth_image/1.raw"):
    # Read the raw depth and colour images
    rgb_image = cv2.imread(rgb)
    rows, cols, channel = rgb_image.shape
    #print(para.img_rows, para.img_cols)
    f = open(depth, mode='rb')
    raw_depth = np.fromfile(f, dtype=np.uint16)
    raw_depth = raw_depth.reshape((rows, cols))
    #print(np.array(raw_depth).shape)
    return rgb_image, raw_depth


def segment_raw_depth_image(segmented_rgb, raw_depth):

    segmented_gray = cv2.cvtColor(segmented_rgb, cv2.COLOR_BGR2GRAY)
    rows, cols = np.array(segmented_gray).shape

    #print(np.array(raw_depth).shape)

    for r in range(rows):
        for c in range(cols):
            segmented_gray_value = segmented_gray[r, c]  # use this for the ply file
            # print(raw_depth_value)
            if segmented_gray_value == 0:
                raw_depth[r, c] = 0

    return raw_depth


if __name__ == "__main__":
    output_rows = 0
    output_cols = 0
    if para.model_type == "segnet":
        print('Initialising SegNet...')
        model, output_cols, output_rows = segnet_depool.SegNet()
        #   =====================================

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[dice_coef, precision, recall, 'accuracy'])
    print(model.summary())
    print(para.misc_dir_eval)
    # Load the model weights
    model.load_weights(para.misc_dir_eval + "/weights.h5")

    depth_scale = 0.0010000000474974513
    depth_intrinsic = {'width': 640, 'height': 480, 'ppx': 320.156, 'ppy': 238.092, 'fx': 385.367, 'fy': 385.367, 'model': 'Brown Conrady', 'coeffs': [0, 0, 0, 0, 0]}

    # segment the rgb image using deep learning prediction and mask the segmented image from the depth image
    rgb_image, raw_depth = read_images(rgb="33.png", depth="33.raw")

    # prepare the rgb image for prediction
    rgb_image = cv2.resize(rgb_image, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
    rgb_image = np.array(rgb_image).astype('uint8')
    rgb_image = rgb_image.astype(np.float32)
    rgb_image = rgb_image / 255.0
    rgb_image = rgb_image.reshape((-1, para.img_rows, para.img_cols, 3))

    # Predict RGB Mask
    p_mask = model.predict(rgb_image)
    prediction = np.argmax(p_mask, axis=-1).reshape((para.img_rows, para.img_cols))
    predMask = dataset_utils.visualise_mask(prediction)
    predMask = np.uint8(predMask)

    #print(predMask.shape)
    #print(raw_depth.shape)

    raw_depth_segmented = segment_raw_depth_image(predMask, raw_depth)

    pts3D, rgb_values = depth2D_to_point3D(depth_intrinsic, raw_depth_segmented, rgb_image, depth_scale)  # depth and rgb should have the same dimensions
        #print(np.array(pts3D).shape)
    fn = "captured_images/1.ply"
    write_ply(fn, pts3D, rgb_values)