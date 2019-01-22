import parameters as para
from Models import segnet_depool
import dataset_utils
from custom_metrics import *

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import pickle


def image_file_counter(path):
    files = 0
    for _, _, filenames in os.walk(path):
        files += len(filenames)
    return files + 1


def spatial_filtering(depth_frame, magnitude=2, alpha=0.5, delta=20, holes_fill=0):
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, magnitude)
    spatial.set_option(rs.option.filter_smooth_alpha, alpha)
    spatial.set_option(rs.option.filter_smooth_delta, delta)
    spatial.set_option(rs.option.holes_fill, holes_fill)
    depth_frame = spatial.process(depth_frame)
    return depth_frame


def hole_filling(depth_frame):
    hole_filling = rs.hole_filling_filter()
    depth_frame = hole_filling.process(depth_frame)
    return depth_frame

# define global variables
# ========================
# file names and paths
rgb_img_path = 'captured_images/rgb_image/'
depth_img_path = 'captured_images/depth_image/'
colored_depth_img_path = 'captured_images/coloured_depth_image/'
point_cloud_path = ''
intrinsics = True
rotate_camera = False

#CNN related variables
#model_type = "segnet"


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

    for r in range(rows):
        for c in range(cols):
            segmented_gray_value = segmented_gray[r, c]  # use this for the ply file
            # print(raw_depth_value)
            if segmented_gray_value == 0:
                raw_depth[r, c] = 0

    return raw_depth


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


def metrics2D(img, predMask):
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
    start_y = np.array(predMask).shape[0] - 170
    cv2.putText(added_image, "# of Storage Roots = " + str(len(new_contours)), (start_x, start_y), font_small, 1,
                (255, 0, 255), 1, cv2.LINE_AA)
    # Print the lengths of storage roots in pixels
    for i, cnt in enumerate(new_contours):
        rect = cv2.minAreaRect(cnt)

        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(added_image, [box], 0, (0, 0, 255), 2)

        start_y += 15
        perimeter = cv2.arcLength(cnt, True)
        root_len = perimeter / 2
        # get centriod of countour
        x, y = np.int0(rect[0])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(added_image, str(i), (int(x - 10), int(y)), font, 1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(added_image, "Length of Root #" + str(i) + " = " + str(int(round(root_len))) + " Pixels",
                    (start_x, start_y), font_small, 1,
                    (255, 0, 255), 1, cv2.LINE_AA)

    return added_image


if __name__ == '__main__':
    rotate_camera = False
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

    # ========================
    # 1. Configure all streams
    # ========================
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)

    # ======================
    # 2. Start the streaming
    # ======================
    print("Starting up the Intel Realsense D435...")
    print("")
    profile = pipeline.start(config)

    # =================================
    # 3. The depth sensor's depth scale
    # =================================
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)
    print("")

    # ==========================================
    # 4. Create an align object.
    #    Align the depth image to the rgb image.
    # ==========================================
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        # ===========================================
        # 5. Skip the first 30 frames.
        # This gives the Auto-Exposure time to adjust
        # ===========================================
        for x in range(30):
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

        print("Intel Realsense D435 started successfully.")
        print("")

        while True:
            # ======================================
            # 6. Wait for a coherent pair of frames:
            # ======================================
            frames = pipeline.wait_for_frames()

            # =======================================
            # 7. Align the depth frame to color frame
            # =======================================
            aligned_frames = align.process(frames)

            # ================================================
            # 8. Fetch the depth and colour frames from stream
            # ================================================
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # print the camera intrinsics just once. it is always the same
            depth_intrinsic = depth_frame.profile.as_video_stream_profile().intrinsics
            if intrinsics:
                print("Intel Realsense D435 Camera Intrinsics: ")
                print("========================================")
                print(depth_frame.profile.as_video_stream_profile().intrinsics)
                print(color_frame.profile.as_video_stream_profile().intrinsics)

                print("")
                intrinsics = False

            # =====================================
            # 9. Apply filtering to the depth image
            # =====================================
            # Apply a spatial filter without hole_filling (i.e. holes_fill=0)
            depth_frame = spatial_filtering(depth_frame, magnitude=2, alpha=0.5, delta=50, holes_fill=0)
            # Apply hole filling filter
            depth_frame = hole_filling(depth_frame)

            # ===========================
            # 10. colourise the depth map
            # ===========================
            depth_color_frame = rs.colorizer().colorize(depth_frame)

            # ==================================
            # 11. Convert images to numpy arrays
            # ==================================
            depth_image = np.array(depth_frame.get_data())
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if rotate_camera:
                color_image = np.rot90(color_image, k=3)
                depth_color_image = np.rot90(depth_color_image, k=3)

            # Stack rgb and depth map images horizontally for visualisation only
            images = np.hstack((color_image, depth_color_image))

            cv2.namedWindow('RGB and Depth Map Images')
            cv2.imshow('RGB and Depth Map Images', images)
            c = cv2.waitKey(1)

            if c == ord('s'):
                # Segment the colour rgb image using CNN
                cnn_image, raw_depth = read_images(rgb="images/1.png", depth="images/1.raw")

                if rotate_camera:
                    raw_depth = np.rot90(raw_depth, k=3)
                    cnn_image = np.rot90(cnn_image, k=3)

                #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(raw_depth, alpha=0.03), cv2.COLORMAP_JET)
                #np.savetxt("depth.csv", np.array(raw_depth), delimiter=",")
                #cv2.imshow("rotated image", cnn_image)
                #cv2.waitKey(50000)
                #cv2.imwrite("depth.png", depth_colormap)

                # cnn_image = color_image # uncomment this line ........
                cnn_image_rgb = cnn_image

                # prepare the rgb image for prediction
                cnn_image = cv2.resize(cnn_image, (para.img_cols, para.img_rows), interpolation=cv2.INTER_NEAREST)
                cnn_image = np.array(cnn_image).astype('uint8')
                cnn_image = cnn_image.astype(np.float32)
                cnn_image = cnn_image / 255.0
                cnn_image = cnn_image.reshape((-1, para.img_rows, para.img_cols, 3))

                # Predict RGB Mask
                p_mask = model.predict(cnn_image)
                prediction = np.argmax(p_mask, axis=-1).reshape((para.img_rows, para.img_cols))
                predMask = dataset_utils.visualise_mask(prediction)
                predMask = np.uint8(predMask)

                # Calculate Metrics. returns the measured image with 2 measurements
                metric2D_RGB = metrics2D(cnn_image_rgb, predMask)

                ## Mask the image in the raw depth map
                raw_depth_segmented = segment_raw_depth_image(predMask, raw_depth)
                ## convert the masked depth map to pointcloud and save results
                pts3D, rgb_values = depth2D_to_point3D(depth_intrinsic, raw_depth_segmented, color_image,
                                                       depth_scale)  # depth and rgb should have the same dimensions
                fn = "images/1.ply"
                write_ply(fn, pts3D, rgb_values)

                # calculate volume of each blob in 3D

                # Show horizontally stacked rgb and depth map images
                cv2.imshow('2D Cassava Metrics', metric2D_RGB)
            if c == 27:  # esc to exit
                break

    finally:
        # Stop streaming
        pipeline.stop()