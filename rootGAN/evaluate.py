import parameters as para
from Models import segnet_depool
import dataset_utils
import cv2, os
from custom_metrics import *

IoU_threshold = 0.7

if __name__ == '__main__':
    #   Call model
    #   ==========
    output_rows = 0
    output_cols = 0

    if para.model_type == "segnet":
        print('Initialising SegNet...')
        model, output_cols, output_rows = segnet_depool.SegNet()

    #   Compile the model using sgd optimizer
    #   =====================================
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[dice_coef, precision, recall, 'accuracy'])
    print(model.summary())
    print(para.misc_dir_eval)
    # Load the model weights
    model.load_weights(para.misc_dir_eval + "/weights.h5")
    #   Get the evaluation images and labels

    for cls in range(para.num_classes):
        test_images, test_masks, test_image_names = dataset_utils.prepare_evaluation_images(image_dir=para.test_data, mask_dir=para.test_data_annot,
                                                img_rows=para.img_rows, img_cols=para.img_cols, img_rows_out=output_rows,
                                                img_cols_out=output_cols, classes=cls)

        test_masks = np.reshape(test_masks, (len(test_masks), para.img_rows * para.img_cols, para.num_classes))

        #   Evaluate and print the score
        print("Evaluating test data and computing pixel accuracy...")
        scores = model.evaluate(test_images, test_masks, verbose=1, batch_size=para.batch_size)
        print("%s: %.2f%%" % ("Pixel Accuracy: ", scores[4] * 100))

        #   Perform predictions, display results and save results
        pred_mask = model.predict(test_images, batch_size=para.batch_size)
        mean_iou = MeanIOU(test_masks, pred_mask)
        print("%s: %.2f%%" % ("Mean IoU: ", mean_iou * 100))

        annot_pred_path = para.misc_dir + "/trainvalannot_pred/" + str(cls)
        if not os.path.exists(annot_pred_path):
            os.makedirs(annot_pred_path, mode=0o777)

        imgs_path = para.misc_dir + "/trainval/" + str(cls)
        if not os.path.exists(imgs_path):
            os.makedirs(imgs_path, mode=0o777)

        annot_gt_path = para.misc_dir + "/trainvalannot/" + str(cls)
        if not os.path.exists(annot_gt_path):
            os.makedirs(annot_gt_path, mode=0o777)

        m_iou_path = para.misc_dir + "/m_iou/" + str(cls)
        if not os.path.exists(m_iou_path):
            os.makedirs(m_iou_path, mode=0o777)

        f = open(m_iou_path + "/" + 'mean_iou.csv', 'w')
        for p_mask, gt_mask, t_image, img_names in zip(pred_mask, test_masks, test_images, test_image_names):
            m_iou_score = MeanIOU(gt_mask, p_mask) * 100
            if(m_iou_score >= IoU_threshold):
                f.write(img_names + "," + str(m_iou_score) + "\n")

                prediction = np.argmax(p_mask, axis=1).reshape((para.img_rows, para.img_cols))
                ground_truth = np.argmax(gt_mask, axis=1).reshape((para.img_rows, para.img_cols))

                pred = dataset_utils.visualise_mask(prediction)
                g_truth = dataset_utils.visualise_mask(ground_truth)

                cv2.imwrite(annot_pred_path + "/" + img_names, pred)
                cv2.imwrite(annot_gt_path + "/" + img_names, g_truth)
                cv2.imwrite(imgs_path + "/" + img_names, t_image[0])
        f.close()
