import parameters as para
from Models import segnet_lite, segnet, unet_lite
import dataset_utils
import cv2, os
from custom_metrics import *
from keras.utils import np_utils

if __name__ == '__main__':
    #   Call model
    #   ==========
    output_rows = 0
    output_cols = 0

    if para.model_type == "segnet_lite":
        print('Initialising Segnet Lite...')
        model, output_rows, output_cols = segnet_lite.SegNet()

    if para.model_type == "segnet":
        print('Initialising SegNet...')
        model, output_rows, output_cols = segnet.SegNet()

    if para.model_type == "unet_lite":
        print('Initialising Unet Lite...')
        model, output_rows, output_cols = unet_lite.unet_lite()

    #   Compile the model using sgd optimizer
    #   =====================================
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[dice_coef, precision, recall, 'accuracy'])
    print(model.summary())
    print(para.misc_dir_eval)
    # Load the model weights
    model.load_weights(para.misc_dir_eval + "/weights.h5")

    for cls in range(para.count_classes):
        #   Get the evaluation images and labels
        test_images, test_masks, test_image_names = dataset_utils.prepare_evaluation_images2(image_dir=para.test,
                                                                                             mask_dir=para.testannot,
                                                                                             img_rows=para.img_rows,
                                                                                             img_cols=para.img_cols,
                                                                                             cls=cls)

        test_masks = np_utils.to_categorical(test_masks, para.seg_classes)
        test_masks = np.reshape(test_masks, (len(test_masks), para.img_rows * para.img_cols, para.seg_classes))
        steps_per_epoch = len(test_images) / para.batch_size
        test_images = test_images.astype(np.float32)
        test_images = test_images / 255.0

        if len(test_images) > 0:
            print("Evaluating ", cls , "storage roots")
            print("==================================")
            #   Evaluate and print the score
            print("Evaluating test data and computing pixel accuracy...")
            scores = model.evaluate(test_images, test_masks, verbose=1, batch_size=para.batch_size)
            print("%s: %.2f%%" % ("Dice Score: ", scores[1]*100))
            print("%s: %.2f%%" % ("Precision: ", scores[2] * 100))
            print("%s: %.2f%%" % ("Recall: ", scores[3] * 100))
            print("%s: %.2f%%" % ("Pixel Accuracy: ", scores[4] * 100))

            #   Perform predictions, display results and save results
            pred_mask = model.predict(test_images, batch_size=para.batch_size)
            mean_iou = MeanIOU(test_masks, pred_mask)
            print("%s: %.2f%%" % ("Mean IoU: ", mean_iou * 100))

            '''Visualise the predictions and save results'''

            image_path = para.misc_dir + "/results"

            if not os.path.exists(image_path):
                os.makedirs(image_path, mode=0o777)

            annot_path = para.misc_dir + "/annot/" + str(cls)
            if not os.path.exists(annot_path):
                os.makedirs(annot_path, mode=0o777)

            prediction_k = []
            f = open(image_path + "/" + 'mean_iou.csv', 'w')
            f.write("Dice Score: , " + str(scores[1]*100) + "\n")
            f.write("Precision: , " + str(scores[2] * 100) + "\n")
            f.write("Recall: , " + str(scores[3] * 100) + "\n")
            f.write("Pixel Accuracy: , " + str(scores[4] * 100) + "\n")
            f.write("Mean IoU: , " + str(mean_iou * 100) + "\n")

            for p_mask, gt_mask, t_image, img_names in zip(pred_mask, test_masks, test_images, test_image_names):
                f.write(img_names + "," + str(MeanIOU(gt_mask, p_mask) * 100) + "\n")

                prediction = np.argmax(p_mask, axis=1).reshape((para.img_rows, para.img_cols))
                ground_truth = np.argmax(gt_mask, axis=1).reshape((para.img_rows, para.img_cols))

                pred = dataset_utils.visualise_mask(prediction)
                g_truth = dataset_utils.visualise_mask(ground_truth)

                cv2.imwrite(annot_path + "/" + img_names, pred)

            f.close()