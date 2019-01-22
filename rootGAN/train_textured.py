import datetime
import numpy as np
import parameters as para
import data_loader
from keras.optimizers import Adam
from Models import cGAN
from keras.layers import Input
from keras.models import Model
import os
import cv2


def train(epochs, batch_size=1, sample_interval=50, input_shape=(para.img_rows, para.img_cols,  para.channels), monitor="gloss"):

    start_time = datetime.datetime.now()

    optimizer = Adam(0.0002, 0.5)

    # We use a pre-trained VGG19 model to extract image features from the high resolution
    # and the generated high resolution images and minimize the mse between them
    vgg19 = cGAN.build_vgg19()
    vgg19.trainable = False
    vgg19.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    # Build and compile the discriminator
    discriminator, rows, cols = cGAN.discriminativeModel()
    discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    # Calculate output shape of D (PatchGAN)
    disc_patch = (rows, cols, 1) #=====

    # -------------------------
    # Construct Computational
    #   Graph of Generators
    # -------------------------

    # Build the generators
    generator = cGAN.generativeModel()

    # Input images and their conditioning images
    img_A = Input(shape=input_shape)
    img_B = Input(shape=input_shape)

    # By conditioning on B generate a fake version of A
    fake_A = generator(img_B)

    # Extract image features of the generated img
    fake_A_features = vgg19(fake_A)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # Discriminators determines validity of translated images / condition pairs
    validity = discriminator([fake_A, img_B])

    combined = Model(inputs=[img_A, img_B], outputs=[validity, fake_A_features])
    combined.compile(loss=['binary_crossentropy', 'mae'], loss_weights=[1e-3, 1], optimizer=optimizer)

    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + disc_patch)
    fake = np.zeros((batch_size,) + disc_patch)

    best_dloss = 9999999
    best_gloss = 9999999
    for epoch in range(epochs):
        sum_gloss=0
        sum_dloss=0
        sum_dacc=0
        #n_batches = 0
        for batch_i, (imgs_A, imgs_B, n_batches) in enumerate(data_loader.load_batch(image_dir=para.trainval, mask_dir=para.trainvalannot, batch_size=batch_size)):
            #print(imgs_A.shape)
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Condition on B and generate a translated version
            fake_A = generator.predict(imgs_B)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = discriminator.train_on_batch([fake_A, imgs_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            imgs_A_features = vgg19.predict(imgs_A)

            # Train the generators
            g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A_features])

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" %
                  (epoch, epochs, batch_i, n_batches, d_loss[0], 100 * d_loss[1], g_loss[0], elapsed_time))

            sum_dloss = sum_dloss + d_loss[0]
            sum_gloss = sum_gloss + g_loss[0]
            sum_dacc = sum_dacc + d_loss[1]

            #print(batch_i % sample_interval)
            # If at save interval => save generated image samples
            #if batch_i % sample_interval == 0 and epoch == 200:
            if batch_i == n_batches - para.batch_size and epoch % 100 == 0:
                print("saving image samples.....")
                sample_images(generator, epoch)
                #save models

        if monitor=="dloss" and best_dloss > (sum_dloss/n_batches):
            best_dloss = sum_dloss/n_batches
            print("Summary of Epoch........")
            print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] " %
                  (epoch, epochs, sum_dloss/n_batches, 100 * (sum_dacc/n_batches), sum_gloss/n_batches))
            #print("Saving the model........")
            checkpointing(generator, discriminator)
        if monitor=="gloss" and best_gloss > (sum_gloss/n_batches):
            best_gloss = sum_gloss/n_batches
            print("Summary of Epoch........")
            print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] " %
                  (epoch, epochs, sum_dloss/n_batches, 100 * (sum_dacc/n_batches), sum_gloss/n_batches))
            #print("Saving the model........")
            checkpointing(generator, discriminator)
        elif monitor=="gloss" and best_gloss < (sum_gloss/n_batches):
            print("Summary of Epoch........")
            print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] " %
                  (epoch, epochs, sum_dloss / n_batches, 100 * (sum_dacc / n_batches), sum_gloss / n_batches))
            #print("Saving the model........")
            #checkpointing(generator)


def checkpointing(generator, discriminator):
    checkpoint_folder = para.home_dir + 'Models/' + para.model_type + '/'
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder, mode=0o777)

    checkpoint_file = checkpoint_folder + 'weights_gen.h5'
    generator.save_weights(checkpoint_file, overwrite=True)

    checkpoint_file = checkpoint_folder + 'weights_dis.h5'
    discriminator.save_weights(checkpoint_file, overwrite=True)


def sample_images(generator, epoch, classes=7):
    start_time_gen = datetime.datetime.now()
    for clses in range(classes):

        generator_folder = para.misc_dir2 + '/synthetic_cassava_textured3/' + str(epoch) + '/' + str(clses)
        synthetic_mask_dir = para.data_dir + para.dataset + "/synthetic_mask/"
        if not os.path.exists(generator_folder):
            os.makedirs(generator_folder, mode=0o777)

        images_B = data_loader.load_synthetic_mask(synthetic_mask_dir, is_testing=True, cls=clses)

        for i, imgs_B in enumerate(images_B):
            imgs_B = np.expand_dims(imgs_B, axis=0)
            print("Predicting image ", i, "in class ", clses)
            fake_A = generator.predict(imgs_B)

            gen_imgs = np.concatenate([imgs_B, fake_A])

            # Rescale images 0 - 1
            #gen_imgs = 0.5 * gen_imgs + 0.5

            img = np.array(gen_imgs[0] * 127.5 + 127.5).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(generator_folder + "/" + str(i) + "_mask.png", img)

            img = np.array(gen_imgs[1] * 127.5 + 127.5).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(generator_folder + "/" + str(i) + "_syn.png", img)

    elapsed_time_gen = datetime.datetime.now() - start_time_gen
    print("Synthetic images generated in ", elapsed_time_gen, " for seven classes")


if __name__ == '__main__':
    train(para.num_epoch, batch_size=para.batch_size, sample_interval=50, monitor="gloss")