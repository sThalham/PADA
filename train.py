from __future__ import print_function, division
import scipy

#from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Subtract, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import sigmoid
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import datetime
import sys
from data_loader import DataLoader
from pathlib import Path
import numpy as np
import os
import cv2
from glob import glob

from keras_contrib.applications import resnet
import keras_resnet
import keras_resnet.models


class default_model():
    def __init__(self, input_shape=(224, 224)):
        # Input shape
        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))
        self.model_name = 'saved_model/' + '_model.h5'

        optimizer = Adam(0.0002, 0.5)

        img_observed = Input(shape=self.img_shape)
        img_rendered = Input(shape=self.img_shape)
        img_da = Input(shape=self.img_shape)

        self.backbone_obs = self.resnet_no_top()
        print(self.backbone_obs)
        self.backbone_ren = self.resnet_no_top()
        estimator = self.build_generator()

        delta, aux_task = estimator([img_observed, img_rendered, img_da])

        self.model = Model(inputs=[img_observed, img_rendered, img_da], outputs=[delta, aux_task])
        print(self.model.summary())
        self.model.compile(loss=['mae', 'binary_crossentropy'], weights=[1, 1], optimizer=optimizer)

    def resnet_no_top(self):

        input = Input(shape=self.img_shape)
        resnet = keras_resnet.models.ResNet18(input, include_top=False, freeze_bn=True)

        outputs = self.PFPN(resnet.outputs[1], resnet.outputs[2], resnet.outputs[3])

        return Model(inputs=input, outputs=outputs)

    def PFPN(self, C3, C4, C5, feature_size=256):

        P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3)
        P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4)
        P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5)

        P5_upsampled = UpSampling2D(size=2, interpolation="bilinear")(P5)
        P4_upsampled = UpSampling2D(size=2, interpolation="bilinear")(P4)
        P4_mid = Add()([P5_upsampled, P4])
        P4_mid = Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P4_mid)
        P3_mid = Add()([P4_upsampled, P3])sigmoid
        P3_mid = Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P3_mid)
        P3_down = Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P3_mid)
        #P3_fin = Add()([P3_mid, P3])
        #P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3_fin)

        P4_fin = Add()([P3_down, P4_mid])
        P4_down = Conv2D(feature_size, kernel_size=3, strides=2, padding='same')(P4_fin)
        #P4_fin = keras.layers.Add()([P4_fin, P4])  # skip connection
        #P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4_fin)

        P5_fin = Add()([P4_down, P5])
        P5 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5_fin)

        return P5

    def build_generator(self, pyramid_features=256, head_features=256):

        # Image input
        obs = Input(shape=self.img_shape)
        ren = Input(shape=self.img_shape)
        real = Input(shape=self.img_shape)

        model_obs = self.backbone_obs(obs)
        model_ren = self.backbone_ren(ren)

        diff = Subtract()([model_obs, model_ren])
        delta = Conv2D(4, kernel_size=3)(diff)

        da_obs = self.backbone_obs(real)
        da_ren = self.backbone_ren(real)
        da_out_obs = Conv2D(4, kernel_size=3)(da_obs)
        da_out_ren = Conv2D(4, kernel_size=3)(da_ren)
        da_act_obs = Activation('sigmoid')(da_out_obs)
        da_act_ren = Activation('sigmoid')(da_out_ren)
        da_out = Concatenate()([da_act_obs, da_act_ren])

        return Model(inputs=[obs, ren, real], outputs=[delta, da_out])

    def train(self, epochs, batch_size=1, sample_interval=50):

        model_file = Path(self.model_name)
        if model_file.is_file():
            print("MODEL EXISTS... skip training. Please delete model file to retrain")
            self.combined.load_weights(self.model_name)
            return

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

            # Save samples once every epoch
            self.sample_images(epoch)

            #if epoch % 10 == 0:
            
        self.save_model_weights(self.combined, self.model_name)	
        print("Training finished!")


    def test(self, batch_size=1):
        files = glob('./datasets/%s/%s/*' % (self.dataset_name, "test"))
        lenFolder = len("./datasets/" + self.dataset_name + "/test/")        
        amoFiles = len(files)   
        print("found ", amoFiles, " to process")     

        apro = 0
        ppro = batch_size
        while (ppro) < amoFiles:
            if (ppro) >= amoFiles:
                ppro = apro + (amoFiles - apro)

            paths = files[apro:ppro]
            imgs = self.data_loader.load_test_data(paths)
            print("Test batch [%d:%d] of [%d] loaded" % (apro, ppro, amoFiles))
 
            fakes = self.generator.predict(imgs)
            fakes = 127.5 * fakes + 127.5

            for i in range(batch_size):
                fn = paths[i]
                fn = fn[lenFolder:]
                fn = "./results/" + fn
                print(fn)
                img = scipy.misc.imresize(fakes[i], (480, 640))
                cv2.imwrite(fn, img)
            print("processed [%d:%d] of [%d]" % (apro, ppro, amoFiles))
            apro = apro + batch_size
            ppro = ppro + batch_size
        print("generated images under /results")   


    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3
        batch_size = 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        # Rescale images 0 - 255
        imgs_A = (imgs_A + 1.0 ) * 127.5
        imgs_B = (imgs_B + 1.0 ) * 127.5
        fake_A = (fake_A + 1.0 ) * 127.5

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # titles = ['Condition', 'Generated', 'Original']
        for i in range(batch_size):
            fn = ("images/%s/%d_%d.png" % (self.dataset_name, epoch, i))
            cv2.imwrite(fn, gen_imgs[i])
        print('samples generated!')

    def save_model_weights(self, model, filepath, overwrite=True):
        model_file = Path(self.model_name)
        if model_file.is_file():
            model.save_weights(filepath, overwrite=True)
        else:
            model.save_weights(filepath, overwrite=False)


if __name__ == '__main__':
    gan = default_model()
    gan.train(epochs=100, batch_size=5, sample_interval=10)
    gan.test(batch_size=5)

