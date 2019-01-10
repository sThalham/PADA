from __future__ import print_function, division
import scipy

#from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
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


class Pix2Pix():
    def __init__(self, dataset):
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = dataset
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))
        self.model_name = 'saved_model/' + self.dataset_name + '_model.h5'


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        # patch = int( 24 )
        self.disc_patch = (patch, patch, 1)
        #self.disc_patch = ( 30 , 40, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64
        #self.gf = np.array([ 120, 160 ])
        #self.df = np.array([ 120, 160 ])

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
  
        # to rid useless error
        #self.discriminator.compile(optimizer)

    def set_dataset_name(self, dataset):
        self.dataset_name = dataset
        print('dataset_name set to: ', dataset)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = Dropout(0.5)(d)
            return d

        def deconv2d(layer_input, skip_input=None, filters=0, f_size=4, dropout_rate=0.5):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = LeakyReLU(alpha=0.2)(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            if skip_input == None:
                u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)
        print('input shape: ', self.img_shape)

        # Downsampling
        d1 = conv2d(d0, filters=self.gf, bn=True)
        d2 = conv2d(d1, filters=self.gf*2, bn=True) # 126
        d3 = conv2d(d2, filters=self.gf*4, bn=True) # 256
        d4 = conv2d(d3, filters=self.gf*8, bn=True) # 512
        d5 = conv2d(d4, filters=self.gf*8, bn=True) # 512
        d6 = conv2d(d5, filters=self.gf*8, bn=True) # 512
        d7 = conv2d(d6, filters=self.gf*8, bn=True)
        d8 = conv2d(d7, filters=self.gf*8, bn=True)

        # Upsampling
        u1 = deconv2d(d8, filters=self.gf*8)
        u2 = deconv2d(u1, d7, filters=self.gf*16)
        u3 = deconv2d(u2, d6, filters=self.gf*16)
        u4 = deconv2d(u3, d5, filters=self.gf*16)
        u5 = deconv2d(u4, d4, filters=self.gf*16)
        u6 = deconv2d(u5, d3, filters=self.gf*8)
        u7 = deconv2d(u6, d2, filters=self.gf*4)
        u8 = deconv2d(u7, d1, filters=self.gf*2)

        #u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u8)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = BatchNormalization(momentum=0.8)(d)
            d = LeakyReLU(alpha=0.2)(d)
            d = Dropout(0.5)(d)

            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        #d_out = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        #validity = sigmoid(d_out)

        return Model([img_A, img_B], validity)

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
    gan = Pix2Pix(sys.argv[1])
    gan.train(epochs=100, batch_size=5, sample_interval=10)
    gan.test(batch_size=5)

