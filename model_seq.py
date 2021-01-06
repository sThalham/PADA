from __future__ import print_function, division

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Subtract, Add
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam

#from tensorflow.keras.applications import imagenet_utils
#import tensorflow.keras.applications.imagenet_utils
from tf2_resnets.models import ResNet18
#from tf2cv.model_provider import get_model as tf2cv_get_model


class default_model_seq():
    def __init__(self, input_shape=(224, 224)):
        # Input shape
        self.img_rows = input_shape[0]
        self.img_cols = input_shape[1]
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(lr=1e-5, clipnorm=0.001)

        img_observed = Input(shape=self.img_shape)
        img_rendered = Input(shape=self.img_shape)
        #img_da = Input(shape=self.img_shape)

        self.backbone_obs = self.resnet_no_top()
        self.backbone_ren = self.resnet_no_top()
        estimator = self.build_generator()

        #delta, aux_task = estimator([img_observed, img_rendered, img_da])
        delta = estimator([img_observed, img_rendered])

        model = tf.keras.Sequential()
        model.add(delta)
        model.compile(optimizer=optimizer, loss='mse')

    def resnet_no_top(self):

        input = Input(shape=self.img_shape)
        # DenseNet
        #backbone = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_tensor=input, input_shape=self.img_shape, pooling=None, classes=1)
        #layer_names = [138, 310, 426]
        #backbone_outputs = [backbone.layers[idx].output for idx in layer_names]

        # tf2 ResNet
        backbone = ResNet18(input_tensor=input, input_shape=self.img_shape, weights='imagenet')
        layer_names = [44, 64, 84]
        backbone_outputs = [backbone.layers[idx].output for idx in layer_names]
        #print(backbone_outputs)

        # tf2cv
        #backbone = tf2cv_get_model("resnet18", pretrained=True, data_format="channels_last")
        #outputs = backbone(input)
        #print(outputs)
        #return Model(inputs=input, outputs=outputs)

        #for i, layer in enumerate(backbone.layers):
        #    print(i, layer.name)

        #return Model(inputs=input, outputs=outputs)
        #resnet = keras_resnet.models.ResNet18(input, include_top=False, freeze_bn=True)

        #outputs = self.PFPN(resnet.outputs[1], resnet.outputs[2], resnet.outputs[3])
        return Model(inputs=input, outputs=backbone_outputs)

        #return Model(inputs=input, outputs=[backbone.outputs[1], backbone.outputs[2], backbone.outputs[3]])

    def PFPN(self, C3, C4, C5, feature_size=256):

        P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C3)
        P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C4)
        P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same')(C5)

        P5_upsampled = UpSampling2D(size=2, interpolation="bilinear")(P5)
        P4_upsampled = UpSampling2D(size=2, interpolation="bilinear")(P4)
        P4_mid = Add()([P5_upsampled, P4])
        P4_mid = Conv2D(feature_size, kernel_size=3, strides=1, padding='same')(P4_mid)
        P3_mid = Add()([P4_upsampled, P3])
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
        #real = Input(shape=self.img_shape)

        model_obs = self.backbone_obs(obs)
        model_ren = self.backbone_ren(ren)

        diff_P3 = Subtract()([model_obs[0], model_ren[0]])
        diff_P4 = Subtract()([model_obs[1], model_ren[1]])
        diff_P5 = Subtract()([model_obs[2], model_ren[2]])

        fusion_py = self.PFPN(diff_P3, diff_P4, diff_P5)

        head_tra = Conv2D(head_features, kernel_size=3, strides=1, padding='same')(fusion_py)
        head_tra = Conv2D(head_features, kernel_size=3, strides=1, padding='same')(head_tra)
        head_tra = Conv2D(head_features, kernel_size=3, strides=1, padding='same')(head_tra)
        delta_tra = Conv2D(3, kernel_size=3)(head_tra)

        head_rot = Conv2D(head_features, kernel_size=3, strides=1, padding='same')(fusion_py)
        head_rot = Conv2D(head_features, kernel_size=3, strides=1, padding='same')(head_rot)
        head_rot = Conv2D(head_features, kernel_size=3, strides=1, padding='same')(head_rot)
        delta_rot = Conv2D(4, kernel_size=3)(head_rot)
        print(delta_rot)
        #delta_rot = l2_normalize(delta_rot, axis=-1)

        delta = Concatenate(axis=-1)([delta_tra, delta_rot])

        #da_obs = self.backbone_obs(real)
        #da_ren = self.backbone_ren(real)
        #da_out_obs = Conv2D(4, kernel_size=3)(da_obs)
        #da_out_ren = Conv2D(4, kernel_size=3)(da_ren)
        #da_act_obs = Activation('sigmoid')(da_out_obs)
        #da_act_ren = Activation('sigmoid')(da_out_ren)
        #da_out = Concatenate()([da_act_obs, da_act_ren])

        #return Model(inputs=[obs, ren, real], outputs=[delta, da_out])

        return Model(inputs=[obs, ren], outputs=delta)
