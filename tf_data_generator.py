import scipy
from glob import glob
import numpy as np
import sys
import open3d
import os
import json
import yaml
import cv2
import transforms3d as tf3d
import copy
import imgaug.augmenters as iaa
import multiprocessing
import tensorflow as tf
import time

bop_renderer_path = '/home/stefan/bop_renderer/build'
sys.path.append(bop_renderer_path)

import bop_renderer


class TFDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, dataset_type, dataset_path, real_path, mesh_path, mesh_info, object_id, batch_size, img_res=(224, 224, 3), is_testing=False):
        self.data_type = dataset_type
        self.img_res = img_res
        self.dataset_path = dataset_path
        self.real_path = [os.path.join(real_path, x) for x in os.listdir(real_path)]
        self.batch_size = batch_size
        self.is_testing = is_testing
        self.ply_path = mesh_path
        self.obj_id = int(object_id)

        # annotate
        self.train_info = os.path.join(self.dataset_path, 'annotations', 'instances_' + 'train' + '.json')
        self.val_info = os.path.join(self.dataset_path, 'annotations', 'instances_' + 'val' + '.json')
        # self.mesh_info = os.path.join(self.dataset_path, 'annotations', 'models_info' + '.yml')
        self.mesh_info = mesh_info
        with open(self.train_info, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        anno_ann = data["annotations"]
        self.image_ids = []
        self.Anns = []

        # init renderer
        # < 11 ms;
        self.ren = bop_renderer.Renderer()
        self.ren.init(640, 480)
        self.ren.add_object(self.obj_id, self.ply_path)

        stream = open(self.mesh_info, 'r')
        for key, value in yaml.load(stream).items():
            # for key, value in yaml.load(open(self.mesh_info)).items():
            if int(key) == self.obj_id + 1:
                self.model_dia = value['diameter']

        for ann in anno_ann:
            y_mean = (ann['bbox'][0] + ann['bbox'][2] * 0.5)
            x_mean = (ann['bbox'][1] + ann['bbox'][3] * 0.5)
            max_side = np.max(ann['bbox'][2:])
            x_min = int(x_mean - max_side * 0.75)
            x_max = int(x_mean + max_side * 0.75)
            y_min = int(y_mean - max_side * 0.75)
            y_max = int(y_mean + max_side * 0.75)
            if ann['category_id'] != 2 or ann[
                'feature_visibility'] < 0.5 or x_min < 0 or x_max > 639 or y_min < 0 or y_max > 479:
                continue
            else:
                self.Anns.append(ann)
                # for img_info in image_ann:
                # print(img_info)
                #    if img_info['id'] == ann['id']:
                #        self.image_ids.append(img_info['file_name'])
                #        print(img_info['file_name'])
                template_name = '00000000000'
                id = str(ann['image_id'])
                # print(ann['id'])
                name = template_name[:-len(id)] + id + '.png'
                # print(name)
                self.image_ids.append(name)

        self.fx = image_ann[0]["fx"]
        self.fy = image_ann[0]["fy"]
        self.cx = image_ann[0]["cx"]
        self.cy = image_ann[0]["cy"]

        #self.image_idxs = range(len(self.image_ids))
        c = list(zip(self.Anns, self.image_ids))#, self.image_idxs))
        np.random.shuffle(c)
        self.Anns, self.image_ids = zip(*c)

        self.img_seq = iaa.Sequential([
            # blur
            iaa.SomeOf((0, 2), [
                iaa.GaussianBlur((0.0, 2.0)),
                iaa.AverageBlur(k=(3, 7)),
                iaa.MedianBlur(k=(3, 7)),
                iaa.BilateralBlur(d=(1, 7)),
                iaa.MotionBlur(k=(3, 7))
            ]),
            # color
            iaa.SomeOf((0, 2), [
                # iaa.WithColorspace(),
                iaa.AddToHueAndSaturation((-15, 15)),
                # iaa.ChangeColorspace(to_colorspace[], alpha=0.5),
                iaa.Grayscale(alpha=(0.0, 0.2))
            ]),
            # brightness
            iaa.OneOf([
                iaa.Sequential([
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.Multiply((0.75, 1.25), per_channel=0.5)
                ]),
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.Multiply((0.75, 1.25), per_channel=0.5),
                iaa.FrequencyNoiseAlpha(
                    exponent=(-4, 0),
                    first=iaa.Multiply((0.75, 1.25), per_channel=0.5),
                    second=iaa.LinearContrast((0.7, 1.3), per_channel=0.5))
            ]),
            # contrast
            iaa.SomeOf((0, 2), [
                iaa.GammaContrast((0.75, 1.25), per_channel=0.5),
                iaa.SigmoidContrast(gain=(0, 10), cutoff=(0.25, 0.75), per_channel=0.5),
                iaa.LogContrast(gain=(0.75, 1), per_channel=0.5),
                iaa.LinearContrast(alpha=(0.7, 1.3), per_channel=0.5)
            ]),
        ], random_order=True)

        self.n_batches = int(np.floor(len(self.image_ids) / self.batch_size))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = self.indexes[indexes]
        start_t = time.time()
        inputs, outputs = self.__data_generation(list_IDs_temp)
        print('time batch: ', time.time() - start_t)

        return inputs, outputs

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_ids))

        np.random.shuffle(self.indexes)

    def render_img(self, extrinsics, obj_id):
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3] #* 0.001
        R_list = R.flatten().tolist()
        t_list = t.flatten().tolist()

        light_pose = [np.random.rand() * 2000.0 - 1000.0, np.random.rand() * 2000.0 - 1000.0, 0.0]
        # light_color = [np.random.rand() * 0.1 + 0.9, np.random.rand() * 0.1 + 0.9, np.random.rand() * 0.1 + 0.9]
        light_color = [1.0, 1.0, 1.0]
        light_ambient_weight = np.random.rand()
        light_diffuse_weight = 0.75 + np.random.rand() * 0.25
        light_spec_weight = 0.25 + np.random.rand() * 0.25
        light_spec_shine = np.random.rand() * 3.0

        # time negligible
        self.ren.set_light(light_pose, light_color, light_ambient_weight, light_diffuse_weight, light_spec_weight,
                           light_spec_shine)

        # render + get < 23 ms i5-6600k
        self.ren.render_object(obj_id, R_list, t_list, self.fx, self.fy, self.cx, self.cy)
        rgb_img = self.ren.get_color_image(obj_id)

        return rgb_img

    def __data_sample(self, obsv_img, annotation):
        # real_img = cv2.imread(batch_real[idx]).astype(np.float)

        pad_val = 150
        obsv_img = obsv_img.astype(np.uint8)
        obsv_img = self.img_seq.augment_image(obsv_img)
        obsv_img_pad = np.pad(obsv_img, ((pad_val, pad_val), (pad_val, pad_val), (0, 0)), mode='edge')

        # annotate
        rand_pose = np.eye((4), dtype=np.float32)
        rand_pose[:3, :3] = tf3d.euler.euler2mat(np.random.normal(scale=np.pi * 0.15),
                                                 np.random.normal(scale=np.pi * 0.15),
                                                 np.random.normal(scale=np.pi * 0.15))
        rand_quat = tf3d.quaternions.mat2quat(rand_pose[:3, :3])
        rand_pose[:3, 3] = [np.random.normal(scale=10), np.random.normal(scale=10), np.random.normal(scale=10)]
        anno_pose = np.array(
            [rand_pose[0, 3], rand_pose[1, 3], rand_pose[2, 3], rand_quat[1], rand_quat[1], rand_quat[2],
             rand_quat[3]])
        # normalize annotation
        anno_pose[:3] = anno_pose[:3] * (1 / 10)
        anno_pose = np.repeat(anno_pose[np.newaxis, :], repeats=25, axis=0)
        anno_pose = np.reshape(anno_pose, (5, 5, 7))

        # render and sample crops
        true_pose = np.eye((4), dtype=np.float32)
        true_pose[:3, :3] = tf3d.quaternions.quat2mat(annotation['pose'][3:])
        true_pose[:3, 3] = annotation['pose'][:3]
        obsv_pose = np.eye((4), dtype=np.float32)
        obsv_pose[:3, :3] = np.matmul(true_pose[:3, :3], rand_pose[:3, :3])
        obsv_pose[:3, 3] = true_pose[:3, 3] + rand_pose[:3, 3]

        obsv_center_y = ((obsv_pose[0, 3] * self.fx) / obsv_pose[2, 3]) + self.cx
        obsv_center_x = ((obsv_pose[1, 3] * self.fy) / obsv_pose[2, 3]) + self.cy
        dia_pixX = ((self.model_dia * self.fx) / obsv_pose[2, 3])
        dia_pixY = ((self.model_dia * self.fy) / obsv_pose[2, 3])

        x_min = int(obsv_center_x - dia_pixX * 0.75)
        x_max = int(obsv_center_x + dia_pixX * 0.75)
        y_min = int(obsv_center_y - dia_pixY * 0.75)
        y_max = int(obsv_center_y + dia_pixY * 0.75)
        # print(x_min, y_min, x_max, y_max)

        img_rend_v = self.render_img(obsv_pose, self.obj_id)
        # img_rend_v = np.zeros((640,480, 3), dtype=np.uint8)
        img_rend = np.pad(img_rend_v, ((pad_val, pad_val), (pad_val, pad_val), (0, 0)), mode='constant')
        img_rend = img_rend[(x_min + pad_val):(x_max + pad_val), (y_min + pad_val):(y_max + pad_val), :]
        img_obsv = obsv_img_pad[(x_min + pad_val):(x_max + pad_val), (y_min + pad_val):(y_max + pad_val), :]
        # img_real = real_img[(x_min+pad_val):(x_max+pad_val), (y_min+pad_val):(y_max+pad_val), :]

        # print(x_min, y_min, x_max, y_max)
        # print(rand_pose)
        # print(img_rend.shape)
        # print(img_obsv.shape)
        # print(img_real.shape)
        # print(real_img.shape)
        # img_input = np.concatenate([img_obsv, img_rend], axis=1)
        # cv2.imwrite('/home/stefan/PADA_viz/img_input.png', img_input)

        # print(x_max, y_max)
        # if x_min < -100 or y_min < -100 or x_max > 740 or y_max > 580:
        #    print(x_min, y_min, x_max, y_max)
        #    print(rand_pose)
        # img_input = np.concatenate([img_obsv, img_rend], axis=1)
        #    img_viz = np.where(img_rend > 0, img_rend, img_obsv)
        #    cv2.imwrite('/home/stefan/PADA_viz/img_input.png', img_viz)

        img_rend = cv2.resize(img_rend, self.img_res)
        img_obsv = cv2.resize(img_obsv, self.img_res)

        img_obsv = np.array(img_obsv) / 127.5 - 1.
        img_rend = np.array(img_rend) / 127.5 - 1.
        # imgs_real = np.array(imgs_real) / 127.5 - 1.

        return img_obsv, img_rend, anno_pose

    '''
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #batch = self.image_ids[list_IDs_temp]
        #annos = self.Anns[list_IDs_temp]
        batch = np.array(self.image_ids)[list_IDs_temp.astype(int)]
        annos = np.array(self.Anns)[list_IDs_temp.astype(int)]
        #batch_real = np.random.choice(self.real_path, self.batch_size)

        imgs_obsv, imgs_rend, imgs_real, poses = [], [], [], []
        for idx, current_path in enumerate(batch):

            # render < 25 ms, all < 40
            source_obs, source_ren, target = self.__data_sample(idx, current_path, annos[idx])
            imgs_obsv.append(source_obs)
            imgs_rend.append(source_ren)
            poses.append(target)

        imgs_obsv = np.array(imgs_obsv) / 127.5 - 1.
        imgs_rend = np.array(imgs_rend) / 127.5 - 1.
        # imgs_real = np.array(imgs_real) / 127.5 - 1.
        poses = np.array(poses)

        return [imgs_obsv, imgs_rend], poses
    '''

    def generate_batch(self):
        def generator():
            for i in range(self.batch_size):
                idx = np.random.choice(np.arange(len(self.image_ids)), 1, replace=False)
                # Synthesize an image and a class label.
                current_path = self.image_ids[idx[0]]
                img_path = os.path.join(self.dataset_path, 'images', self.data_type, current_path)
                img_path = img_path[:-4] + '_rgb.png'
                img = cv2.imread(img_path).astype(np.float)
                anno = self.Anns[idx[0]]
                x,y,a = self.__data_sample(img, anno)
                yield (x, y), a
        return generator

    def sample_batch(self):
        def generator():
            for i in range(self.batch_size):
                idx = np.random.choice(np.arange(len(self.image_ids)), 1, replace=False)
                # Synthesize an image and a class label.
                # dirty hack
                x = np.zeros((self.img_res[0], self.img_res[1], 3), dtype=np.float64)
                y = np.zeros((self.img_res[0], self.img_res[1], 3), dtype=np.float64)
                a = np.zeros((5, 5, 7), dtype=np.float64)
                x[0, 0, 0] = idx
                yield (x, y), a
        return generator

    def load_and_prepare_batch(self, x, a):
        idx = x[0][0, 0, 0]

        current_path = self.image_ids[idx]
        img_path = os.path.join(self.dataset_path, 'images', self.data_type, current_path)
        img_path = img_path[:-4] + '_rgb.png'
        img = cv2.imread(img_path).astype(np.float)
        anno = self.Anns[idx]
        x, y, a = self.__data_sample(img, anno)
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        y = tf.convert_to_tensor(y, dtype=tf.float64)
        a = tf.convert_to_tensor(a, dtype=tf.float64)

        return (x, y), a

    def wrap_tf_function(self, x, a):
        output = tf.py_function(func=self.load_and_prepare_batch, inp=[x, a], Tout=((tf.float64, tf.float64), tf.float64))
        return output



