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

bop_renderer_path = '/home/stefan/bop_renderer/build'
sys.path.append(bop_renderer_path)

import bop_renderer


class Dataset():
    def __init__(self, dataset_type, dataset_path, obj_id, real_path, mesh_path, mesh_info, batch_size, img_res=(224, 224), is_testing=False, workers=3):
        self.data_type = dataset_type
        self.img_res = img_res
        self.dataset_path = dataset_path
        self.real_path = [os.path.join(real_path, x) for x in os.listdir(real_path)]
        self.batch_size = batch_size
        self.is_testing = is_testing
        self.ply_path = mesh_path
        self.obj_id = int(obj_id)

        # multi-proc
        self.pool = multiprocessing.Pool(workers)
        #self.obsv_batch = []
        #self.anno_batch = []
        #self.real_batch = []
        #self.data_type = None

        # annotate
        self.train_info = os.path.join(self.dataset_path, 'annotations', 'instances_' + 'train' + '.json')
        self.val_info = os.path.join(self.dataset_path, 'annotations', 'instances_' + 'val' + '.json')
        #self.mesh_info = os.path.join(self.dataset_path, 'annotations', 'models_info' + '.yml')
        self.mesh_info = mesh_info
        with open(self.train_info, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        anno_ann = data["annotations"]
        self.image_ids = []
        self.Anns = []

        # init renderer
        self.ren = bop_renderer.Renderer()
        self.ren.init(640, 480)
        self.ren.add_object(self.obj_id, self.ply_path)

        stream = open(self.mesh_info, 'r')
        for key, value in yaml.load(stream).items():
        #for key, value in yaml.load(open(self.mesh_info)).items():
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
            if ann['category_id'] != 2 or ann['feature_visibility'] < 0.5 or x_min < 0 or x_max > 639 or y_min < 0 or y_max > 479:
                continue
            else:
                self.Anns.append(ann['pose'])
                #for img_info in image_ann:
                    #print(img_info)
                #    if img_info['id'] == ann['id']:
                #        self.image_ids.append(img_info['file_name'])
                #        print(img_info['file_name'])
                template_name = '00000000000'
                id = str(ann['image_id'])
                name = template_name[:-len(id)] + id + '_rgb.png'
                img_path = os.path.join(self.dataset_path, 'images', self.data_type, name)
                self.image_ids.append(img_path)

        self.fx = image_ann[0]["fx"]
        self.fy = image_ann[0]["fy"]
        self.cx = image_ann[0]["cx"]
        self.cy = image_ann[0]["cy"]

        c = list(zip(self.Anns, self.image_ids))
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
        self.dataset_length = len(self.image_ids)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batches

    def __img_list__(self):
        return self.image_ids

    def __anno_list__(self):
        return self.Anns

    def __renderer__(self):
        return self.ren

    def __augmenter__(self):
        return self.img_seq

    def __image_shape__(self):
        return img_res

    def __model_diameter__(self):
        return self.model_dia

    def __get_intrinsics__(self):
        return [self.fx, self.fy, self.cx, self.cy]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_ids))

        np.random.shuffle(self.indexes)

    def __get_batch__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_indices = self.indexes[indexes]
        return batch_indices


def load_data_sample(idx, img_list, anno_list, renderer, augmenter, intrinsics, obj_id, model_dia):

    img_path = img_list[idx]
    obsv_img = cv2.imread(img_path).astype(np.float)
    annotation = anno_list[idx]
    obsv_img, rend_img, annotation = annotate_batches(obsv_img, annotation, renderer, augmenter, intrinsics, img_res, obj_id, model_dia)

    return obsv_img, rend_img, annotation


def render_img(ren, intrinsics, extrinsics, obj_id):
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]  # * 0.001
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
    ren.set_light(light_pose, light_color, light_ambient_weight, light_diffuse_weight, light_spec_weight,
                       light_spec_shine)

    # render + get < 23 ms i5-6600k
    ren.render_object(obj_id, R_list, t_list, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3])
    rgb_img = ren.get_color_image(obj_id)

    return rgb_img


def annotate_batches(obsv_img, annotation, renderer, augmenter, intrinsics, img_res, obj_id, model_dia):

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
    anno_pose[:3] = anno_pose[:3] * (1 / 30)
    anno_pose = np.repeat(anno_pose[np.newaxis, :], repeats=25, axis=0)

        # render and sample crops
    true_pose = np.eye((4), dtype=np.float32)
    true_pose[:3, :3] = tf3d.quaternions.quat2mat(annotation[3:])
    true_pose[:3, 3] = annotation[:3]
    obsv_pose = np.eye((4), dtype=np.float32)
    obsv_pose[:3, :3] = np.matmul(true_pose[:3, :3], rand_pose[:3, :3])
    obsv_pose[:3, 3] = true_pose[:3, 3] + rand_pose[:3, 3]

    obsv_center_y = ((obsv_pose[0, 3] * intrinsics[0]) / obsv_pose[2, 3]) + intrinsics[2]
    obsv_center_x = ((obsv_pose[1, 3] * intrinsics[1]) / obsv_pose[2, 3]) + intrinsics[3]
    dia_pixX = ((model_dia * intrinsics[0]) / obsv_pose[2, 3])
    dia_pixY = ((model_dia * intrinsics[1]) / obsv_pose[2, 3])

    x_min = int(obsv_center_x - dia_pixX * 0.75)
    x_max = int(obsv_center_x + dia_pixX * 0.75)
    y_min = int(obsv_center_y - dia_pixY * 0.75)
    y_max = int(obsv_center_y + dia_pixY * 0.75)
    # print(x_min, y_min, x_max, y_max)

    img_rend = render_img(renderer, intrinsics, obsv_pose, obj_id)
    img_rend = np.pad(img_rend, ((pad_val, pad_val), (pad_val, pad_val), (0, 0)), mode='edge')
    img_rend = img_rend[(x_min + pad_val):(x_max + pad_val), (y_min + pad_val):(y_max + pad_val), :]
    img_obsv = obsv_img[(x_min + pad_val):(x_max + pad_val), (y_min + pad_val):(y_max + pad_val), :]
    img_obsv = augmenter.augment_image(img_obsv)
    #img_real = real_img[(x_min + pad_val):(x_max + pad_val), (y_min + pad_val):(y_max + pad_val), :]

    img_rend = cv2.resize(img_rend, img_res)
    img_obsv = cv2.resize(img_obsv, img_res)

    return img_rend, img_obsv, anno_pose

