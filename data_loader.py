import scipy
from glob import glob
import numpy as np
import sys
import open3d
import os
import json
import cv2
import transforms3d as tf3d
import copy

bop_renderer_path = '/home/stefan/workspace/bop_renderer/build'
sys.path.append(bop_renderer_path)

import bop_renderer


class DataLoader():
    def __init__(self, dataset_path, real_path, mesh_path, batch_size, img_res=(224, 224), is_testing=False):
        self.img_res = img_res
        self.dataset_path = dataset_path
        self.real_path = [os.path.join(real_path, x) for x in os.listdir(real_path)]
        self.batch_size = batch_size
        self.is_testing = is_testing
        self.ply_path = mesh_path
        self.obj_id = 1

        # annotate
        self.train_info = os.path.join(self.dataset_path, 'annotations', 'instances_' + 'train' + '.json')
        self.val_info = os.path.join(self.dataset_path, 'annotations', 'instances_' + 'val' + '.json')
        self.mesh_info = os.path.join(self.dataset_path, 'annotations', 'models_info' + '.yml')
        with open(self.train_info, 'r') as js:
            data = json.load(js)
        image_ann = data["images"]
        anno_ann = data["annotations"]
        self.image_ids = []
        self.Anns = []

        # init renderer
        self.ren = bop_renderer.Renderer()
        self.ren.init(640, 480)
        light_pose = [np.random.rand() * 2 - 1.0, np.random.rand() * 2 - 1.0, 0.0]
        # light_color = [np.random.rand() * 0.1 + 0.9, np.random.rand() * 0.1 + 0.9, np.random.rand() * 0.1 + 0.9]
        light_color = [1.0, 1.0, 1.0]
        light_ambient_weight = np.random.rand()
        light_diffuse_weight = 0.75 + np.random.rand() * 0.25
        light_spec_weight = 0.25 + np.random.rand() * 0.25
        light_spec_shine = np.random.rand() * 3.0
        self.ren.set_light(light_pose, light_color, light_ambient_weight, light_diffuse_weight, light_spec_weight,
                           light_spec_shine)
        print(self.ply_path)
        self.ren.add_object(self.obj_id, self.ply_path)

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
                self.Anns.append(ann)
                #for img_info in image_ann:
                    #print(img_info)
                #    if img_info['id'] == ann['id']:
                #        self.image_ids.append(img_info['file_name'])
                #        print(img_info['file_name'])
                template_name = '00000000000'
                id = str(ann['image_id'])
                #print(ann['id'])
                name = template_name[:-len(id)] + id + '.png'
                #print(name)
                self.image_ids.append(name)

        self.fx = image_ann[0]["fx"]
        self.fy = image_ann[0]["fy"]
        self.cx = image_ann[0]["cx"]
        self.cy = image_ann[0]["cy"]

        c = list(zip(self.Anns, self.image_ids))
        np.random.shuffle(c)
        self.Anns, self.image_ids = zip(*c)

        self.n_batches = int(len(self.image_ids) / self.batch_size)

    def render_img(self, extrinsics, obj_id):
        R = extrinsics[:3, :3]
        t = extrinsics[:3, 3] * 0.001
        print(extrinsics)
        R_list = R.flatten().tolist()
        t_list = t.flatten().tolist()
        self.ren.render_object(obj_id, R_list, t_list, self.fx, self.fy, self.cx, self.cy)
        rgb_img = self.ren.get_color_image(obj_id)
        cv2.imwrite('/home/stefan/PADA_viz/img_full.png', rgb_img)

        return rgb_img

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        print('path to load: ', path)
        batch_indices = np.random.randint(2, size=10)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)

            if img.shape[2] < 3:
                img = np.repeat(img, 3, axis=2) 

            h, w, _ = img.shape
            _w = int(w/2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_test_data(self, paths):
        imgs = []
        for path in paths:
            img = self.imread(path)
            if img.shape[2] < 3:
                img = np.repeat(img, 3, axis=2)
            img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)
        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self):
        data_type = "train" if not self.is_testing else "val"

        batch_real = np.random.choice(self.real_path, self.batch_size)

        for i in range(self.n_batches-1):
            batch = self.image_ids[i * self.batch_size:(i+1) * self.batch_size]
            annos = self.Anns[i * self.batch_size:(i + 1) * self.batch_size]
            imgs_obsv, imgs_rend, imgs_real = [], [], []
            for idx, current_path in enumerate(batch):
                img_path = os.path.join(self.dataset_path, 'images', data_type, current_path)
                img_path = img_path[:-4] + '_rgb.png'
                obsv_img = cv2.imread(img_path).astype(np.float)
                real_img = cv2.imread(batch_real[idx]).astype(np.float)
                annotation = annos[idx]
                y_mean = (annotation['bbox'][0] + annotation['bbox'][2] * 0.5)
                x_mean = (annotation['bbox'][1] + annotation['bbox'][3] * 0.5)
                max_side = np.max(annotation['bbox'][2:])
                print(annotation['bbox'])
                print(x_mean, y_mean)
                print(np.max(annotation['bbox'][2:]))
                x_min = int(x_mean - max_side * 0.5)
                x_max = int(x_mean + max_side * 0.5)
                y_min = int(y_mean - max_side * 0.5)
                y_max = int(y_mean + max_side * 0.5)
                #x_min = np.maximum(int(x_mean - max_side * 0.75), 0)
                #x_max = np.minimum(int(x_mean + max_side * 0.75), 479)
                #y_min = np.maximum(int(y_mean - max_side * 0.75), 0)
                #y_max = np.minimum(int(y_mean + max_side * 0.75), 639)
                print(x_min, x_max, y_min, y_max)
                img_obsv = obsv_img[x_min:x_max, y_min:y_max, :]
                img_real = real_img[x_min:x_max, y_min:y_max, :]
                print(img_obsv.shape, img_real.shape)

                img_obsv = cv2.resize(img_obsv, self.img_res)
                img_real = cv2.resize(img_real, self.img_res)

                pose = np.zeros((4, 4), dtype=np.float32)
                pose[:3, :3] = tf3d.quaternions.quat2mat(annotation['pose'][3:])
                pose[:3, 3] = annotation['pose'][:3]
                pose[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).T
                img_rend = self.render_img(pose, self.obj_id)
                img_rend = img_rend[x_min:x_max, y_min:y_max, :]
                img_rend = cv2.resize(img_rend, self.img_res)

                img_input = np.concatenate([img_obsv, img_rend, img_real], axis=1)
                cv2.imwrite('/home/stefan/PADA_viz/img_input.png', img_input)
                cv2.imwrite('/home/stefan/PADA_viz/ori_input.png', obsv_img)

                #if not self.is_testing and np.random.random() > 0.5:
                #        img_A = np.fliplr(img_A)
                #        img_B = np.fliplr(img_B)

                imgs_obsv.append(img_obsv)
                imgs_rend.append(img_rend)
                imgs_real.append(img_real)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

