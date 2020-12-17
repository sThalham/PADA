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
        self.ren.add_object(self.obj_id, self.ply_path)

        for key, value in yaml.load(open(self.mesh_info)).items():
            if key == self.obj_id + 1:
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
        R_list = R.flatten().tolist()
        t_list = t.flatten().tolist()
        self.ren.render_object(obj_id, R_list, t_list, self.fx, self.fy, self.cx, self.cy)
        rgb_img = self.ren.get_color_image(obj_id)

        return rgb_img

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

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
            imgs_obsv, imgs_rend, imgs_real, poses = [], [], [], []
            for idx, current_path in enumerate(batch):
                img_path = os.path.join(self.dataset_path, 'images', data_type, current_path)
                img_path = img_path[:-4] + '_rgb.png'
                obsv_img = cv2.imread(img_path).astype(np.float)
                real_img = cv2.imread(batch_real[idx]).astype(np.float)
                annotation = annos[idx]
                #y_mean = (annotation['bbox'][0] + annotation['bbox'][2] * 0.5)
                #x_mean = (annotation['bbox'][1] + annotation['bbox'][3] * 0.5)
                #max_side = np.max(annotation['bbox'][2:])
                #x_min = int(x_mean - max_side * 0.75)
                #x_max = int(x_mean + max_side * 0.75)
                #y_min = int(y_mean - max_side * 0.75)
                #y_max = int(y_mean + max_side * 0.75)
                pad_val = 100
                obsv_img = np.pad(obsv_img, ((pad_val, pad_val), (pad_val, pad_val), (0, 0)), mode='edge')
                real_img = np.pad(real_img, ((pad_val, pad_val), (pad_val, pad_val), (0, 0)), mode='edge')

                true_pose = np.eye((4), dtype=np.float32)
                true_pose[:3, :3] = tf3d.quaternions.quat2mat(annotation['pose'][3:])
                true_pose[:3, 3] = annotation['pose'][:3]
                rand_pose = np.eye((4), dtype=np.float32)
                rand_pose[:3, :3] = tf3d.euler.euler2mat(np.random.normal(scale=np.pi*0.25), np.random.normal(scale=np.pi*0.25), np.random.normal(scale=np.pi*0.25))
                rand_pose[:3, 3] = [np.random.normal(scale=30), np.random.normal(scale=30), np.random.normal(scale=30)]
                obsv_pose = np.matmul(true_pose, rand_pose)
                poses.append(obsv_pose)
                obsv_center_y = ((obsv_pose[0, 3] * self.fx) / obsv_pose[2, 3]) + self.cx
                obsv_center_x = ((obsv_pose[1, 3] * self.fy) / obsv_pose[2, 3]) + self.cy
                dia_pixX = ((self.model_dia * self.fx) / obsv_pose[2, 3])
                dia_pixY = ((self.model_dia * self.fy) / obsv_pose[2, 3])

                x_min = int(obsv_center_x - dia_pixX * 0.75)
                x_max = int(obsv_center_x + dia_pixX * 0.75)
                y_min = int(obsv_center_y - dia_pixY * 0.75)
                y_max = int(obsv_center_y + dia_pixY * 0.75)
                print(x_min, y_min, x_max, y_max)

                img_rend = self.render_img(obsv_pose, self.obj_id)
                img_rend = np.pad(img_rend, ((pad_val, pad_val), (pad_val, pad_val), (0, 0)), mode='edge')
                img_rend = img_rend[(x_min+pad_val):(x_max+pad_val), (y_min+pad_val):(y_max+pad_val), :]
                img_obsv = obsv_img[(x_min+pad_val):(x_max+pad_val), (y_min+pad_val):(y_max+pad_val), :]
                img_real = real_img[(x_min+pad_val):(x_max+pad_val), (y_min+pad_val):(y_max+pad_val), :]

                img_rend = cv2.resize(img_rend, self.img_res)
                img_obsv = cv2.resize(img_obsv, self.img_res)
                img_real = cv2.resize(img_real, self.img_res)

                img_input = np.concatenate([img_obsv, img_rend, img_real], axis=1)
                cv2.imwrite('/home/stefan/PADA_viz/img_input.png', img_input)
                cv2.imwrite('/home/stefan/PADA_viz/ori_input.png', obsv_img)

                #if not self.is_testing and np.random.random() > 0.5:
                #        img_A = np.fliplr(img_A)
                #        img_B = np.fliplr(img_B)

                imgs_obsv.append(img_obsv)
                imgs_rend.append(img_rend)
                imgs_real.append(img_real)

            imgs_obsv = np.array(imgs_obsv)/127.5 - 1.
            imgs_rend = np.array(imgs_rend)/127.5 - 1.
            imgs_real = np.array(imgs_real) / 127.5 - 1.
            poses = np.array(poses)

            yield imgs_obsv, imgs_rend, imgs_real, poses

