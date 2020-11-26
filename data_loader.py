import scipy
from glob import glob
import numpy as np
import ply_loader
import open3d

class DataLoader():
    def __init__(self, batch_size, img_res=(224, 224), is_testing=False):
        self.img_res = img_res
        self.batch_size = batch_size
        self.is_testing = is_testing
        ply_path = '/home/stefan/data/Meshes/linemod_13/obj_02.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.pcd_model = open3d.PointCloud()
        self.pcd_model.points = open3d.Vector3dVector(model_vsd['pts'])

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        print('path to load: ', path)
        batch_images = np.random.choice(path, size=batch_size)

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
            if img.shape[2] <3:
                img = np.repeat(img, 3, axis=2)
            img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)
        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self):
        data_type = "train" if not self.is_testing else "val"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        self.n_batches = int(len(path) / self.batch_size)

        for i in range(self.n_batches-1):
            batch = path[i * self.batch_size:(i+1) * self.batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = self.imread(img)
                h, w, _ = img.shape
                half_w = int(w/2)
                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not self.is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
