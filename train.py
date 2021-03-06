from __future__ import print_function, division

import scipy
import datetime
import sys
from data_loader import load_data_sample, Dataset, crop_rendering
#from tf_data_generator import TFDataGenerator
from model import default_model
from model_seq import default_model_seq
import numpy as np
import os
import cv2
from glob import glob

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

from multiprocessing import Pool
import multiprocessing
from functools import partial
import copy
import time

bop_renderer_path = '/home/stefan/bop_renderer/build'
sys.path.append(bop_renderer_path)

import bop_renderer

bop_render = bop_renderer.Renderer()
bop_render.init(640, 480)
bop_render.add_object(int(1), '/home/stefan/data/Meshes/lm_models/models/obj_000002.ply')


def render_top_level(lists):
    bop_render.render_object(lists[3], lists[0], lists[1], lists[2][0], lists[2][1], lists[2][2], lists[2][3])
    print('rendering done')
    return bop_render.get_color_image(lists[3])


def get_result(result):
    global results
    results.append(result)


#def render_top_level(ren, pose, intrinsics, obj_id):
#    ren.render_object(obj_id, pose[0], pose[1], intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3])
#    return ren.get_color_image(obj_id)

def train(network, dataset_path, real_path, mesh_path, mesh_info, object_id, epochs, batch_size=1, sample_interval=50):

    dalo = Dataset('train', dataset_path, object_id, real_path, mesh_path, mesh_info, batch_size)
    optimizer = Adam(lr=1e-5, clipnorm=0.001)
    network.compile(loss='mse', optimizer=optimizer)

    multiproc = Pool(1)

    for epoch in range(epochs):
        for batch_i in range(dalo.n_batches):

            start_t = time.time()
            batch = dalo.__get_batch__(batch_i)

            img_list = dalo.__img_list__()
            ann_list = copy.deepcopy(dalo.__anno_list__())
            ia = copy.deepcopy(dalo.__augmenter__())
            intri = copy.deepcopy(dalo.__get_intrinsics__())
            diameter = copy.deepcopy(dalo.__model_diameter__())
            img_res = copy.deepcopy(dalo.__image_shape__())

            parallel_loaded = multiproc.map(partial(load_data_sample, img_list=img_list, anno_list=ann_list, augmenter=ia, intrinsics=intri, img_res=img_res, model_dia=diameter), batch)

            imgs_obsv = []
            imgs_rend = []
            targets = []
            ren_Rot = []
            ren_Tra = []
            bboxes = [] # temp for separate rendering
            for sample in parallel_loaded:
                imgs_obsv.append(sample[0])
                #imgs_rend.append(sample[1])
                targets.append(sample[1])
                ren_Rot.append(sample[2])
                ren_Tra.append(sample[3])
                bboxes.append(sample[4])

            # looping over render
            #for idx, pose in enumerate(extrinsics):
            #    imgs_rend.append(render_crop(obsv_pose=pose, bbox=bboxes[idx], renderer=bop_render, intrinsics=intri, obj_id=object_id,, img_res=img_res))

            # multiproc render and cropping
            #triple_list = []
            #for idx, rot in enumerate(ren_Rot):
            #   triple_list.append([rot, ren_Tra[idx], bboxes[idx]])
            #parallel_rendered = multiproc.map(partial(render_crop, renderer=bop_render, intrinsics=intri, obj_id=object_id, img_res=img_res), triple_list)

            '''
            # multiproc only rendering
            double_list = []
            for idx, rot in enumerate(ren_Rot):
                double_list.append([rot, ren_Tra[idx]])

            light_pose = [np.random.rand() * 2000.0 - 1000.0, np.random.rand() * 2000.0 - 1000.0, 0.0]
            # light_color = [np.random.rand() * 0.1 + 0.9, np.random.rand() * 0.1 + 0.9, np.random.rand() * 0.1 + 0.9]
            light_color = [1.0, 1.0, 1.0]
            light_ambient_weight = np.random.rand()
            light_diffuse_weight = 0.75 + np.random.rand() * 0.25
            light_spec_weight = 0.25 + np.random.rand() * 0.25
            light_spec_shine = np.random.rand() * 3.0

            # time negligible
            bop_render.set_light(light_pose, light_color, light_ambient_weight, light_diffuse_weight, light_spec_weight,
                          light_spec_shine)

            # render + get < 23 ms i5-6600k
            #bop_renderer.render_object(obj_id, R_list, t_list, intri[0], intri[1], intri[2], intri[3])
            parallel_rendered = multiproc.map(partial(render_top_level, ren=bop_render, intrinsics=intri, obj_id=object_id), double_list)
            '''

            quat_list = []
            img_sizes = []

            for idx, rot in enumerate(ren_Rot):
                quat_list.append([rot, ren_Tra[idx], intri, int(object_id)])
                img_sizes.append(img_res)

            print('start rendering')
            full_renderings = multiproc.map(render_top_level, quat_list)
            print('rendering done')

            for img in full_renderings:
                print(img.shape)
            parallel_cropping = multiproc.map(partial(crop_rendering, bbox=bboxes, img_res=img_res), full_renderings)

            imgs_obsv = np.array(imgs_obsv, dtype=np.float32)
            imgs_rend = np.array(parallel_cropping, dtype=np.float32)
            targets = np.array(targets, dtype=np.float32)
            imgs_obsv = imgs_obsv / 127.5 - 1.
            imgs_rend = imgs_rend / 127.5 - 1.
            print('T data preparation: ', time.time()-start_t)

            network.fit(x=[imgs_obsv, imgs_rend],
                        y=targets,
                        batch_size=batch_size,
                        verbose=1,
                        steps_per_epoch=1,
                        # steps_per_epoch=data_generator.__len__(),
                        epochs=1)

            #elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            #print("Epoch %d/%d     Iteration: %d/%d Loss: %f || pose: %f da: %f" % (epoch, epochs,
            #                                                            batch_i, data_loader.n_batches,
            #                                                            g_loss[0] + g_loss[1],
            #                                                            g_loss[0], g_loss[1]))
            #print("Epoch %d/%d     Iteration: %d/%d Loss: %f" % (epoch, epochs, batch_i, data_loader.n_batches, g_loss))

        snapshot_path = './models'
        try:
            os.makedirs(snapshot_path)
        except OSError:
            if not os.path.isdir(snapshot_path):
                raise

        network.save(snapshot_path, 'linemod_{oi}_{{epoch:02d}}.h5'.format(oi=object_id))
    print("Training finished!")


def train_with_generator(network, dataset_path, real_path, mesh_path, mesh_info, object_id, epochs, batch_size=1, sample_interval=50):
    # Parameters

    # Generators
    data_generator = DataGenerator('train', dataset_path, real_path, mesh_path, mesh_info, object_id, batch_size)
    optimizer = Adam(lr=1e-5, clipnorm=0.001)
    network.compile(loss='mse', optimizer=optimizer)

    callbacks = []

    # ensure directory created first; otherwise h5py will error after epoch.
    snapshot_path = './models'
    try:
        os.makedirs(snapshot_path)
    except OSError:
        if not os.path.isdir(snapshot_path):
            raise
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            snapshot_path, 'linemod_{oi}_{{epoch:02d}}.h5'.format(oi=object_id)),
        save_best_only=False,
        save_freq=1,
    )
    callbacks.append(checkpoint)

    # Train model on dataset
    network.fit(x=data_generator,
                batch_size=batch_size,
                          verbose=1,
                          steps_per_epoch=1,
                          #steps_per_epoch=data_generator.__len__(),
                          epochs=epochs,
                          callbacks=callbacks,
                          use_multiprocessing=False,
                          max_queue_size=10)
                          #workers=1)

'''
x_shape = (32, 32, 3)
y_shape = ()  # A single item (not array).
classes = 10

def generator_fn(n_samples):
    """Return a function that takes no arguments and returns a generator."""
    def generator():
        for i in range(n_samples):
            # Synthesize an image and a class label.
            x = np.random.random_sample(x_shape).astype(np.float32)
            y = np.random.randint(0, classes, size=y_shape, dtype=np.int32)
            yield x, y
    return generator

def augment(x, y):
    return x * tf.random.normal(shape=x_shape), y

samples = 10
batch_size = 5
epochs = 2

# Create dataset.
gen = generator_fn(n_samples=samples)
dataset = tf.data.Dataset.from_generator(
    generator=gen, 
    output_types=(np.float32, np.int32), 
    output_shapes=(x_shape, y_shape)
)
# Parallelize the augmentation.
dataset = dataset.map(
    augment, 
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
    # Order does not matter.
    deterministic=False
)
dataset = dataset.batch(batch_size, drop_remainder=True)
# Prefetch some batches.
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
'''

def train_with_data(network, dataset_path, real_path, mesh_path, mesh_info, object_id, epochs, batch_size=1, sample_interval=50):
    # Parameters

    # Generators
    data_generator = TFDataGenerator('train', dataset_path, real_path, mesh_path, mesh_info, object_id, batch_size)

    '''
    # loading and calling
    gen_fn = data_generator.generate_batch()
    dataset = tf.data.Dataset.from_generator(
        generator=gen_fn,
        output_types=((np.float64, np.float64), np.float64),
        output_shapes=(([224, 224, 3], [224, 224, 3]), [5, 5, 7])
    )
    '''
    gen = data_generator.generate_batch()
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=tf.dtypes.int64,
        output_shapes=(1,),
    )
    #print('autotune: ', tf.data.experimental.AUTOTUNE)
    # Parallelize the augmentation.
    dataset = dataset.map(
        data_generator.wrap_tf_function,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        # Order does not matter.
        deterministic=False
    )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    # Prefetch some batches.
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    optimizer = Adam(lr=1e-5, clipnorm=0.001)
    network.compile(loss='mse', optimizer=optimizer)

    callbacks = []

    # ensure directory created first; otherwise h5py will error after epoch.
    snapshot_path = './models'
    try:
        os.makedirs(snapshot_path)
    except OSError:
        if not os.path.isdir(snapshot_path):
            raise
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            snapshot_path, 'linemod_{oi}_{{epoch:02d}}.h5'.format(oi=object_id)),
        save_best_only=False,
        save_freq=1,
    )
    callbacks.append(checkpoint)

    # Train model on dataset
    network.fit(x=dataset.repeat(),
                batch_size=batch_size,
                          verbose=1,
                          steps_per_epoch=data_generator.__len__(),
                          epochs=epochs,
                          callbacks=callbacks,
                          use_multiprocessing=False,
                          max_queue_size=10,
                          workers=3)




def save_model_weights(model, filepath, overwrite=True):
    cwd = os.getcwd()
    filepath = filepath + '.h5'
    model_path = os.path.join(cwd, 'saved_model', filepath)
    if os.path.isfile(model_path):
        model.save(model_path, overwrite=True)
    else:
        model.save_weights(model_path, overwrite=False)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    PAUDA = default_model()
    dataset_path = '/home/stefan/data/train_data/linemod_PBR_BOP'
    mesh_path = '/home/stefan/data/Meshes/lm_models/models/obj_000002.ply'
    mesh_info = '/home/stefan/data/Meshes/lm_models/models/models_info.json'
    real_path = '/home/stefan/data/datasets/cocoval2017'
    object_id = str(1)
    #train_with_data(PAUDA, dataset_path, real_path, mesh_path, mesh_info, object_id, epochs=100, batch_size=32)
    train(PAUDA, dataset_path, real_path, mesh_path, mesh_info, object_id, epochs=100, batch_size=32)


