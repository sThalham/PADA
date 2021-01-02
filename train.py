from __future__ import print_function, division

import scipy
import datetime
import sys
from data_loader import DataLoader
from tf_data_generator import TFDataGenerator
from model import default_model
from model_seq import default_model_seq
import numpy as np
import os
import cv2
from glob import glob

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf


def train(network, dataset_path, real_path, mesh_path, mesh_info, epochs, batch_size=1, sample_interval=50):

    #model_file = Path(self.model_name)
    #if model_file.is_file():
    #    print("MODEL EXISTS... skip training. Please delete model file to retrain")
    #    self.combined.load_weights(self.model_name)
    #    return

    data_loader = DataLoader(dataset_path, real_path, mesh_path, mesh_info, batch_size)

    for epoch in range(epochs):
        #for batch_i, (obsv, rend, real, delta_gt, da_gt) in enumerate(data_loader.load_batch()):
        for batch_i, (obsv, rend, delta) in enumerate(data_loader.load_batch()):

            #g_loss = network.model.train_on_batch([obsv, rend, real], [delta_gt, da_gt])
            g_loss = network.model.train_on_batch([obsv, rend], delta)

            #elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            #print("Epoch %d/%d     Iteration: %d/%d Loss: %f || pose: %f da: %f" % (epoch, epochs,
            #                                                            batch_i, data_loader.n_batches,
            #                                                            g_loss[0] + g_loss[1],
            #                                                            g_loss[0], g_loss[1]))
            print("Epoch %d/%d     Iteration: %d/%d Loss: %f" % (epoch, epochs, batch_i, data_loader.n_batches, g_loss))

        save_model_weights(network.model, 'linemod_' + str(epoch))
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
    train_with_data(PAUDA, dataset_path, real_path, mesh_path, mesh_info, object_id, epochs=100, batch_size=32)


