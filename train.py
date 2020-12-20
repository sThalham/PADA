from __future__ import print_function, division

import scipy
import datetime
import sys
from data_loader import DataLoader
from model import default_model
import numpy as np
import os
import cv2
from glob import glob


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
    train(PAUDA, dataset_path, real_path, mesh_path, mesh_info, epochs=100, batch_size=16)


