import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class losses():
    def __init__(self, dataset_name, img_res=(240, 320)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def foreground_loss(self, , is_testing=False):
        
