import torch
import torch.nn as nn
import torch.nn.functional as F
import model.data_loader as data_loader
import numpy as np

from scipy import linalg, mat, dot


npy = np.load('data/5fold_128/fold1/test/0001_01_1.npy')
a=1
 