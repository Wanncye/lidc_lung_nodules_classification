from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model.data_loader as data_loader
from scipy import linalg, mat, dot
import os
import matplotlib.pyplot as plt

from utils import get_dataset_label_pt

prob = np.array([
    [1,5],
    [5,8],
    [8,6],
    [8,6],
    [6,4],
    [5,7]
])

a = prob[:,1].repeat(2, axis=0).reshape(-1,2) * [0.1,0.2]
print(a)


