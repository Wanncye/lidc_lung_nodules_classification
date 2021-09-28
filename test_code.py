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

a = [0.955,0.913,0.899,0.874,0.925]
print(np.mean(a))
print(np.std(a))

