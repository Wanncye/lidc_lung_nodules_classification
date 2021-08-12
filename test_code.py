from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model.data_loader as data_loader
from scipy import linalg, mat, dot
import os
import matplotlib.pyplot as plt

data_dir = 'data/5fold_128/fold3/train'
print(os.listdir(data_dir))