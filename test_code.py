from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model.data_loader as data_loader
from scipy import linalg, mat, dot
import os
import matplotlib.pyplot as plt

relu = torch.nn.ReLU()
a = torch.tensor([
    [-1,2],
    [5,-1]
])
print(torch.clamp(a,0,3))