from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.data_loader as data_loader
import numpy as np

from scipy import linalg, mat, dot


a = torch.randn((16,512))
b = torch.randn((16,224))
c = torch.cat((a,b),axis=0)
print(c.shape)
 