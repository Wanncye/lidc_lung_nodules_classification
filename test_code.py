import torch
import torch.nn as nn
import torch.nn.functional as F
import model.data_loader as data_loader
import numpy as np

from scipy import linalg, mat, dot



def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diagflat(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

a = np.array([[1,2,4],[5,2,1]], dtype=np.float32)
print(normalize_features(a))