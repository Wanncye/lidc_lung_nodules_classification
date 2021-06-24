import torch
import torch.nn as nn
import torch.nn.functional as F
import model.data_loader as data_loader
import numpy as np

from scipy import linalg, mat, dot



adj = np.ones((4,4))
#将邻接矩阵改一改,随机置1
for i in range(4):
    for j in range(4):
        random_num = np.random.rand()
        print(random_num)
        if random_num > 0.5:
            adj[i,j] = 0
print(adj)