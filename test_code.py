import torch as t
import matplotlib
# matplotlib.use('Agg')
import numpy as np
import torch.nn.functional as F

np.random.seed(2021)
adj = t.ones((6, 6))
#将邻接矩阵改一改,随机置1
def rand_adj(adj):
    for i in range(6):
        for j in range(6):
            random_num = np.random.rand()
            if random_num > 0.5:
                adj[i,j] = 0
    return adj
adj = rand_adj(adj)
print(adj)