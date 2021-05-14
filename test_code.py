import torch as t
import numpy as np
import torch.nn.functional as F

np.random.seed(2021)
a = t.zeros((6,6))
print(a)
for i in range(6):
    for j in range(6):
        random_num = np.random.rand()
        if random_num > 0.5:
            a[i,j] = 1
print(a)




