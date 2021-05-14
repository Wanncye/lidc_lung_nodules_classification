import torch as t
import numpy as np
import torch.nn.functional as F
 
a = t.randn((4,1,6))
print(a.shape)
a = a.squeeze(1)
print(a.shape)



