import torch as t
import numpy as np
import torch.nn.functional as F
 
a = t.randn((4,6))
print(a)
b = F.dropout(a, 0.6, True)
print(b)
