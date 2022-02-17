import torch

tensorA = torch.tensor(
    [[1,2,3],
    [3,2,1]]
)
tensorB = torch.tensor(
    [[1,2,3]]
)
print(tensorA * tensorB)