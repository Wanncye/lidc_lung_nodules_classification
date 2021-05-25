import torch
import torch.nn as nn
import torch.nn.functional as F
import model.data_loader as data_loader

dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 1, data_dir='data/nodules3d_128_npy', train_shuffle=False)
train_dl = dataloaders['train']
test_dl = dataloaders['test']
mask_dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 1, data_dir='data/nodules3d_128_mask_npy', train_shuffle=False)
mask_train_dl = dataloaders['train']
mask_test_dl = dataloaders['test']

for i, (x, target, _) in enumerate(test_dl):
    if i == 0:
        print(target)
for i, (x, target, _) in enumerate(mask_test_dl):
    if i == 0:
        print(target)