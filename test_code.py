from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model.data_loader as data_loader
from scipy import linalg, mat, dot
import os
import matplotlib.pyplot as plt

from utils import get_dataset_label_pt
import utils

fold = 0

# dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 1, data_dir="data/5fold_128<=20mm_aug/fold"+str(fold+1), train_shuffle=False)
# train_dl_save = dataloaders['train']
# test_dl_save = dataloaders['test']
# for i, (x, target, fileName, gcn_middle_feature) in enumerate(test_dl_save):
#     a=1
# fold 0 : 33,62
# fold 1 : 29,127
# fold 2 : 87,130
# fold 3 : 87,148
# fold 4 : 134,139


for fold in range(5):
    gcnFeature = torch.load('data/feature/5fold_128<=20mm_aug/gcn_test_middle_feature_fold_'+str(fold)+'.pt')
    tradFeature = torch.load('data/feature/addition_feature_mask<=20_aug/fold_'+str(fold)+'_test_addition_feature.pt')
    if fold == 0:
        idx = [33,62-1]
    if fold == 1:
        idx = [29,127-1]
    if fold == 2:
        idx = [87,130-1]
    if fold == 3:
        idx = [87,148-1]
    if fold == 4:
        idx = [134,139-1]
    for i in idx:
        gcnFeature = gcnFeature[torch.arange(gcnFeature.size(0))!=i] 
        tradFeature = tradFeature[torch.arange(tradFeature.size(0))!=i] 

    torch.save(gcnFeature,'data/feature/gcn_and_traditional_feature/gcn_test_middle_feature_fold_'+str(fold)+'.pt')
    torch.save(tradFeature,'data/feature/gcn_and_traditional_feature/fold_'+str(fold)+'_test_addition_feature.pt')