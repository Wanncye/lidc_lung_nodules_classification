import enum
from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import model.data_loader as data_loader
from scipy import linalg, mat, dot
import os
import matplotlib.pyplot as plt

from utils import get_dataset_label_pt,get_matrix_similarity
import utils
import json

output_batch = torch.tensor([[1.593,2.245],[2.589,1.203]])
m = nn.Softmax(dim=1)
probability = m(output_batch)
print(probability)
predict = np.argmax(output_batch, axis=1)
print(predict)
print(probability[:, 1])



# for fold in range(5):
#     dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 3000, data_dir="data/5fold_128<=20mm_aug/fold"+str(fold+1), train_shuffle=False)
#     train_dl = dataloaders['train']
#     path = 'fold_'+str(fold)+'_train_order.json'
#     fp = open(path,'w')
#     for dataloader_index, (data_batch, labels_batch, filename, one_feature) in enumerate(train_dl):
#         dicts = {}
#         for index,name in enumerate(filename):
#             dicts[index] =  name
#         dicts = json.dump(dicts,fp)

# for fold in range(2):
#     tradFeature = torch.load('data/feature/addition_feature_mask<=20_aug/fold_'+str(fold)+'_test_addition_feature.pt')
#     if fold == 0:
#         idx = [33,62-1]
#     if fold == 1:
#         idx = [29,127-1]
#     if fold == 2:
#         idx = [87,130-1]
#     if fold == 3:
#         idx = [87,148-1]
#     if fold == 4:
#         idx = [134,139-1]
#     for i in idx:
#         tradFeature = tradFeature[torch.arange(tradFeature.size(0))!=i] 

#     torch.save(tradFeature,'data/feature/gcn_and_traditional_feature/fold_'+str(fold)+'_test_addition_feature.pt')



# from sklearn.model_selection import StratifiedKFold,KFold
# from glob import glob
# import os
# import shutil
# data_path = 'data/all_nodule/'
# skf = StratifiedKFold(n_splits=10)
# data_list = []
# label_list = []
# pos_list = glob(os.path.join(data_path,"*_1.npy"))
# neg_list = glob(os.path.join(data_path,"*_0.npy"))
# for pos in pos_list:
#     data_list.append(pos)
#     label_list.append(1)
# for neg in neg_list:
#     data_list.append(neg)
#     label_list.append(0)

# def mycopyfile(srcfile,dstpath):                       # 复制函数
#     if not os.path.isfile(srcfile):
#         print ("%s not exist!"%(srcfile))
#     else:
#         fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
#         if not os.path.exists(dstpath):
#             os.makedirs(dstpath)                       # 创建路径
#         shutil.copy(srcfile, dstpath + fname)          # 复制文件
#         # print ("copy %s -> %s"%(srcfile, dstpath + fname))


# folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2021).split(data_list,label_list)
# for fold, (trn_idx, val_idx) in enumerate(folds):
#         train_data_list = []

#         test_data_list = []
#         for i in trn_idx:
#             train_data_list.append(data_list[i])
#         for i in val_idx:
#             test_data_list.append(data_list[i])
#         # if  os.path.exists('data/10fold/'+str(fold))==False:
#         #     os.makedirs('data/10fold/fold'+str(fold+1)+'/train')
#         #     os.makedirs('data/10fold/fold'+str(fold+1)+'/test')

#         src_dir = './'
#         train_dst_dir = 'data/10fold/fold'+str(fold+1)+'/train/'
#         test_dst_dir = 'data/10fold/fold'+str(fold+1)+'/test/'
                       
#         for srcfile in train_data_list:
#             mycopyfile(srcfile, train_dst_dir)   
#         for srcfile in test_data_list:
#             mycopyfile(srcfile, test_dst_dir)