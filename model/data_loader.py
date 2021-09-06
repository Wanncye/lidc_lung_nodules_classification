import random
import os

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

import glob

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
tfms_train = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale = (.8, 1)),
    transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.Resize((224, 224)),  # resize the image to 64x64 (remove if images are already 64x64),
    # transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    # transforms.RandomAffine(10, translate=(.1, .1), scale=(.1, .1), shear=.1, resample=False, fillcolor=0),
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
# eval_transformer = transforms.Compose([
#     # transforms.Resize((224, 224)),  # resize the image to 64x64 (remove if images are already 64x64)
#     transforms.ToTensor()])  # transform it into a torch tensor

tfms_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class LIDCDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform, fold, split, add_middle_feature=False):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.data_dir = data_dir
        self.transform = transform
        self.npy_list = os.listdir(data_dir)
        self.npy_list.sort(key= lambda x:int(x[:6]))
        self.fold = fold
        self.add_middle_feature = add_middle_feature
        if self.add_middle_feature:
            self.gcn_middle_feature = torch.load('data/feature/5fold_128<=20mm_aug/gcn_'+split+'_middle_feature_fold_'+str(fold)+'.pt')
            self.gcn_middle_feature.requires_grad = False
            self.addition_feature = torch.load('data/feature/addition_feature_mask<=20_aug/fold_' + str(fold) + '_' + split + '_addition_feature.pt')
            self.addition_feature.requires_grad = False
        

        #对新加进来的feature某些特征做一下归一化，因为这些特征不在同一个数量级上
        # for jndex in range(248,255):
        #     max = self.addition_feature[:, jndex].max()  
        #     min = self.addition_feature[:, jndex].min()  
        #     self.addition_feature[:, jndex] = (self.addition_feature[:, jndex] - min) / (max-min)

    def __len__(self):
        # return size of dataset
        return len(self.npy_list)

    def __getitem__(self, idx):
        filename = self.npy_list[idx]
        cube = np.load(os.path.join(self.data_dir,filename))
        cube = torch.tensor(cube)
        cube = cube.transpose(0,2)
        cube = torch.unsqueeze(cube,0)  #2d卷积的时候把这行注释掉
        cube = cube.type(torch.FloatTensor)
        #非数据增强
        # label = self.npy_list[idx].split('.')[0][-1]
        #数据增强
        label = self.npy_list[idx].split('_')[2][0]
        # print(label)
        label = np.array(int(label))
        label = torch.tensor(label)
        if self.add_middle_feature:
            one_gcn_middle_feature = self.gcn_middle_feature[idx]
            one_addition_feature = self.addition_feature[idx]
            one_feature = torch.cat((one_gcn_middle_feature,one_addition_feature), axis = 0)
        else:
            one_feature = np.zeros((1,255))
        return cube, label, filename, one_feature


def fetch_dataloader(types = ["train"], data_dir = "data/nodules3d_128_mask_npy", df = None, params = None, batch_size = 128, train_shuffle=True, tfms = [], fold = None, add_middle_feature=False):

    print('data_dir:',data_dir)
    dataloaders = {}
    splits = [x for x in types]

    for split in splits:
        if split in types:
            path = data_dir
            path = os.path.join(path,split)
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(LIDCDataset(path, tfms_train, fold, split, add_middle_feature), 
                                        batch_size = batch_size,
                                        shuffle=train_shuffle,
                                        num_workers=0,
                                        pin_memory=False)
            else:
                # dl = DataLoader(SEGMENTATIONDataset(path, eval_transformer, df[df.split.isin([split])]), 
                dl = DataLoader(LIDCDataset(path, tfms_eval, fold, split, add_middle_feature), 
                                batch_size = batch_size,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=False)

            dataloaders[split] = dl

    return dataloaders


class LIDC_N_folder_Dataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, path_list, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.transform = transform
        self.npy_list = []
        for sub_dir in path_list:
            for npy_file in glob.glob(sub_dir + '/*.npy'):
                self.npy_list.append(npy_file)
        


    def __len__(self):
        # return size of dataset
        return len(self.npy_list)

    def __getitem__(self, idx):
        filename = self.npy_list[idx]
        cube = np.load(self.npy_list[idx])
        cube = torch.tensor(cube)
        cube = cube.transpose(0,2)
        cube = torch.unsqueeze(cube,0)
        cube = cube.type(torch.FloatTensor)
        label = self.npy_list[idx].split('.')[0][-1]
        label = np.array(int(label))
        label = torch.tensor(label)
        return cube, label, filename

def fetch_N_folders_dataloader(test_folder, types = ["train"], data_dir = "data/nodules3d_128_npy_5_folders", df = None, params = None, batch_size = 16, tfms = []):

    print('data_dir:',data_dir)
    dataloaders = {}
    splits = [x for x in types]

    for split in splits:
        if split in types:
            path = data_dir
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                path_list = []
                for i in range(5):
                    if i != test_folder:
                        path_list.append(os.path.join(path, str(i))) 
                dl = DataLoader(LIDC_N_folder_Dataset(path_list, tfms_train), 
                                        batch_size = batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True)
            elif split == 'test':
                path_list = [os.path.join(path, str(test_folder))]
                dl = DataLoader(LIDC_N_folder_Dataset(path_list, tfms_eval), 
                                batch_size = batch_size,
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True)

            dataloaders[split] = dl

    return dataloaders