import torch
import torch.nn as nn

import netron     
import torch.onnx
from torch.autograd import Variable

import numpy as np

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv3d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm3d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv3d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm3d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv3d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv3d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm3d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv3d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv3d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm3d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=2):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)



        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # dropout 是对所有元素以一定的概率随机失活，dropout2d 是对所有通道以一定概率随机失活
        self.dropout = nn.Dropout3d(p=0.4)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, num_class)
        # self.linear3 = nn.Linear(512 + 56 * 5 + 255 +38, num_class)
        self.linear3 = nn.Linear(512 + 255, num_class)

    def forward(self, x, gcn_feature, add_gcn_middle_feature):
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)
        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        feature = self.linear1(x)
        if add_gcn_middle_feature:
            x = torch.cat((feature,gcn_feature),axis=1)
            out = self.linear3(x)
        else:
            out = self.linear2(feature)
        return out,feature

def googlenet():
    return GoogleNet()