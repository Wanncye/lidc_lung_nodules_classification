from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn

num = 256

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv3d(1, 64, kernel_size=3, stride=1)  #in_channel由3改成了1
        self.c1 = nn.Conv3d(64, 128, kernel_size=3, stride=1)
        self.c2 = nn.Conv3d(128, 256, kernel_size=3, stride=1)
        self.c3 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.l1 = nn.Linear(512*61*61, num)

        self.b1 = nn.BatchNorm3d(128)
        self.b2 = nn.BatchNorm3d(256)
        self.b3 = nn.BatchNorm3d(512)

    def forward(self, x):
        h = F.relu(self.c0(x))  #16*64*6*126*126
        features = F.relu(self.b1(self.c1(h))) #16*128*4*124*124
        h = F.relu(self.b2(self.c2(features))) #16*256*2*122*122
        h = F.relu(self.b3(self.c3(h)))         #16, 512, 1, 61, 61
        encoded = self.l1(h.view(x.shape[0], -1))
        return encoded, features


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.c1 = nn.Conv3d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32*2*122*122 + num, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M)) #M16, 128, 4, 124, 124
        h = self.c1(h)#16,32,2,122,122
        h = h.view(y.shape[0], -1) # #16,32*2*122*122
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv3d(128 + num, 512, kernel_size=1)
        self.c1 = nn.Conv3d(512, 512, kernel_size=1)
        self.c2 = nn.Conv3d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(num, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class Classifier(nn.Module):
    def __init__(self,class_num=2):
        super().__init__()
        self.l1 = nn.Linear(num, 15)
        self.bn1 = nn.BatchNorm1d(15)
        self.l2 = nn.Linear(15, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.l3 = nn.Linear(10, class_num)
        self.bn3 = nn.BatchNorm1d(class_num)

    def forward(self, x):
        encoded, _ = x[0], x[1]
        clazz = F.relu(self.bn1(self.l1(encoded)))
        clazz = F.relu(self.bn2(self.l2(clazz)))
        clazz = F.softmax(self.bn3(self.l3(clazz)), dim=1)
        return clazz


class DeepInfoAsLatent(nn.Module):
    def __init__(self, run):
        super().__init__()
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load(str(run)))
        self.classifier = Classifier()

    def forward(self, x):
        z, features = self.encoder(x)
        z = z.detach()
        return self.classifier((z, features))