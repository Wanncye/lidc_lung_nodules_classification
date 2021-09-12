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

datasetMean, datasetStd = utils.getDatasetMeanAndStd()
# datasetMean = datasetMean.T.expand(8,2)
# print(datasetMean)
# datasetMean = datasetMean.T.expand(8,2).unsqueeze(-1)
# print(datasetMean)
datasetMean = datasetMean.T.expand(8,2).unsqueeze(-1).expand(8,2,2)
print(datasetMean)


