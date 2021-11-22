import json
import logging
import os
import shutil
from matplotlib import cm
from skimage.feature import greycomatrix, greycoprops

import torch
import glob
from tqdm.std import tqdm

import csv
from model.threeDresnet import generate_model
from model.threeDGoogleNet import googlenet
from model.threeDVGG import vgg16_bn, vgg11_bn, vgg13_bn, vgg19_bn
from model.threeDDensenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201

import visdom
import time
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt 
import re
from torch.autograd import Variable

import torch 
import model.data_loader as data_loader
from tqdm import tqdm
# torch.cuda.set_device(0)
from matplotlib import pyplot as plt

from model.graphNet import GAT,GCN
import warnings
import logging
warnings.filterwarnings('ignore')

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    if os.path.exists(log_path):
        print('exist log file,deleting...')
        os.remove(log_path)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint, N_folder, params, descript):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'folder.'+ str(N_folder)+ '.' +params.loss +'_alpha_'+str(params.FocalLossAlpha)+descript+'.best.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    if is_best:
        torch.save(state, filepath)
    return filepath
        


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        color = 1
        for k, v in d.items():
            self.plot(k, v, color)
            color += 1 

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, color, **kwargs):
        """
        self.plot('loss',1.00)
        """
        if color == 1:
            lc =  np.array([ [255, 0, 0], ])
        elif color == 2:
            lc =  np.array([ [0, 255, 0], ])
        elif color == 3:
            lc =  np.array([ [0, 0, 255], ])
        else:
            lc =  np.array([ [255, 255, 0], ])

        x = self.index.get(name, 0)

        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name,linecolor = lc ),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


def split_data_to_5folders(data_path,dest_path):
    ## 思路：取得所有结节，将良性分为5份，恶性分为5份，然后按顺序结合这些集合，得到5份
    npy_list = glob.glob(data_path+'/*/*.npy')
    benign = []
    malignancy = []
    for npy in npy_list:
        if npy.split('.')[1].split('_')[-1] == '0':
            benign.append(npy)
        elif npy.split('.')[1].split('_')[-1] == '1':
            malignancy.append(npy)
    benign1 = benign[:97]
    benign2 = benign[97:194]
    benign3 = benign[194:291]
    benign4 = benign[291:388]
    benign5 = benign[388:]
    malignancy1 = malignancy[:63]
    malignancy2 = malignancy[63:126]
    malignancy3 = malignancy[126:189]
    malignancy4 = malignancy[189:252]
    malignancy5 = malignancy[252:]
    folder1 = benign1 + malignancy1
    folder2 = benign2 + malignancy2
    folder3 = benign3 + malignancy3
    folder4 = benign4 + malignancy4
    folder5 = benign5 + malignancy5
    folder = [folder1,folder2,folder3,folder4,folder5]
    for index,one_folder in enumerate(folder):
        if not os.path.exists(os.path.join(dest_path,str(index))):
            os.makedirs(os.path.join(dest_path,str(index)))
        for npy in one_folder:
            npy_name = npy.split('/')[-1]
            shutil.copyfile(npy, os.path.join(os.path.join(dest_path,str(index)), npy_name))



def get_metrics(target, pred):
    prec, recall, _, _ = metrics.precision_recall_fscore_support(target, pred>0.5, average='binary')
    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label = 1)
    auc = metrics.auc(fpr, tpr)
    return prec, recall, auc

def get_specificity(target, pred):
    TN = 0
    FP = 0
    for index, one_label in enumerate(target):
        if one_label == 0 and pred[index] < 0.5:
            TN += 1
        elif one_label == 0 and pred[index] > 0.5:
            FP += 1
    specificity = TN / (TN + FP)
    return specificity

def plot_figure(png_dir, log_file):
    f = open(log_file, 'r')
    lines = f.readlines()
    metric_dic = {}
    for line in lines:
        if 'folder' in line:
            folder_num = line.split(' ')[3][0]
            metric_dic[folder_num] = {
                'TrainLoss': [0 for i in range(100)],
                'TrainAcc': [0 for i in range(100)],
                'TestAcc': [0 for i in range(100)],
                'Specificity': [0 for i in range(100)],
                'Sensitivity': [0 for i in range(100)],
                'AUC': [0 for i in range(100)],
                'Precision': [0 for i in range(100)],
            }
        elif 'Epoch' in line:
            epoch_num = int(line.split(' ')[-1].split('/')[0])
        elif 'Train metrics' in line:
            metric_dic[folder_num]['TrainLoss'][epoch_num-1] = float(line.split(' ')[9])
            metric_dic[folder_num]['TrainAcc'][epoch_num-1] = float(line.split(' ')[6])
        elif 'Eval metrics' in line:
            metric_dic[folder_num]['TestAcc'][epoch_num-1] = float(line.split(' ')[7])
        elif 'precision' in line:
            metric_dic[folder_num]['Specificity'][epoch_num-1] = float(line.split(' ')[-1])
            metric_dic[folder_num]['Sensitivity'][epoch_num-1] = float(line.split(')')[1].split(' ')[1])
            metric_dic[folder_num]['Precision'][epoch_num-1] = float(line.split('precision: ')[1].split(' ')[0])
            metric_dic[folder_num]['AUC'][epoch_num-1] = float(line.split('auc: ')[1].split(' ')[0])

    #将5折TrainLoss放在一个图上
    folder = ['0', '1', '2', '3', '4']
    for key in metric_dic['0']:
        max_value = 0
        max_indx = 0
        max_folder = 0
        for index in folder:
            plt.plot([i for i in range(100)], metric_dic[index][key], label = 'folder_'+index)
            temp_max_indx=np.argmax(metric_dic[index][key])
            temp_max_value = metric_dic[index][key][temp_max_indx]
            if temp_max_value > max_value:
                max_value = temp_max_value
                max_indx = temp_max_indx
                max_folder = int(index)
        show_max='['+ str(max_folder) + ' ' + str(max_indx)+' '+str(metric_dic[str(max_folder)][key][max_indx])+']'
        plt.annotate(show_max,xytext=(max_indx,metric_dic[index][key][max_indx]),xy=(max_indx,metric_dic[str(max_folder)][key][max_indx]))
        plt.title(key)
        plt.legend()
        plt.savefig(os.path.join(png_dir, 'FocalLoss_alpha_0.25_correct-alpha_' + key + '.png'))
        plt.cla()

    #计算50个epoch之后5折平均最大准确率，为什么选50，因为50epoch之后，模型在训练集上趋于收敛
    # GoogLeNet 50 收敛
    # ResNet 43 收敛
    # VGG16  0  有一些较为棘手的问题，因此从0开始
    coverage_epoch = 20
    average_dic = {}
    for key in metric_dic['0']: #这个循环时得到所有评价指标
        var_name = 'average_' + key
        average_dic[var_name] = [0 for i in range(100-coverage_epoch)]
        for index in folder: #这个循环是计算一个指标的平均值
            average_dic[var_name] = [average_dic[var_name][i] + metric_dic[index][key][coverage_epoch:][i] for i in range(100-coverage_epoch)]
        average_dic[var_name] = [i/5 for i in average_dic[var_name]]
    for key in average_dic:
        plt.plot([i+coverage_epoch for i in range(100-coverage_epoch)], average_dic[key], 'r-o', label = key)
        array = np.array(average_dic[key])
        max_indx=np.argmax(array)
        min_indx=np.argmin(array)
        show_max='['+str(max_indx+coverage_epoch)+' '+str(array[max_indx])+']'
        plt.annotate(show_max,xytext=(max_indx+coverage_epoch,array[max_indx]),xy=(max_indx+coverage_epoch,array[max_indx]))
        show_min='['+str(min_indx+coverage_epoch)+' '+str(array[min_indx])+']'
        plt.annotate(show_min,xytext=(min_indx+coverage_epoch,array[min_indx]),xy=(min_indx+coverage_epoch,array[min_indx]))
        plt.title(key)
        plt.legend()
        plt.savefig(os.path.join(png_dir, 'FocalLoss_alpha_0.25_correct-alpha_' + key + '.png'))
        plt.cla()

#特征提取辅助函数
def binary_to_decimal(array):
    my_str = ''
    for element in array:
        my_str += str(element)
    return int(my_str, 2)

#特征提取辅助函数
def normalization(array):
    array = np.array(array, dtype=np.float)
    array_max = np.max(array)
    array_min = np.min(array)
    distribute = array_max - array_min
    for index in range(len(array)):
        array[index] = float(array[index] - array_min) / float(distribute)
    return array

#一个结节的3d LBP特征
def LBP(cube):
    cube = cube.squeeze(0).squeeze(0)
    feature=[]
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            for k in range(cube.shape[2]):
                middle_pixel = cube[i, j, k]
                binary_code = np.zeros(26, dtype=int)
                if i-1 < 0 or j-1 < 0 or k-1 < 0 or i+1 > cube.shape[0]-1 or j+1 > cube.shape[1]-1 or k+1 > cube.shape[2]-1:
                    #属于cube边缘的像素,可以什么都不做，因为结节图片边缘像素没什么价值
                    if i-1 < 0:
                        if j-1 < 0 or k-1 < 0  or j+1 > cube.shape[1]-1 or k+1 > cube.shape[2]-1:
                            pass
                        else:
                            #第一张切片
                            if cube[i,j-1,k-1] > middle_pixel: binary_code[9] = 1
                            if cube[i,j,k-1] > middle_pixel: binary_code[10] = 1
                            if cube[i,j+1,k-1] > middle_pixel: binary_code[11] = 1
                            if cube[i,j-1,k] > middle_pixel: binary_code[12] = 1
                            if cube[i,j+1,k] > middle_pixel: binary_code[13] = 1
                            if cube[i,j-1,k+1] > middle_pixel: binary_code[14] = 1
                            if cube[i,j,k+1] > middle_pixel: binary_code[15] = 1
                            if cube[i,j+1,k+1] > middle_pixel: binary_code[16] = 1
                            if cube[i+1,j-1,k-1] > middle_pixel: binary_code[17] = 1
                            if cube[i+1,j,k-1] > middle_pixel: binary_code[18] = 1
                            if cube[i+1,j+1,k-1] > middle_pixel: binary_code[19] = 1
                            if cube[i+1,j-1,k] > middle_pixel: binary_code[20] = 1
                            if cube[i+1,j,k] > middle_pixel: binary_code[21] = 1
                            if cube[i+1,j+1,k] > middle_pixel: binary_code[22] = 1
                            if cube[i+1,j-1,k+1] > middle_pixel: binary_code[23] = 1
                            if cube[i+1,j,k+1] > middle_pixel: binary_code[24] = 1
                            if cube[i+1,j+1,k+1] > middle_pixel: binary_code[25] = 1
                            feature.append(binary_to_decimal(binary_code))
                    if i+1 > cube.shape[0]-1:
                        if j-1 < 0 or k-1 < 0  or j+1 > cube.shape[1]-1 or k+1 > cube.shape[2]-1:
                            pass
                        else:
                            #最后一张切片
                            if cube[i-1,j-1,k-1] > middle_pixel: binary_code[0] = 1
                            if cube[i-1,j,k-1] > middle_pixel: binary_code[1] = 1
                            if cube[i-1,j+1,k-1] > middle_pixel: binary_code[2] = 1
                            if cube[i-1,j-1,k] > middle_pixel: binary_code[3] = 1
                            if cube[i-1,j,k] > middle_pixel: binary_code[4] = 1
                            if cube[i-1,j+1,k] > middle_pixel: binary_code[5] = 1
                            if cube[i-1,j-1,k+1] > middle_pixel: binary_code[6] = 1
                            if cube[i-1,j,k+1] > middle_pixel: binary_code[7] = 1
                            if cube[i-1,j+1,k+1] > middle_pixel: binary_code[8] = 1
                            if cube[i,j-1,k-1] > middle_pixel: binary_code[9] = 1
                            if cube[i,j,k-1] > middle_pixel: binary_code[10] = 1
                            if cube[i,j+1,k-1] > middle_pixel: binary_code[11] = 1
                            if cube[i,j-1,k] > middle_pixel: binary_code[12] = 1
                            if cube[i,j+1,k] > middle_pixel: binary_code[13] = 1
                            if cube[i,j-1,k+1] > middle_pixel: binary_code[14] = 1
                            if cube[i,j,k+1] > middle_pixel: binary_code[15] = 1
                            if cube[i,j+1,k+1] > middle_pixel: binary_code[16] = 1
                            feature.append(binary_to_decimal(binary_code))
                else:
                    if cube[i-1,j-1,k-1] > middle_pixel: binary_code[0] = 1
                    if cube[i-1,j,k-1] > middle_pixel: binary_code[1] = 1
                    if cube[i-1,j+1,k-1] > middle_pixel: binary_code[2] = 1
                    if cube[i-1,j-1,k] > middle_pixel: binary_code[3] = 1
                    if cube[i-1,j,k] > middle_pixel: binary_code[4] = 1
                    if cube[i-1,j+1,k] > middle_pixel: binary_code[5] = 1
                    if cube[i-1,j-1,k+1] > middle_pixel: binary_code[6] = 1
                    if cube[i-1,j,k+1] > middle_pixel: binary_code[7] = 1
                    if cube[i-1,j+1,k+1] > middle_pixel: binary_code[8] = 1
                    if cube[i,j-1,k-1] > middle_pixel: binary_code[9] = 1
                    if cube[i,j,k-1] > middle_pixel: binary_code[10] = 1
                    if cube[i,j+1,k-1] > middle_pixel: binary_code[11] = 1
                    if cube[i,j-1,k] > middle_pixel: binary_code[12] = 1
                    if cube[i,j+1,k] > middle_pixel: binary_code[13] = 1
                    if cube[i,j-1,k+1] > middle_pixel: binary_code[14] = 1
                    if cube[i,j,k+1] > middle_pixel: binary_code[15] = 1
                    if cube[i,j+1,k+1] > middle_pixel: binary_code[16] = 1
                    if cube[i+1,j-1,k-1] > middle_pixel: binary_code[17] = 1
                    if cube[i+1,j,k-1] > middle_pixel: binary_code[18] = 1
                    if cube[i+1,j+1,k-1] > middle_pixel: binary_code[19] = 1
                    if cube[i+1,j-1,k] > middle_pixel: binary_code[20] = 1
                    if cube[i+1,j,k] > middle_pixel: binary_code[21] = 1
                    if cube[i+1,j+1,k] > middle_pixel: binary_code[22] = 1
                    if cube[i+1,j-1,k+1] > middle_pixel: binary_code[23] = 1
                    if cube[i+1,j,k+1] > middle_pixel: binary_code[24] = 1
                    if cube[i+1,j+1,k+1] > middle_pixel: binary_code[25] = 1
                    feature.append(binary_to_decimal(binary_code))

    max_bins = int(max(feature) + 1)
    nor_feature, _ = np.histogram(feature, density=False, bins=64, range=(0, max_bins))
    nor_feature = normalization(nor_feature)
    return nor_feature
    
def HOG(cube):
    cube = cube.squeeze(0).squeeze(0)
    all_gredient = []
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            for k in range(cube.shape[2]):
                if i-1 < 0 or j-1 < 0 or k-1 < 0 or i+1 > cube.shape[0]-1 or j+1 > cube.shape[1]-1 or k+1 > cube.shape[2]-1:
                    #属于cube边缘的像素,可以什么都不做，因为结节图片边缘像素没什么价值
                    if i-1 < 0:
                        if j-1 < 0 or k-1 < 0  or j+1 > cube.shape[1]-1 or k+1 > cube.shape[2]-1:
                            pass
                        else:
                            dis_x = cube[i, j+1, k] - cube[i, j-1, k]
                            dis_y = cube[i, j, k+1] - cube[i, j, k-1]
                            dis_z = cube[i+1, j, k] - 0
                            pixel_gradient = np.sqrt(np.square(dis_x) + np.square(dis_y) + np.square(dis_z)) 
                            all_gredient.append(pixel_gradient)
                    if i+1 > cube.shape[0]-1:
                        if j-1 < 0 or k-1 < 0  or j+1 > cube.shape[1]-1 or k+1 > cube.shape[2]-1:
                            pass
                        else:
                            dis_x = cube[i, j+1, k] - cube[i, j-1, k]
                            dis_y = cube[i, j, k+1] - cube[i, j, k-1]
                            dis_z = 0 - cube[i-1, j, k]
                            pixel_gradient = np.sqrt(np.square(dis_x) + np.square(dis_y) + np.square(dis_z)) 
                            all_gredient.append(pixel_gradient)
                else:#中间层像素,介于不知道怎么把方向融合进来，这里就只统计像素点的梯度信息，就像LBP一样
                    dis_x = cube[i, j+1, k] - cube[i, j-1, k]
                    dis_y = cube[i, j, k+1] - cube[i, j, k-1]
                    dis_z = cube[i+1, j, k] - cube[i-1, j, k]
                    pixel_gradient = np.sqrt(np.square(dis_x) + np.square(dis_y) + np.square(dis_z)) 
                    all_gredient.append(pixel_gradient)
                    # pixel_direcet = math.degrees(math.atan(dis_y/dis_x))
                    # print(pixel_direcet)

    gredient_max = np.max(all_gredient)
    gredient_min = np.min(all_gredient)
    nor_feature, _ = np.histogram(all_gredient, density=True, bins=64, range=(gredient_min, gredient_max))
    nor_feature = normalization(nor_feature)
    return nor_feature

def GLCM(cube):
    cube = cube.squeeze(0).squeeze(0)
    feature = np.zeros(64)
    for index in range(8):
        img = torch.tensor(cube[index],dtype=torch.int)
        img = torch.clamp(img,0,255)
        # glcm = greycomatrix(img,            #共生矩阵（代码中glcm）为四维，前两维表示行列，后两维分别表示距离和角度
        #                     [1, 2, 4, 8],      #距离
        #                     [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],  #方向
        #                     256,        #灰度级别
        #                     symmetric=True,     #是否对称
        #                     normed=True)        #是否标准化
        # attribute = ['contrast', 'dissimilarity', 'energy', 'correlation']

        glcm = greycomatrix(img,            #共生矩阵（代码中glcm）为四维，前两维表示行列，后两维分别表示距离和角度
                            [1, 2],      #距离
                            [0, np.pi / 2],  #方向
                            256,        #灰度级别
                            symmetric=True,     #是否对称
                            normed=True)        #是否标准化
        attribute = ['contrast', 'correlation']

        for jndex, prop in enumerate(attribute):
            temp = greycoprops(glcm, prop).flatten()      #输出的每种特征（代码中temp）行表示距离，列表示角度。
            feature[index*8 + jndex*4 : index*8 + (jndex+1)*4] = temp
    return normalization(feature)

import cv2
def HU(cube):
    cube = cube.squeeze(0).squeeze(0)
    feature = np.zeros(56)
    for index in range(8):
        moments = cv2.moments(cube[index].numpy())
        hu_moments = cv2.HuMoments(moments)
        feature[index*len(hu_moments):(index+1)*len(hu_moments)] = hu_moments.flatten()
    return normalization(feature)

def feature_extract():
    # 提取两种类型的特征：一种是非mask的结节特征，另一种是mask的结节特征。
    # 共提取6种特征，即resnet,vgg,googlenet,lbp,hog,glcm
    batch_size = 1
    dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = batch_size, data_dir='./data/nodules3d_128_npy_no_same_patient_in_two_dataset', train_shuffle=False)
    train_dl = dataloaders['train']
    train_len = 641
    test_dl = dataloaders['test']
    test_len = 157
    feature_len = 512
    lbp_train_feature = torch.zeros((train_len,feature_len))
    hog_train_feature = torch.zeros((train_len,feature_len))
    glcm_train_feature = torch.zeros((train_len,feature_len))
    lbp_test_feature = torch.zeros((test_len,feature_len))
    hog_test_feature = torch.zeros((test_len,feature_len))
    glcm_test_feature = torch.zeros((test_len,feature_len))

    # # #googlenet提取特征
    # model = googlenet()
    # #strict=False 使得预训练模型参数中和新模型对应上的参数会被载入，对应不上或没有的参数被抛弃
    # model.load_state_dict(torch.load('./experiments/googlenet_mask/folder.0.FocalLoss_alpha_0.25.best.pth.tar'), strict=False)
    with tqdm(total=len(train_dl)) as t:
        for i, (x, target, _) in enumerate(train_dl):
            feature = LBP(x)
            lbp_train_feature[(i*batch_size):((i+1)*batch_size), :] = torch.from_numpy(feature)
            feature = HOG(x)
            hog_train_feature[(i*batch_size):((i+1)*batch_size), :] = torch.from_numpy(feature)
            feature = GLCM(x)
            glcm_train_feature[(i*batch_size):((i+1)*batch_size), :] = torch.from_numpy(feature)
            t.update()
    
    with tqdm(total=len(test_dl)) as t:
        for i, (x, target, _) in enumerate(test_dl):
            feature = LBP(x)
            lbp_test_feature[(i*batch_size):((i+1)*batch_size), :] = torch.from_numpy(feature)
            feature = HOG(x)
            hog_test_feature[(i*batch_size):((i+1)*batch_size), :] = torch.from_numpy(feature)
            feature = GLCM(x)
            glcm_test_feature[(i*batch_size):((i+1)*batch_size), :] = torch.from_numpy(feature)
            t.update()

    torch.save(lbp_train_feature,'./data/feature/lbp_train_feature.pt')
    torch.save(hog_train_feature,'./data/feature/hog_train_feature.pt')
    torch.save(glcm_train_feature,'./data/feature/glcm_train_feature.pt')
    torch.save(lbp_test_feature,'./data/feature/lbp_test_feature.pt')
    torch.save(hog_test_feature,'./data/feature/hog_test_feature.pt')
    torch.save(glcm_test_feature,'./data/feature/glcm_test_feature.pt')

def numpy_to_tensor_and_save():
    hog_train_feature = torch.from_numpy(np.load('./data/feature/hog_train_feature.pt.npy'))
    hog_test_feature = torch.from_numpy(np.load('./data/feature/hog_test_feature.pt.npy'))
    lbp_train_feature = torch.from_numpy(np.load('./data/feature/lbp_train_feature.pt.npy'))
    lbp_test_feature = torch.from_numpy(np.load('./data/feature/lbp_test_feature.pt.npy'))
    torch.save(hog_train_feature,'./data/feature/hog_train_feature.pt')
    torch.save(hog_test_feature,'./data/feature/hog_test_feature.pt')
    torch.save(lbp_train_feature,'./data/feature/lbp_train_feature.pt')
    torch.save(lbp_test_feature,'./data/feature/lbp_test_feature.pt')

#对6种特征使用svm进行分类,并将最好的预测结果存储在csv文件中
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
def svm_classification():
    # googlenet_train_feature = torch.load('./data/feature/googlenet_train_feature.pt')
    # googlenet_test_feature = torch.load('./data/feature/googlenet_test_feature.pt')
    # resnet_train_feature = torch.load('./data/feature/resnet_train_feature.pt')
    # resnet_test_feature = torch.load('./data/feature/resnet_test_feature.pt')
    # vgg_train_feature = torch.load('./data/feature/vgg_train_feature.pt')
    # vgg_test_feature = torch.load('./data/feature/vgg_test_feature.pt')
    # hog_train_feature = torch.load('./data/feature/hog_train_feature.pt')
    # hog_test_feature = torch.load('./data/feature/hog_test_feature.pt')
    # lbp_train_feature = torch.load('./data/feature/lbp_train_feature.pt')
    # lbp_test_feature = torch.load('./data/feature/lbp_test_feature.pt')
    # glcm_train_feature = torch.load('./data/feature/glcm_train_feature.pt')
    # glcm_test_feature = torch.load('./data/feature/glcm_test_feature.pt')
    # train_label = torch.load('./data/feature/train_label.pt')
    # test_label = torch.load('./data/feature/test_label.pt')

    googlenet_train_feature = torch.load('./data/mask_feature/googlenet_train_feature.pt')
    googlenet_test_feature = torch.load('./data/mask_feature/googlenet_test_feature.pt')
    resnet_train_feature = torch.load('./data/mask_feature/resnet_train_feature.pt')
    resnet_test_feature = torch.load('./data/mask_feature/resnet_test_feature.pt')
    vgg_train_feature = torch.load('./data/mask_feature/vgg_train_feature.pt')
    vgg_test_feature = torch.load('./data/mask_feature/vgg_test_feature.pt')
    hog_train_feature = torch.load('./data/mask_feature/hog_train_feature.pt')
    hog_test_feature = torch.load('./data/mask_feature/hog_test_feature.pt')
    lbp_train_feature = torch.load('./data/mask_feature/lbp_train_feature.pt')
    lbp_test_feature = torch.load('./data/mask_feature/lbp_test_feature.pt')
    glcm_train_feature = torch.load('./data/mask_feature/glcm_train_feature.pt')
    glcm_test_feature = torch.load('./data/mask_feature/glcm_test_feature.pt')
    train_label = torch.load('./data/mask_feature/train_label.pt')
    test_label = torch.load('./data/mask_feature/test_label.pt')

    batch_size = 1
    dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = batch_size, data_dir='./data/nodules3d_128_mask_npy', train_shuffle=False)
    test_dl = dataloaders['test']
    file_name_list = []
    for i, (x, target, file_name) in enumerate(test_dl):
        file_name_list.append(file_name[0])

    #glcm竖直方向上归一化
    glcm_train_feature = glcm_train_feature.transpose(0,1)
    glcm_test_feature = glcm_test_feature.transpose(0,1)
    for index in range(len(glcm_train_feature)):
        max = glcm_train_feature[index].max()  #170
        min = glcm_train_feature[index].min()  #1.88
        glcm_train_feature[index] = (glcm_train_feature[index] - min) / (max-min)
    for index in range(len(glcm_train_feature)):
        max = glcm_test_feature[index].max()  #170
        min = glcm_test_feature[index].min()  #1.88
        glcm_test_feature[index] = (glcm_test_feature[index] - min) / (max-min)
    glcm_test_feature = glcm_test_feature.transpose(0,1)
    glcm_train_feature = glcm_train_feature.transpose(0,1)

    train_list = {  'glcm' : glcm_train_feature,
                    'googlenet' : googlenet_train_feature, 
                    'resnet' : resnet_train_feature, 
                    'vgg' : vgg_train_feature, 
                    'hog' : hog_train_feature,
                    'lbp' : lbp_train_feature,
                    }
    test_list = {   'glcm' : glcm_test_feature,
                    'googlenet' : googlenet_test_feature, 
                    'resnet' : resnet_test_feature, 
                    'vgg' : vgg_test_feature, 
                    'hog' : hog_test_feature,
                    'lbp' : lbp_test_feature,
                    'glcm' : glcm_test_feature,
                    }
    #SVM
    #glcm max_iter=500 效果最佳
    set_logger('./experiments/svm/log_mask.log')
    kernel_function = ['linear', 'poly', 'rbf', 'sigmoid']
    C = [1e-2,1e-1,1,1e1,1e2] #C是对错误的惩罚
    gamma = [0.0001,0.0005,0.001,0.005,0.01,0.1] 
    max_iter = [50,100,200,500,700,1000,1500,2000]
    with tqdm(total = len(train_list)*len(kernel_function)*len(C)*len(gamma)*len(max_iter)) as t:
        for feature_type in train_list:
            train_feature = train_list[feature_type]
            test_feature = test_list[feature_type]
            best_accuracy = 0
            for kf in kernel_function:
                for c in C:
                    for ga in gamma:
                        for miter in max_iter:
                            clf = SVC(kernel=kf, C=c, gamma=ga, max_iter=miter,probability=True)
                            clf = clf.fit(train_feature, train_label)

                            Y_pred = clf.predict(test_feature)
                            # Y_pred_prob = clf.predict_proba(test_feature)
                            con_matrix = confusion_matrix(test_label,Y_pred,labels=range(2))
                            
                            accuracy = (con_matrix[0][0] + con_matrix[1][1])/160
                            if accuracy > best_accuracy:
                                save_matrix = con_matrix
                                param_list = [c, ga, miter,kf]
                                best_accuracy = accuracy
                                #将最好的情况的预测结果写进csv
                                predict_csv = open('./data/mask_feature/'+feature_type+'_testset_result.csv','w',encoding='utf-8')
                                csv_writer = csv.writer(predict_csv)
                                csv_writer.writerow(["filename","truth_label","predict_label","probability","is_right"])
                                for i_npy in range(len(file_name_list)):
                                    if test_label[i_npy] == Y_pred[i_npy]:
                                        is_right = True
                                    else:
                                        is_right = False
                                    data = [file_name_list[i_npy], int(test_label[i_npy].detach().numpy()), int(Y_pred[i_npy]), 0, is_right]
                                    csv_writer.writerow(data)
                            t.update()
            
            TN = save_matrix[0][0]
            TP = save_matrix[1][1]
            FN = save_matrix[1][0]
            FP = save_matrix[0][1]
            logging.info('TN:{0}, TP:{1}, FN:{2}, FP:{3} '.format(TN, TP, FN, FP))
            logging.info('{0} svm classification, kernel_function={1}, c={2}, gamma={3}, max_iter={4}, test_accuracy={5}'.format(feature_type, 
                                                                                                                                    param_list[3],
                                                                                                                                    param_list[0], 
                                                                                                                                    param_list[1], 
                                                                                                                                    param_list[2],
                                                                                                                                    best_accuracy))
                    
#使用训练的GCN模型来提取中间层特征用于svm分类和直方图观察
def get_gcn_feature():
    model = GCN(nfeat=512,
            nhid=64,
            nclass=2,
            fc_num=2,
            dropout=0.6)
    model.load_state_dict(torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar'), strict=False)

    googlenet_train_feature = torch.load('./data/mask_feature/googlenet_train_feature.pt')
    googlenet_test_feature = torch.load('./data/mask_feature/googlenet_test_feature.pt')
    resnet_train_feature = torch.load('./data/mask_feature/resnet_train_feature.pt')
    resnet_test_feature = torch.load('./data/mask_feature/resnet_test_feature.pt')
    vgg_train_feature = torch.load('./data/mask_feature/vgg_train_feature.pt')
    vgg_test_feature = torch.load('./data/mask_feature/vgg_test_feature.pt')
    hog_train_feature = torch.load('./data/mask_feature/hog_train_feature.pt')
    hog_test_feature = torch.load('./data/mask_feature/hog_test_feature.pt')

    feature_type = 4
    adj = Variable(torch.ones((feature_type, feature_type)))
    train_middle_feature_64 = torch.zeros((639,feature_type,64))
    test_middle_feature_64 = torch.zeros((160,feature_type,64))
    train_middle_feature_2 = torch.zeros((639,feature_type,2))
    test_middle_feature_2 = torch.zeros((160,feature_type,2))
    for index, one_nodule_feature in enumerate(zip(googlenet_train_feature, 
                                                    resnet_train_feature, 
                                                    vgg_train_feature,
                                                    hog_train_feature)):
        temp = torch.zeros((len(one_nodule_feature),512))
        for i, feature in enumerate(one_nodule_feature):
            temp[i] = feature
        one_nodule_feature = temp

        adj = torch.ones((len(one_nodule_feature), len(one_nodule_feature)))
        one_nodule_feature = torch.from_numpy(np.array(one_nodule_feature))
        features, adj = Variable(one_nodule_feature), Variable(adj)
        model.eval()
        middle_feature_64, middle_feature_2, _ = model(features, adj)
        train_middle_feature_64[index] = middle_feature_64
        train_middle_feature_2[index] = middle_feature_2

    
    for index, one_nodule_feature in enumerate(zip(googlenet_test_feature, 
                                                    resnet_test_feature, 
                                                    vgg_test_feature,
                                                    hog_test_feature)):
        temp = torch.zeros((len(one_nodule_feature),512))
        for i, feature in enumerate(one_nodule_feature):
            temp[i] = feature
        one_nodule_feature = temp

        adj = torch.ones((len(one_nodule_feature), len(one_nodule_feature)))
        one_nodule_feature = torch.from_numpy(np.array(one_nodule_feature))
        features, adj = Variable(one_nodule_feature), Variable(adj)
        model.eval()
        middle_feature_64, middle_feature_2, _ = model(features, adj)
        test_middle_feature_64[index] = middle_feature_64
        test_middle_feature_2[index] = middle_feature_2

    torch.save(train_middle_feature_64, './experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/train_middle_feature_64.pt')
    torch.save(train_middle_feature_2, './experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/train_middle_feature_2.pt')
    torch.save(test_middle_feature_64, './experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/test_middle_feature_64.pt')
    torch.save(test_middle_feature_2, './experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/test_middle_feature_2.pt')

#将提取出来的GCN特征（64维和2维）用折线图可视化
def gcn_feature_line():
    train_middle_feature_64 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/train_middle_feature_64.pt')
    train_middle_feature_2 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/train_middle_feature_2.pt')
    test_middle_feature_64 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/test_middle_feature_64.pt')
    test_middle_feature_2 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/test_middle_feature_2.pt')
    train_label = torch.load('./data/mask_feature/train_label.pt')
    test_label = torch.load('./data/mask_feature/test_label.pt')

    # 训练集 64
    print("train 64")
    for index in range(train_middle_feature_64.shape[0]):
        plt.rcParams['figure.figsize'] = (12, 12)
        label = train_label[index].detach().numpy()
        if int(label) == 0:
            plt.subplot(3,1,1)
            plt.title('label=0')
            plt.plot([i for i in range(train_middle_feature_64.shape[2])],train_middle_feature_64[index,0,:].detach().numpy())
        elif int(label) == 1:
            plt.subplot(3,1,2)
            plt.title('label=1')
            plt.plot([i for i in range(train_middle_feature_64.shape[2])],train_middle_feature_64[index,0,:].detach().numpy())
        plt.subplot(3,1,3)
        plt.title('all label')
        plt.plot([i for i in range(train_middle_feature_64.shape[2])],train_middle_feature_64[index,0,:].detach().numpy())
    plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/line/train_fig_dim_64.png')
    plt.clf()
    for index in range(train_middle_feature_64.shape[0]):
        print(index)
        label = train_label[index].detach().numpy()
        for jndex in range(train_middle_feature_64.shape[1]):
            plt.rcParams['figure.figsize'] = (12, 12)
            plt.subplot(5,1,jndex+1)
            plt.plot([i for i in range(train_middle_feature_64.shape[2])],train_middle_feature_64[index,jndex,:].detach().numpy())
            plt.subplot(5,1,5)
            plt.plot([i for i in range(train_middle_feature_64.shape[2])],train_middle_feature_64[index,jndex,:].detach().numpy())
        plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/line/train_64/train_middle_feature_64_'+'index_'+str(index)+'_label_'+str(int(label))+'.png')
        plt.clf()#清除图

    #训练集 2
    print("train 2")
    for index in range(train_middle_feature_2.shape[0]):
        plt.rcParams['figure.figsize'] = (12, 12)
        label = train_label[index].detach().numpy()
        if int(label) == 0:
            plt.subplot(3,1,1)
            plt.title('label=0')
            plt.plot([i for i in range(train_middle_feature_2.shape[2])],train_middle_feature_2[index,0,:].detach().numpy())
        elif int(label) == 1:
            plt.subplot(3,1,2)
            plt.title('label=1')
            plt.plot([i for i in range(train_middle_feature_2.shape[2])],train_middle_feature_2[index,0,:].detach().numpy())
        plt.subplot(3,1,3)
        plt.title('all label')
        plt.plot([i for i in range(train_middle_feature_2.shape[2])],train_middle_feature_2[index,0,:].detach().numpy())
    plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/line/train_fig_dim_2.png')
    plt.clf()
    for index in range(train_middle_feature_2.shape[0]):
        print(index)
        label = train_label[index].detach().numpy()
        for jndex in range(train_middle_feature_2.shape[1]):
            plt.rcParams['figure.figsize'] = (12, 12)
            plt.subplot(5,1,jndex+1)
            plt.plot([i for i in range(train_middle_feature_2.shape[2])],train_middle_feature_2[index,jndex,:].detach().numpy())
            plt.subplot(5,1,5)
            plt.plot([i for i in range(train_middle_feature_2.shape[2])],train_middle_feature_2[index,jndex,:].detach().numpy())
        plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/line/train_2/train_middle_feature_2_'+'index_'+str(index)+'_label_'+str(int(label))+'.png')
        plt.clf()#清除图

    #测试集 64
    print("test 64")
    for index in range(test_middle_feature_64.shape[0]):
        plt.rcParams['figure.figsize'] = (12, 12)
        label = test_label[index].detach().numpy()
        if int(label) == 0:
            plt.subplot(3,1,1)
            plt.title('label=0')
            plt.plot([i for i in range(test_middle_feature_64.shape[2])],test_middle_feature_64[index,0,:].detach().numpy())
        elif int(label) == 1:
            plt.subplot(3,1,2)
            plt.title('label=1')
            plt.plot([i for i in range(test_middle_feature_64.shape[2])],test_middle_feature_64[index,0,:].detach().numpy())
        plt.subplot(3,1,3)
        plt.title('all label')
        plt.plot([i for i in range(test_middle_feature_64.shape[2])],test_middle_feature_64[index,0,:].detach().numpy())
    plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/line/test_fig_dim_64.png')
    plt.clf()
    for index in range(test_middle_feature_64.shape[0]):
        print(index)
        label = test_label[index].detach().numpy()
        for jndex in range(test_middle_feature_64.shape[1]):
            plt.rcParams['figure.figsize'] = (12, 12)
            plt.subplot(5,1,jndex+1)
            plt.plot([i for i in range(test_middle_feature_64.shape[2])],test_middle_feature_64[index,jndex,:].detach().numpy())
            plt.subplot(5,1,5)
            plt.plot([i for i in range(test_middle_feature_64.shape[2])],test_middle_feature_64[index,jndex,:].detach().numpy())
        plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/line/test_64/test_middle_feature_64_'+'index_'+str(index)+'_label_'+str(int(label))+'.png')
        plt.clf()#清除图

    #测试集 2
    print("test 2")
    for index in range(test_middle_feature_2.shape[0]):
        plt.rcParams['figure.figsize'] = (12, 12)
        label = test_label[index].detach().numpy()
        if int(label) == 0:
            plt.subplot(3,1,1)
            plt.title('label=0')
            plt.plot([i for i in range(test_middle_feature_2.shape[2])],test_middle_feature_2[index,0,:].detach().numpy())
        elif int(label) == 1:
            plt.subplot(3,1,2)
            plt.title('label=1')
            plt.plot([i for i in range(test_middle_feature_2.shape[2])],test_middle_feature_2[index,0,:].detach().numpy())
        plt.subplot(3,1,3)
        plt.title('all label')
        plt.plot([i for i in range(test_middle_feature_2.shape[2])],test_middle_feature_2[index,0,:].detach().numpy())
    plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/line/test_fig_dim_2.png')
    plt.clf()
    for index in range(test_middle_feature_2.shape[0]):
        print(index)
        label = test_label[index].detach().numpy()
        for jndex in range(test_middle_feature_2.shape[1]):
            plt.rcParams['figure.figsize'] = (12, 12)
            plt.subplot(5,1,jndex+1)
            plt.plot([i for i in range(test_middle_feature_2.shape[2])],test_middle_feature_2[index,jndex,:].detach().numpy())
            plt.subplot(5,1,5)
            plt.plot([i for i in range(test_middle_feature_2.shape[2])],test_middle_feature_2[index,jndex,:].detach().numpy())
        plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/line/test_2/test_middle_feature_2_'+'index_'+str(index)+'_label_'+str(int(label))+'.png')
        plt.clf()#清除图

#将提取出来的GCN特征（64维和2维）用直方图可视化
def gcn_feature_histogram():
    train_middle_feature_64 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/train_middle_feature_64.pt')
    train_middle_feature_2 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/train_middle_feature_2.pt')
    test_middle_feature_64 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/test_middle_feature_64.pt')
    test_middle_feature_2 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/test_middle_feature_2.pt')
    train_label = torch.load('./data/mask_feature/train_label.pt')
    test_label = torch.load('./data/mask_feature/test_label.pt')

    # 训练集 64
    print("train 64")
    for index in range(train_middle_feature_64.shape[0]):
        print(index)
        label = train_label[index].detach().numpy()
        for jndex in range(train_middle_feature_64.shape[1]):
            plt.rcParams['figure.figsize'] = (12, 12)
            plt.subplot(5,1,jndex+1)
            Max = max(train_middle_feature_64[index,jndex,:]).detach().numpy()
            Min = min(train_middle_feature_64[index,jndex,:]).detach().numpy()
            seper = (Max-Min)/20
            plt.hist(train_middle_feature_64[index,jndex,:].detach().numpy(),bins=np.arange(Min,Max,seper))
            plt.subplot(5,1,5)
            plt.hist(train_middle_feature_64[index,jndex,:].detach().numpy(),bins=np.arange(Min,Max,seper))
        plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/hist/train_64/train_middle_feature_64_'+'index_'+str(index)+'_label_'+str(int(label))+'.png')
        plt.clf()#清除图

    #训练集 2
    print("train 2")
    for index in range(train_middle_feature_2.shape[0]):
        print(index)
        label = train_label[index].detach().numpy()
        for jndex in range(train_middle_feature_2.shape[1]):
            plt.rcParams['figure.figsize'] = (12, 12)
            plt.subplot(5,1,jndex+1)
            Max = max(train_middle_feature_2[index,jndex,:]).detach().numpy()
            Min = min(train_middle_feature_2[index,jndex,:]).detach().numpy()
            seper = (Max-Min)/20
            plt.hist(train_middle_feature_2[index,jndex,:].detach().numpy(),bins=np.arange(Min,Max,seper))
            plt.subplot(5,1,5)
            plt.hist(train_middle_feature_2[index,jndex,:].detach().numpy(),bins=np.arange(Min,Max,seper))
        plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/hist/train_2/train_middle_feature_2_'+'index_'+str(index)+'_label_'+str(int(label))+'.png')
        plt.clf()#清除图

    #测试集 64
    print("test 64")
    for index in range(test_middle_feature_64.shape[0]):
        print(index)
        label = test_label[index].detach().numpy()
        for jndex in range(test_middle_feature_64.shape[1]):
            plt.rcParams['figure.figsize'] = (12, 12)
            plt.subplot(5,1,jndex+1)
            Max = max(test_middle_feature_64[index,jndex,:]).detach().numpy()
            Min = min(test_middle_feature_64[index,jndex,:]).detach().numpy()
            seper = (Max-Min)/20
            plt.hist(test_middle_feature_64[index,jndex,:].detach().numpy(),bins=np.arange(Min,Max,seper))
            plt.subplot(5,1,5)
            plt.hist(test_middle_feature_64[index,jndex,:].detach().numpy(),bins=np.arange(Min,Max,seper))
        plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/hist/test_64/test_middle_feature_64_'+'index_'+str(index)+'_label_'+str(int(label))+'.png')
        plt.clf()#清除图

    #测试集 2
    print("test 2")
    for index in range(test_middle_feature_2.shape[0]):
        print(index)
        label = test_label[index].detach().numpy()
        for jndex in range(test_middle_feature_2.shape[1]):
            plt.rcParams['figure.figsize'] = (12, 12)
            plt.subplot(5,1,jndex+1)
            Max = max(test_middle_feature_2[index,jndex,:]).detach().numpy()
            Min = min(test_middle_feature_2[index,jndex,:]).detach().numpy()
            seper = (Max-Min)/20
            plt.hist(test_middle_feature_2[index,jndex,:].detach().numpy(),bins=np.arange(Min,Max,seper))
            plt.subplot(5,1,5)
            plt.hist(test_middle_feature_2[index,jndex,:].detach().numpy(),bins=np.arange(Min,Max,seper))
        plt.savefig('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/hist/test_2/test_middle_feature_2_'+'index_'+str(index)+'_label_'+str(int(label))+'.png')
        plt.clf()#清除图

#使用svm对gcn的中间层特征进行分类
def svm_classification_gcn_middle_feature():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    
    for fold in range(10):
        gcn_train_feature = torch.load('data/feature/10fold_gcn_feature_noNorm_random_adj_addGoogleNet/gcn_train_middle_feature_fold_'+str(fold)+'.pt')   #639*64
        gcn_test_feature = torch.load('data/feature/10fold_gcn_feature_noNorm_random_adj_addGoogleNet/gcn_test_middle_feature_fold_'+str(fold)+'.pt')   #160*64
        train_label = torch.load('data/feature/10fold_model_feature/fold_'+str(fold)+'_train_label.pt')
        test_label = torch.load('data/feature/10fold_model_feature/fold_'+str(fold)+'_test_label.pt')
        
        testSet_len = test_label.shape[0]
        #SVM
        #glcm max_iter=500 效果最佳
        set_logger('./experiments/gcn/gcm_middle_feature_svm_classification_log.log')
        kernel_function = ['linear', 'poly', 'rbf', 'sigmoid']
        C = [1e-2,1e-1,1,1e1,1e2] #C是对错误的惩罚
        gamma = [0.0001,0.0005,0.001,0.005,0.01,0.1] 
        max_iter = [50,100,200,500,700,1000,1500,2000]
        with tqdm(total = len(kernel_function)*len(C)*len(gamma)*len(max_iter)) as t:
            train_feature = gcn_train_feature.detach().numpy()
            test_feature = gcn_test_feature.detach().numpy()
            best_accuracy = 0
            for kf in kernel_function:
                for c in C:
                    for ga in gamma:
                        for miter in max_iter:
                            clf = SVC(kernel=kf, C=c, gamma=ga, max_iter=miter)
                            clf = clf.fit(train_feature, train_label)

                            Y_pred = clf.predict(test_feature)
                            con_matrix = confusion_matrix(test_label,Y_pred,labels=range(2))
                            
                            accuracy = (con_matrix[0][0] + con_matrix[1][1])/testSet_len
                            if accuracy > best_accuracy:
                                save_matrix = con_matrix
                                param_list = [c, ga, miter,kf]
                                best_accuracy = accuracy
                            t.update()
            
            TN = save_matrix[0][0]
            TP = save_matrix[1][1]
            FN = save_matrix[1][0]
            FP = save_matrix[0][1]
            logging.info('TN:{0}, TP:{1}, FN:{2}, FP:{3} '.format(TN, TP, FN, FP))
            logging.info('{0} svm classification, kernel_function={1}, c={2}, gamma={3}, max_iter={4}, test_accuracy={5}'.format('gcn', 
                                                                                                                                    param_list[3],
                                                                                                                                    param_list[0], 
                                                                                                                                    param_list[1], 
                                                                                                                                    param_list[2],
                                                                                                                                    best_accuracy))

#计算5种方法之间的预测相似度，以此来构建邻接矩阵
def caculate_five_method_predict_similarity():
    modelList = ['alexnet','attention56','vgg13','resnet34','googlenet']
    descripe = 'para1_10fold_noNorm_add_gcn_includeGoogLeNet_traditional'
    sim_adj = np.zeros((5,5))
    for fold in range(10):
        for model in modelList:
            jsonFileName = 'folder.'+str(fold)+'.FocalLoss_alpha_0.25_'+descripe+'.metrics_val_best_weights.json'
            jsonFilePath = 'experiments/'+model+'_nomask/'+jsonFileName
            f = open(jsonFilePath,'r')
            jsonData = json.load(f)
            jsonAcc = jsonData['accuracy']
            jsonEpoch = jsonData['epoch']
            csvPath = 'experiments/'+\
                        model+\
                        '_nomask/result_'+\
                        descripe +\
                        '/folder_'+\
                        str(fold)+\
                        '_result_'+\
                        str(int(jsonEpoch-1))+\
                        '.csv'
            csvReader = pd.read_csv(csvPath)
            if model=='alexnet':
                alexnetPredLabel = csvReader['predict_label']
            if model=='attention56':
                attention56PredLabel = csvReader['predict_label']
            if model=='vgg13':
                vgg13PredLabel = csvReader['predict_label']
            if model=='resnet34':
                resnet34PredLabel = csvReader['predict_label']
            if model=='googlenet':
                googlenetPredLabel = csvReader['predict_label']
        data_len = len(googlenetPredLabel)
        all_predict_list = [resnet34PredLabel, vgg13PredLabel, alexnetPredLabel, attention56PredLabel, googlenetPredLabel]
        method_num = 5
        sim_matrix = np.zeros((method_num,method_num))
        cord = 0
        for i_predict_list in all_predict_list:
            for j_predict_list in all_predict_list:
                correct = np.sum(i_predict_list == j_predict_list)
                row = int(cord / method_num)
                coloum = cord % method_num
                # if row != coloum:
                #     sim_matrix[row][coloum] = (1-correct/data_len)*10   
                #     sim_matrix[row][coloum] = correct/data_len
                # else:
                #     sim_matrix[row][coloum] = 0
                sim_matrix[row][coloum] = (1-correct/data_len)*10  
                cord += 1
    #     sim_adj = sim_adj + sim_matrix
    # sim_adj = sim_adj/10
    # print(sim_adj)
    return sim_matrix


#提取15种特征的10个样本（5个正样本，5个负样本）的512维特征，写进txt文档中
def exract_15_feature_10_sample_write_in_txt():
    googlenet_train_feature = torch.load('./data/feature/googlenet_train.pt')
    googlenet_test_feature = torch.load('./data/feature/googlenet_test.pt')
    resnet10_train_feature = torch.load('./data/feature/resnet10_train.pt')
    resnet10_test_feature = torch.load('./data/feature/resnet10_test.pt')
    resnet18_train_feature = torch.load('./data/feature/resnet18_train.pt')
    resnet18_test_feature = torch.load('./data/feature/resnet18_test.pt')
    resnet34_train_feature = torch.load('./data/feature/resnet34_train.pt')
    resnet34_test_feature = torch.load('./data/feature/resnet34_test.pt')
    resnet50_train_feature = torch.load('./data/feature/resnet50_train.pt')
    resnet50_test_feature = torch.load('./data/feature/resnet50_test.pt')
    resnet101_train_feature = torch.load('./data/feature/resnet101_train.pt')
    resnet101_test_feature = torch.load('./data/feature/resnet101_test.pt')
    resnet152_train_feature = torch.load('./data/feature/resnet152_train.pt')
    resnet152_test_feature = torch.load('./data/feature/resnet152_test.pt')
    resnet200_train_feature = torch.load('./data/feature/resnet200_train.pt')
    resnet200_test_feature = torch.load('./data/feature/resnet200_test.pt')
    vgg11_train_feature = torch.load('./data/feature/vgg11_train.pt')
    vgg11_test_feature = torch.load('./data/feature/vgg11_test.pt')
    vgg13_train_feature = torch.load('./data/feature/vgg13_train.pt')
    vgg13_test_feature = torch.load('./data/feature/vgg13_test.pt')
    vgg16_train_feature = torch.load('./data/feature/vgg16_train.pt')
    vgg16_test_feature = torch.load('./data/feature/vgg16_test.pt')
    vgg19_train_feature = torch.load('./data/feature/vgg19_train.pt')
    vgg19_test_feature = torch.load('./data/feature/vgg19_test.pt')
    hog_train_feature = torch.load('./data/feature/hog_train_feature.pt')
    hog_test_feature = torch.load('./data/feature/hog_test_feature.pt')
    lbp_train_feature = torch.load('./data/feature/lbp_train_feature.pt')
    lbp_test_feature = torch.load('./data/feature/lbp_test_feature.pt')
    glcm_train_feature = torch.load('./data/feature/glcm_train_feature.pt')
    glcm_test_feature = torch.load('./data/feature/glcm_test_feature.pt')
    train_label = torch.load('./data/feature/train_label.pt')
    test_label = torch.load('./data/feature/test_label.pt')

    import model.data_loader as data_loader
    dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 1, data_dir='data/nodules3d_128_npy', train_shuffle=False)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']
    filename_list = []
    for i, (x, target, file_name) in enumerate(train_dl):
        filename_list.append(file_name[0])
    flag_positive = 0
    flag_negative = 0
    for i in range(100):
        if train_label[i] == 0: # 负样本
            if flag_negative == 5:
                continue
            f = open('./experiments/feature_15_sample_10/'+filename_list[i]+'.txt','w')
            f.write('googlenet ' + str(googlenet_train_feature[i]) + '\n')
            f.write('resnet10_train_feature ' + str(resnet10_train_feature[i]) + '\n')
            f.write('resnet18_train_feature ' + str(resnet18_train_feature[i]) + '\n')
            f.write('resnet34_train_feature ' + str(resnet34_train_feature[i]) + '\n')
            f.write('resnet50_train_feature ' + str(resnet50_train_feature[i]) + '\n')
            f.write('resnet101_train_feature ' + str(resnet101_train_feature[i]) + '\n')
            f.write('resnet152_train_feature ' + str(resnet152_train_feature[i]) + '\n')
            f.write('resnet200_train_feature ' + str(resnet200_train_feature[i]) + '\n')
            f.write('vgg11_train_feature ' + str(vgg11_train_feature[i]) + '\n')
            f.write('vgg13_train_feature ' + str(vgg13_train_feature[i]) + '\n')
            f.write('vgg16_train_feature ' + str(vgg16_train_feature[i]) + '\n')
            f.write('vgg19_train_feature ' + str(vgg19_train_feature[i]) + '\n')
            f.write('hog_train_feature ' + str(hog_train_feature[i]) + '\n')
            f.write('lbp_train_feature ' + str(lbp_train_feature[i]) + '\n')
            f.write('glcm_train_feature ' + str(glcm_train_feature[i]) + '\n')
            f.close()
            flag_negative += 1
        if train_label[i] == 1: # 负样本
            if flag_positive == 5:
                continue
            f = open('./experiments/feature_15_sample_10/'+filename_list[i]+'.txt','w')
            f.write('googlenet ' + str(googlenet_train_feature[i]) + '\n')
            f.write('resnet10_train_feature ' + str(resnet10_train_feature[i]) + '\n')
            f.write('resnet18_train_feature ' + str(resnet18_train_feature[i]) + '\n')
            f.write('resnet34_train_feature ' + str(resnet34_train_feature[i]) + '\n')
            f.write('resnet50_train_feature ' + str(resnet50_train_feature[i]) + '\n')
            f.write('resnet101_train_feature ' + str(resnet101_train_feature[i]) + '\n')
            f.write('resnet152_train_feature ' + str(resnet152_train_feature[i]) + '\n')
            f.write('resnet200_train_feature ' + str(resnet200_train_feature[i]) + '\n')
            f.write('vgg11_train_feature ' + str(vgg11_train_feature[i]) + '\n')
            f.write('vgg13_train_feature ' + str(vgg13_train_feature[i]) + '\n')
            f.write('vgg16_train_feature ' + str(vgg16_train_feature[i]) + '\n')
            f.write('vgg19_train_feature ' + str(vgg19_train_feature[i]) + '\n')
            f.write('hog_train_feature ' + str(hog_train_feature[i]) + '\n')
            f.write('lbp_train_feature ' + str(lbp_train_feature[i]) + '\n')
            f.write('glcm_train_feature ' + str(glcm_train_feature[i]) + '\n')
            f.close()
            flag_positive += 1

#得到两个矩阵的余弦相似度
def get_matrix_similarity(_matrixA, _matrixB):
    _matrixA_matrixB = np.dot(_matrixA, _matrixB.T)
    _matrixA_norm = np.sqrt(np.multiply(_matrixA,_matrixA).sum(axis=1))
    _matrixB_norm = np.sqrt(np.multiply(_matrixB,_matrixB).sum(axis=1))
    return np.divide(_matrixA_matrixB, np.dot(_matrixA_norm.reshape(_matrixA_norm.shape[0],1), _matrixB_norm.reshape(1,_matrixA_norm.shape[0])))

def search_different_resnet_feature_correlation():
    matirx_a = torch.load('data/feature/resnet18_test.pt').numpy()
    matirx_b = torch.load('data/feature/resnet34_test.pt').numpy()
    print(matirx_b[29])
    print(matirx_b[0])
    sim = get_matrix_similarity(matirx_b, matirx_b)
    for i in range(160):
        print(sim[i][i], np.max(sim[i]), np.argmax(sim[i]))

#修改测试集训练集之后，需要重新获取数据集的标签，并且保存下来
def get_dataset_label_pt():
    datasetPath = '10fold'
    for index in range(10):
        dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 3000, data_dir="data/"+datasetPath+"/fold"+str(index+1), train_shuffle=False,fold=index)
        train_dl = dataloaders['train']
        test_dl = dataloaders['test']
        for i, (train_batch, labels_batch, _,_) in enumerate(train_dl):
            torch.save(labels_batch,'data/feature/'+'10fold_model_feature'+'/fold_'+str(index)+'_train_label.pt')
        for i, (train_batch, labels_batch, _,_) in enumerate(test_dl):
            torch.save(labels_batch,'data/feature/'+'10fold_model_feature'+'/fold_'+str(index)+'_test_label.pt')

#将所有模型的（目前是已经训练好的模型）预测结果汇总到同一个csv中
def converage_all_result():
    best_json = glob.glob(r'./experiments/*_nomask/*best*')
    print(len(best_json))

import json
#计算所有特征之间的预测相似度，包括gcn
def caculate_all_method_predict_similarity():
    best_json = glob.glob(r'./experiments/*_nomask/folder.0.FocalLoss_alpha_0.25.metrics_val_best_weights*')
    model_name_list = ['googlenet', 'resnet10', 'resnet18', 'resnet34', 'resnet50',
    'resnet101', 'resnet152', 'resnet200', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'alexnet',
    'attention56', 'attention92', 'mobilenet', 'mobilenetv2', 'shufflenet', 'squeezenet',
    'preactresnet18', 'preactresnet34', 'preactresnet50', 'preactresnet101', 'preactresnet152',
    'inceptionv3', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'resnext50',
    'resnext101', 'resnext152', 'resnet_in_resnet', 'senet18', 'senet34', 'senet50', 'senet101',
    'senet152', 'xception', 'wideresidual']
    model_name_list.sort()
    print(model_name_list)
    wanted_path = []
    for i in best_json:
        for j in model_name_list:
            if j in i:
                wanted_path.append(i)
                break
    wanted_path.sort() #对路径排序
    model_name = open('./experiments/gcn/model_name_order.txt','w')
    for i in wanted_path:
        name = i.split('/')[2].split('_')[0]
        model_name.write(name + '\n')
    model_name.flush()

    predict_list = []
    for json_path in wanted_path:
        parent_path = json_path.split('/f')[0]
        f = open(json_path, 'r')
        epoch = int(float(f.readlines()[-2].split(':')[1].strip()))
        csv_path = parent_path + '/result/folder_0_result_' + str(epoch-1) + '.csv'

        csv_reader = csv.reader(open(csv_path))
        header = next(csv_reader) #csv头
        temp_predict_list = []
        for row in csv_reader:
            temp_predict_list.append(row[2])
        predict_list.append(temp_predict_list)

    gcn_predict_list = []
    csv_reader = csv.reader(open('experiments/gcn/result/best_result.csv'))
    header = next(csv_reader) 
    for row in csv_reader:
        gcn_predict_list.append(row[2])
    predict_list.append(gcn_predict_list)

    feature_num = 41
    nodule_num = 157
    sim_matrix = np.zeros((feature_num,feature_num))
    cord = 0
    for i_predict_list in predict_list:
        for j_predict_list in predict_list:
            correct = 0
            for index in range(nodule_num):
                if i_predict_list[index] == j_predict_list[index]:
                    correct += 1
            row = int(cord / feature_num)
            coloum = cord % feature_num
            sim_matrix[row][coloum] = correct/nodule_num
            cord += 1
    np.savetxt('./experiments/gcn/sim_matrix.txt', sim_matrix, delimiter = ',', fmt='%.04f') 

#将测试集的图片转化成2d图片并保存下来
def get_2d_test_png():
    npy_path = glob.glob(r'./data/nodules3d_128_npy_no_same_patient_in_two_dataset/test/*.npy')
    save_root = './data/nodules3d_128_npy_no_same_patient_in_two_dataset_test_2d/'
    for one_path in npy_path:
        name = one_path.split('/')[-1].split('.')[0]
        npy = np.load(one_path)
        for i in range(8):
            save_name = save_root + name + '_' + str(i) + '.png'
            slice = npy[:, :, i]
            plt.imsave(save_name, slice, cmap='gray')


#计算一个结节图片中结节像素占整个图像的比例
def calculate_percentage(nodule_name):
    nodule_path = './data/nodules3d_128_mask_no_split/' + nodule_name
    npy = np.load(nodule_path)
    count = 0
    for i in range(8):
        slice = npy[:, :, i]
        for slice_i in range(128):
            for slice_j in range(128):
                if slice[slice_i][slice_j] != 0.0:
                    count += 1
    percentage = (count / (128*128*8))
    return percentage*100

#确认蔡兄的另外4折是否有同一病例的结节即在训练集又在测试集
def confirm_patient_nodule_not_in_two_dataset():
    for fold in range(1,6):
        fold_path = 'data/5fold_128_new/fold'+str(fold)
        train_npy = glob.glob(os.path.join(fold_path,'train/*npy'))
        test_npy = glob.glob(os.path.join(fold_path,'test/*npy'))
        train_patient = []
        test_patient = []
        train_positive, train_negative, test_positive, test_negative = 0, 0, 0, 0
        for npy_name in train_npy:
            train_patient.append(npy_name.split('/')[-1].split('_')[0])
            if npy_name.split('/')[-1].split('_')[-1].split('.')[0] == '0':
                train_negative += 1
            else:
                train_positive += 1
        for npy_name in test_npy:
            test_patient.append(npy_name.split('/')[-1].split('_')[0])
            if npy_name.split('/')[-1].split('_')[-1].split('.')[0] == '0':
                test_negative += 1
            else:
                test_positive += 1
        train_patient = set(train_patient)
        test_patient = set(test_patient)
        print("fold {0} exit {1}".format(fold, train_patient.intersection(test_patient)))
        print("训练集：0类 {0}，1类 {1}；测试集：0类{2}，1类{3}".format(train_negative, train_positive, test_negative, test_positive))
        print('='*15)

    #修改策略：将测试集中重复的病人结节全部移到训练集中去
    return

#提特征多线程部分
def get_various_feature_thread():
    pass


#8月2日 老师提供了许多特征，下面得函数就是提取这一系列特征的函数  
import pandas as pd
import threading
def get_various_feature():
    df = pd.read_csv('./data/pylidc_feature.csv')
    for fold in range(5):
        dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 1, data_dir="data/5fold_128<=20mm_mask_aug/fold"+str(fold+1), train_shuffle=False, fold= fold)
        train_dl = dataloaders['train']
        test_dl = dataloaders['test']
        #255 = 64 * 3 + 56 +7
        feature_train = torch.zeros((len(train_dl),255))
        feature_test = torch.zeros((len(test_dl),255))
        for i, (cube, _, file_name, _) in enumerate(train_dl):
            startTime = time.time()
            # if fold == 0: #因为5折交叉验证的来源不同，文件命名有些差异
            #     patient = file_name[0].split('_')[0][:4]
            #     nodule = file_name[0].split('_')[0][4:]
            # else:
            patient = file_name[0].split('_')[0]
            nodule = file_name[0].split('_')[1]
            nodule_idx = int(patient + nodule)
            diameter = np.array([df[df["nodule_idx"]==nodule_idx]["diameter"].iloc[0]])
            surface_area = np.array([df[df["nodule_idx"]==nodule_idx]["surface_area"].iloc[0]])
            volume = np.array([df[df["nodule_idx"]==nodule_idx]["volume"].iloc[0]])

            try:
                hu = HU(cube)     #56维特征
                glcm = GLCM(cube) #64维特征，后续要归一化
                lbp = LBP(cube)   #64维特征
                hog = HOG(cube)   #64维特征

                cube = np.squeeze(cube).numpy()
                mean = np.array([np.mean(cube)])   #整个结节图片，包括结节外部的组织的均值
                variance = np.array([np.var(cube)])
                pd_cube = pd.Series(cube.flatten())
                skewness = np.array([pd_cube.skew()])
                kurtosis = np.array([pd_cube.kurt()])
                temp_feature = np.concatenate((hu, glcm, lbp, hog, diameter, surface_area, volume, mean, variance, skewness, kurtosis))
                feature_train[i] = torch.from_numpy(temp_feature)
            except:
                print('occur erro!!!----------------------------------------')
                temp_feature = torch.zeros((1,255))
                feature_train[i] = temp_feature
            endTime = time.time()
            print('---fold ' + str(fold) + '---train extracting ' + file_name[0] + ' time:' + str(endTime-startTime))
        torch.save(feature_train, './data/feature/addition_feature_mask<=20_aug/fold_' + str(fold) + '_train_addition_feature.pt')

        for i, (cube, _, file_name, _) in enumerate(test_dl):
            startTime = time.time()
            if fold == 0: #因为5折交叉验证的来源不同，文件命名有些差异
                patient = file_name[0].split('_')[0][:4]
                nodule = file_name[0].split('_')[0][4:]
            else:
                patient = file_name[0].split('_')[0]
                nodule = file_name[0].split('_')[1]
            nodule_idx = int(patient + nodule)
            diameter = np.array([df[df["nodule_idx"]==nodule_idx]["diameter"].iloc[0]])
            surface_area = np.array([df[df["nodule_idx"]==nodule_idx]["surface_area"].iloc[0]])
            volume = np.array([df[df["nodule_idx"]==nodule_idx]["volume"].iloc[0]])
            try:
                hu = HU(cube)     #56维特征
                glcm = GLCM(cube) #64维特征，后续要归一化
                lbp = LBP(cube)   #64维特征
                hog = HOG(cube)   #64维特征

                cube = np.squeeze(cube).numpy()
                mean = np.array([np.mean(cube)])   #整个结节图片，包括结节外部的组织的均值
                variance = np.array([np.var(cube)])
                pd_cube = pd.Series(cube.flatten())
                skewness = np.array([pd_cube.skew()])
                kurtosis = np.array([pd_cube.kurt()])
                temp_feature = np.concatenate((hu, glcm, lbp, hog, diameter, surface_area, volume, mean, variance, skewness, kurtosis))
                feature_test[i] = torch.from_numpy(temp_feature)
            except:
                print('occur erro!!!----------------------------------------')
                temp_feature = torch.zeros((1,255))
            
                feature_test[i] = temp_feature
            endTime = time.time()
            print('---fold ' + str(fold) + '---test extracting ' + file_name[0] + ' time:' + str(endTime-startTime))
        torch.save(feature_test, './data/feature/addition_feature_mask<=20_aug/fold_' + str(fold) + '_test_addition_feature.pt')
            
    return


def get_various_feature_thread():
    df = pd.read_csv('./data/pylidc_feature.csv')
    for fold in range(2):
        dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 16, data_dir="data/5fold_128<=20mm_mask_aug/fold"+str(fold+1), train_shuffle=False, fold= fold)
        train_dl = dataloaders['train']
        test_dl = dataloaders['test']
        #255 = 64 * 3 + 56 +7
        feature_train = torch.zeros((len(train_dl),255))
        feature_test = torch.zeros((len(test_dl),255))
        for i, (cube, _, file_name, _) in enumerate(train_dl):
            #提特征多线程部分
            def get_various_feature_thread_sub(cube,file_name,fold,ith):
                startTime = time.time()
                print('---fold ' + str(fold) + '---train extracting ' + file_name)
                patient = file_name.split('_')[0]
                nodule = file_name.split('_')[1]
                nodule_idx = int(patient + nodule)
                diameter = np.array([df[df["nodule_idx"]==nodule_idx]["diameter"].iloc[0]])
                surface_area = np.array([df[df["nodule_idx"]==nodule_idx]["surface_area"].iloc[0]])
                volume = np.array([df[df["nodule_idx"]==nodule_idx]["volume"].iloc[0]])

                try:
                    hu = HU(cube)     #56维特征
                    glcm = GLCM(cube) #64维特征，后续要归一化
                    lbp = LBP(cube)   #64维特征
                    hog = HOG(cube)   #64维特征

                    cube = np.squeeze(cube).numpy()
                    mean = np.array([np.mean(cube)])   #整个结节图片，包括结节外部的组织的均值
                    variance = np.array([np.var(cube)])
                    pd_cube = pd.Series(cube.flatten())
                    skewness = np.array([pd_cube.skew()])
                    kurtosis = np.array([pd_cube.kurt()])
                    temp_feature = np.concatenate((hu, glcm, lbp, hog, diameter, surface_area, volume, mean, variance, skewness, kurtosis))
                    feature_train[i*8+ith] = torch.from_numpy(temp_feature)
                except:
                    print('occur erro!!!----------------------------------------')
                    temp_feature = torch.zeros((1,255))
                    feature_train[i*8+ith] = temp_feature
                endTime = time.time()
                print(endTime-startTime)
            
            threadList = []
            for m in range(16):
                t = threading.Thread(target=get_various_feature_thread_sub, args=(cube[m],file_name[m],fold,m))
                threadList.append(t)
                t.start()
            for n in range(16):
                threadList[n].join()
            # t0 = threading.Thread(target=get_various_feature_thread, args=(cube[0],file_name[0],fold,0))
            # t1 = threading.Thread(target=get_various_feature_thread, args=(cube[1],file_name[1],fold,1))
            # t2 = threading.Thread(target=get_various_feature_thread, args=(cube[2],file_name[2],fold,2))
            # t3 = threading.Thread(target=get_various_feature_thread, args=(cube[3],file_name[3],fold,3))
            # t0.start()
            # t1.start()
            # t2.start()
            # t3.start()
            # t0.join()
            # t1.join()
            # t2.join()
            # t3.join()
        torch.save(feature_train, './data/feature/addition_feature_mask<=20_aug/fold_' + str(fold) + '_train_addition_feature.pt')

        for i, (cube, _, file_name, _) in enumerate(test_dl):
            def get_various_feature_thread(cube,file_name,fold,ith):
                print('---fold ' + str(fold) + '---test extracting ' + file_name[0])
                if fold == 0: #因为5折交叉验证的来源不同，文件命名有些差异
                    patient = file_name[0].split('_')[0][:4]
                    nodule = file_name[0].split('_')[0][4:]
                else:
                    patient = file_name[0].split('_')[0]
                    nodule = file_name[0].split('_')[1]
                nodule_idx = int(patient + nodule)
                diameter = np.array([df[df["nodule_idx"]==nodule_idx]["diameter"].iloc[0]])
                surface_area = np.array([df[df["nodule_idx"]==nodule_idx]["surface_area"].iloc[0]])
                volume = np.array([df[df["nodule_idx"]==nodule_idx]["volume"].iloc[0]])
                try:
                    hu = HU(cube)     #56维特征
                    glcm = GLCM(cube) #64维特征，后续要归一化
                    lbp = LBP(cube)   #64维特征
                    hog = HOG(cube)   #64维特征

                    cube = np.squeeze(cube).numpy()
                    mean = np.array([np.mean(cube)])   #整个结节图片，包括结节外部的组织的均值
                    variance = np.array([np.var(cube)])
                    pd_cube = pd.Series(cube.flatten())
                    skewness = np.array([pd_cube.skew()])
                    kurtosis = np.array([pd_cube.kurt()])
                    temp_feature = np.concatenate((hu, glcm, lbp, hog, diameter, surface_area, volume, mean, variance, skewness, kurtosis))
                    feature_test[i] = torch.from_numpy(temp_feature)
                except:
                    print('occur erro!!!----------------------------------------')
                    temp_feature = torch.zeros((1,255))
                    feature_test[i] = temp_feature
            t0 = threading.Thread(target=get_various_feature_thread, args=(cube[0],file_name[0],fold,0))
            t1 = threading.Thread(target=get_various_feature_thread, args=(cube[1],file_name[1],fold,1))
            t2 = threading.Thread(target=get_various_feature_thread, args=(cube[2],file_name[2],fold,2))
            t3 = threading.Thread(target=get_various_feature_thread, args=(cube[3],file_name[3],fold,3))
            t4 = threading.Thread(target=get_various_feature_thread, args=(cube[4],file_name[4],fold,4))
            t5 = threading.Thread(target=get_various_feature_thread, args=(cube[5],file_name[5],fold,5))
            t6 = threading.Thread(target=get_various_feature_thread, args=(cube[6],file_name[6],fold,6))
            t7 = threading.Thread(target=get_various_feature_thread, args=(cube[7],file_name[7],fold,7))
            t0.start()
            t1.start()
            t2.start()
            t3.start()
            t4.start()
            t5.start()
            t6.start()
            t7.start()
            t0.join()
            t1.join()
            t2.join()
            t3.join()
            t4.join()
            t5.join()
            t6.join()
            t7.join()
        torch.save(feature_test, './data/feature/addition_feature_mask<=20_aug/fold_' + str(fold) + '_test_addition_feature.pt')
            
    return



from sklearn.manifold import TSNE 
def t_SNE():
    for fold in range(5):
        dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 3000, data_dir="data/5fold_128<=20mm_mask/fold"+str(fold+1), train_shuffle=False, fold=0)
        train_dl = dataloaders['train']
        test_dl = dataloaders['test']
        for i, (train_batch, labels_batch, _, _) in enumerate(train_dl):
            train_label = labels_batch
        for i, (train_batch, labels_batch, _, _) in enumerate(test_dl):
            test_label = labels_batch
        train = torch.load('data/feature/addition_feature_mask<=20/fold_'+str(fold)+'_train_addition_feature.pt')
        test = torch.load('data/feature/addition_feature_mask<=20/fold_'+str(fold)+'_test_addition_feature.pt')
        for jndex in range(248,255):
            max = train[:, jndex].max()  
            min = train[:, jndex].min()  
            train[:, jndex] = (train[:, jndex] - min) / (max-min)
        for jndex in range(248,255):
            max = test[:, jndex].max()  
            min = test[:, jndex].min()  
            test[:, jndex] = (test[:, jndex] - min) / (max-min)


        
        tsne = TSNE(n_components=2).fit_transform(train)
        x_min, x_max = tsne.min(0), tsne.max(0)
        tsne_norm = (tsne - x_min) / (x_max - x_min)

        plt.plot(tsne_norm[train_label == 0][:,0], tsne_norm[train_label == 0][:,1], 'r.', label='benign')
        plt.plot(tsne_norm[train_label == 1][:,0], tsne_norm[train_label == 1][:,1], 'b.', label='maligancy')
        plt.legend()
        plt.title('add_'+str(fold)+'_train_t-sne')
        plt.savefig('data/feature/vis_feature/add'+str(fold)+'mask<=20_train_t-sne.png')
        plt.cla()

        tsne = TSNE(n_components=2).fit_transform(test)
        x_min, x_max = tsne.min(0), tsne.max(0)
        tsne_norm = (tsne - x_min) / (x_max - x_min)

        plt.plot(tsne_norm[test_label == 0][:,0], tsne_norm[test_label == 0][:,1], 'r.', label='benign')
        plt.plot(tsne_norm[test_label == 1][:,0], tsne_norm[test_label == 1][:,1], 'b.', label='maligancy')
        plt.legend()
        plt.title('add_'+str(fold)+'_test_t-sne')
        plt.savefig('data/feature/vis_feature/add'+str(fold)+'mask<=20_test_t-sne.png')
        plt.cla()

    
    
    return


#制作128*128大小的5折交叉验证数据集
from shutil import copy
def make_128_npy_mask_5_fold_dataset():
    dest_path = 'data/5fold_128_mask'
    mask_path = 'data/nodules3d_128_mask_npy'
    mask_npy = glob.glob(mask_path+'/*/*.npy')
    for fold in range(5):
        if not os.path.exists(dest_path + '/fold' + str(fold+1)):
            os.mkdir(dest_path + '/fold' + str(fold+1))
        if not os.path.exists(dest_path + '/fold' + str(fold+1) + '/train'):
            os.mkdir(dest_path + '/fold' + str(fold+1) + '/train')
        if not os.path.exists(dest_path + '/fold' + str(fold+1) + '/test'):
            os.mkdir(dest_path + '/fold' + str(fold+1) + '/test')
        
        path = 'data/5fold_128/fold' + str(fold+1)
        npy = glob.glob(path+'/*/*.npy')
        for one_npy in npy:
            if fold == 0:
                npy_num = one_npy.split('/')[-1].split('_')[0]
                npy_name = one_npy.split('/')[-1]
                dataset_split = one_npy.split('/')[-2]
                one_dest_path = os.path.join(dest_path, 'fold' + str(fold+1) + '/' + dataset_split + '/' + npy_name)
                for mask_one_npy in mask_npy:
                    if npy_num == mask_one_npy.split('/')[-1].split('_')[0]:
                        copy(mask_one_npy, one_dest_path)
            else:
                patient = one_npy.split('/')[-1].split('_')[0]
                nodule = one_npy.split('/')[-1].split('_')[1]
                npy_name = one_npy.split('/')[-1]
                dataset_split = one_npy.split('/')[-2]
                one_dest_path = os.path.join(dest_path, 'fold' + str(fold+1) + '/' + dataset_split + '/' + npy_name)
                for mask_one_npy in mask_npy:
                    if patient + nodule == mask_one_npy.split('/')[-1].split('_')[0]:
                        copy(mask_one_npy, one_dest_path)
    
    return

def get_different_5flod_128_with_5fold_128_mask():
    for fold in range(5):
        path = 'data/5fold_128/fold' + str(fold+1) + '/*/*npy'
        path_mask = 'data/5fold_128_mask/fold' + str(fold+1) + '/*/*npy'
        npy = glob.glob(path)
        npy_mask = glob.glob(path_mask)
        for index, one_npy in enumerate(npy):
            one_npy = one_npy.split('/')[-1]
            npy[index] = one_npy
        for index, one_npy in enumerate(npy_mask):
            one_npy = one_npy.split('/')[-1]
            npy_mask[index] = one_npy
        
        print((fold+1), 'npy中有，而npy_mask中没有的：', list(set(npy_mask).difference(set(npy))))
    return

from endToEnd.data_loader import fetch_dataloader
#每折都提取特征太慢了，这里手动组合mask第5折的特征
def get_fold_5_mask_feature():
    get_fold_feature = 5
    fold_3_train_feature = torch.load('data/feature/addition_feature_mask/fold_'+str(get_fold_feature)+'_train_addition_feature.pt')
    fold_3_test_feature = torch.load('data/feature/addition_feature_mask/fold_'+str(get_fold_feature)+'_test_addition_feature.pt')

    dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 700, data_dir='data/5fold_128_mask/fold'+str(get_fold_feature+1), train_shuffle=False, fold= 3)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']
    file_name_list_train_4 = []
    file_name_list_test_4 = []

    for i, (_, _, file_name, _) in enumerate(train_dl):
        file_name_list_train_4 = file_name
    for i, (_, _, file_name, _) in enumerate(test_dl):
        file_name_list_test_4 = file_name
    

    
    dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 1, data_dir="data/5fold_128<=20mm_mask/fold"+str(get_fold_feature+1), train_shuffle=False, fold= 3)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']
    feature_train_5 = torch.zeros((len(train_dl),255))
    feature_test_5 = torch.zeros((len(test_dl),255))
    flag = 0
    count = 0
    for i, (_, _, file_name,_) in enumerate(train_dl):
        for index, temp_name in enumerate(file_name_list_train_4):
            if temp_name == file_name[0]:
                print('find {0} in train set'.format(file_name))
                count += 1
                feature_train_5[i] = fold_3_train_feature[index]
                break
            else:
                flag += 1
        for index, temp_name in enumerate(file_name_list_test_4):
            if temp_name == file_name[0]:
                print('find {0} in test set'.format(file_name))
                count += 1
                feature_train_5[i] = fold_3_test_feature[index]
                break
            else:
                flag += 1
        if flag == 2:
            print('train set nodule-- ' + file_name[0] + ' --not found in fold '+str(get_fold_feature+1)+' train and test set..........')
        else:
            flag = 0         

    flag = 0
    for i, (_, _, file_name,_) in enumerate(test_dl):
        for index, temp_name in enumerate(file_name_list_train_4):
            if temp_name == file_name[0]:
                print('find {0} in train set'.format(file_name))
                count += 1
                feature_test_5[i] = fold_3_train_feature[index]
                break
            else:
                flag += 1
        for index, temp_name in enumerate(file_name_list_test_4):
            if temp_name == file_name[0]:
                print('find {0} in test set'.format(file_name))
                count += 1
                feature_test_5[i] = fold_3_test_feature[index]
                break
            else:
                flag += 1
        if flag == 2:
            print('test set nodule-- ' + file_name[0] + ' --not found in fold '+str(get_fold_feature+1)+' train and test set..........')
        else:
            flag = 0  
    
    print('find ' + str(count) + ' nodules')
    torch.save(feature_train_5, './data/feature/addition_feature_mask<=20/fold_' + str(get_fold_feature) + '_train_addition_feature.pt')
    torch.save(feature_test_5, './data/feature/addition_feature_mask<=20/fold_' + str(get_fold_feature) + '_test_addition_feature.pt')
        


    return

# 将data/nodules3d_128_npy_no_same_patient_in_two_dataset路径下的结节处理成2d的图片
def npy2png():
    for fold in range(1,6):
        npy_list = glob.glob('data/5fold_128/fold'+str(fold)+'/*/*npy')
        dest_path = 'data/5fold_128_2d/fold'+str(fold)
        for npy_path in npy_list:
            npy = np.load(npy_path)
            npy_name = npy_path.split('/')[-1].split('.')[0]
            split = npy_path.split('/')[-2]
            for index in range(8):
                save_path = dest_path + '/' + split + '/' + npy_name + '_slice' + str(index) + '.png'
                print(save_path)
                plt.imsave(save_path, npy[:, :, index], cmap='gray')
    return 

#统计所有结节直径分布
def nodule_diameter_statistic():
    f1 = open('data/pylidc_feature.csv','r')
    reader = csv.DictReader(f1)
    diameter_list = []
    benign_diameter_list = []
    malignancy_diameter_list = []
    for row in reader:
        if row['malignant_s'] == 'True':
            malignancy_diameter_list.append(float(row['diameter']))
        elif row['malignant_s'] == 'False': 
            benign_diameter_list.append(float(row['diameter']))
        diameter_list.append(float(row['diameter']))

    
    group = np.arange(3,50,1)
    plt.hist(diameter_list, group,rwidth=0.85, color='#0504aa', label='Total', alpha=0.5)

    plt.hist(benign_diameter_list, group,rwidth=0.85, color='green', label='Benign', alpha=0.5)

    plt.hist(malignancy_diameter_list, group,rwidth=0.85, color='red', label='Malignancy', alpha=0.5)
    plt.grid(axis='y', alpha=0.75)

    plt.title('comb nodule diameter statistic.png')
    plt.legend()
    plt.savefig('data/comb_nodule_diameter_statistic.png')

    return

#统计分错结节的直径分布
def mistake_classification_statistic():
    diameter_list=[
        15.08844368,9.608104591,9.228279528,11.03341783,11.94392301,11.38997126,17.11168034,8.576695168,14.25112484,11.54972505,26.0278968,7.851465329,8.734640537,
        7.923046018,8.233029569,11.10364875,14.68168103,8.229867125,8.463862173,12.73413535,16.06151701,13.86602164,9.507920891,11.92859435,9.566399622,7.239114943,8.035873766,15.61190858,8.125,13.64394468,8.963127887,7.70681113,8.140110846,17.32865356,9.699476358,9.761105986,41.67579511,21.147519,6.114248376,
        7.126096407,10.09484819,9.943880924,11.47646212,9.59983727,16.93381902,4.843865682,7.146090192,8.967885684,6.638326808,8.2979141,9.798621793,8.077963999,9.573305463,7.36328125,7.60634244,7.27180543,8.181000554,12.39780613,7.565143974,6.814326839,6.681997216,23.81578069,13.1051092,10.27870161,6.174772177,21.35000595,7.170496191,9.39453125,11.70233225,6.662938691,8.893905919,6.858862091,8.478503115,9.418625709,11.98361359,
        10.93500669,9.675715015,12.45940674,13.40022986,7.393060481,9.512532172,10.546875,6.594653606,9.17138039,26.45116373,9.793398007,8.445280847,11.38379241,8.582734417,10.06721794,10.546875,8.016858458,6.633267983,6.79671336,12.42225886,8.301796813,14.83243812,14.41067457,8.105915839,7.424444457,17.29077129,
        10.49359408,8.705242673,9.140625,9.248165464,10.99219028,5.995655394,15.72235297,17.58788791,15.82373613,11.97543873,9.36465269,4.922815726,18.7056647,8.638068534,7.424444457
    ]
    group = np.arange(3,50,1)
    plt.hist(diameter_list, group,rwidth=0.85, color='#0504aa', label='Total', alpha=0.5)
    plt.title('mistake classify nodule diameter statistic.png')
    plt.savefig('data/mistake_nodule_diameter_statistic.png')


#制作小于20mm的肺结节5折交叉验证数据集获取结节直径工具
def get_nodule_diameter(noduleName):
    f = open('data/pylidc_feature.csv','r')
    reader = csv.DictReader(f)
    for row in reader:
        if row['nodule_idx'] == noduleName:
            diameter = row['diameter']
            break
    return float(diameter)

#统计一折下面训练集测试集的良恶性结节数量
def get_fold_nodule_BorM_number(fold_path, isAug=False):
    npyPathList = glob.glob(fold_path + '/*/*npy')
    train_Benign = 0
    train_Malignancy = 0
    test_Benign = 0
    test_Malignancy = 0
    for oneNpyPath in npyPathList:
        split = oneNpyPath.split('/')[-2]
        if isAug:
            #对于增强的数据集
            label = oneNpyPath.split('/')[-1].split('_')[2].split('.')[0]
        else:
            #对于没增强的数据集
            label = oneNpyPath.split('_')[-1].split('.')[0]
        
        if split == 'train':
            if label == '1':
                train_Malignancy += 1
            if label == '0':
                train_Benign += 1
        if split == 'test':
            if label == '1':
                test_Malignancy += 1
            if label == '0':
                test_Benign += 1
    print('{0} trainSet: {1}Benign,{2}Maligancy; testSet: {3}Benign,{4}Maligancy'.format(fold_path, train_Benign, train_Malignancy, test_Benign, test_Malignancy))
    return train_Benign, train_Malignancy, test_Benign, test_Malignancy

#制作小于20mm的肺结节5折交叉验证数据集
def make_smaller_than_20mm_nodule_dataset():
    sourceFoldpath = 'data/5fold_128_mask'
    destFoldpath = 'data/5fold_128<=20mm_mask'
    for fold in range(5):
        npyPathList = glob.glob(sourceFoldpath + '/fold' + str(fold+1) + '/*/*npy')
        for oneNpyPath in npyPathList:
            if fold == 0:
                noduleName = oneNpyPath.split('/')[-1].split('_')[0]
            else:
                noduleName = oneNpyPath.split('/')[-1].split('_')[0] + oneNpyPath.split('/')[-1].split('_')[1]
            fileName = oneNpyPath.split('/')[-1]
            split = oneNpyPath.split('/')[-2]
            diameter = get_nodule_diameter(noduleName)
            #小于20mm的结节才复制
            if diameter < 20:
                destFileName = destFoldpath + '/fold' + str(fold+1) + '/' + split + '/' + fileName
                copy(oneNpyPath, destFileName) 
        get_fold_nodule_BorM_number(sourceFoldpath + '/fold' + str(fold+1))
        get_fold_nodule_BorM_number(destFoldpath + '/fold' + str(fold+1))
        print('\n')
    return

from PIL import Image
import random
 
def randomRotation(npy, angle):
    '''
    输入npy cube，返回旋转之后的npy cube
    '''
    newNpy = np.zeros_like(npy)
    for i in range(8):
        slice = npy[:, :, i]
        pil_img = Image.fromarray(np.uint8(slice))
        # plt.imsave('test/origi.png', pil_img,cmap='gray')
        new_pil_img = pil_img.rotate(angle, Image.BICUBIC)
        # plt.imsave('test/origi_rotat_'+str(angle)+'.png', new_pil_img,cmap='gray')
        newNpy[:, :, i] = new_pil_img
    return newNpy

def randomFlip(npy):
    newNpy = np.zeros_like(npy)
    for i in range(8):
        slice = npy[:, :, i]
        pil_img = Image.fromarray(np.uint8(slice))
        new_pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        # plt.imsave('test/origi_flip_FLIP_LEFT_RIGHT.png', new_pil_img,cmap='gray')
        newNpy[:, :, i] = new_pil_img
    return newNpy


def randomCrop(npy):
    '''
    先裁剪中间的96*96的区域，然后resize成128*128的
    '''
    newNpy = np.zeros_like(npy)
    for i in range(8):
        random_region = (24,24,104,104)
        slice = npy[:, :, i]
        pil_img = Image.fromarray(np.uint8(slice))
        new_pil_img = pil_img.crop(random_region)
        resize_pil_img = new_pil_img.resize((128,128))
        newNpy[:, :, i] = resize_pil_img
        # plt.imsave('test/origi_crop.png', new_pil_img,cmap='gray')
        # plt.imsave('test/origi_crop_resize.png', resize_pil_img,cmap='gray')
    return newNpy
    

def randomGaussian(npy, mean=10, sigma=5):
    def gaussianNoisy(im, mean, sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    newNpy = np.zeros_like(npy)
    for i in range(8):
        slice = npy[:, :, i]
        slice.flags.writeable = True
        slice_flatten = gaussianNoisy(slice.flatten(), mean, sigma)
        # plt.imsave('test/origin_Gaussian.png', slice_flatten.reshape([128,128]),cmap='gray')
        newNpy[:, :, i] = slice_flatten.reshape([128,128])
    return newNpy

#数据增强,当前就做第三折的数据增强
def dataset_5fold_auto_augment():
    sourcePath = 'data/5fold_128<=20mm_mask'
    destPath = 'data/5fold_128<=20mm_mask_aug'
    for fold in range(1,6):
        sourceFoldPath = sourcePath + '/fold' + str(fold)
        augTrainPath = destPath + '/fold' + str(fold) + '/train' + '/'
        train_Benign, train_Malignancy, test_Benign, test_Malignancy = get_fold_nodule_BorM_number(sourceFoldPath)
        trainNpyPathList = glob.glob(sourceFoldPath + '/train/*npy')

        for oneNpyPath in trainNpyPathList:
            print(oneNpyPath)
            npyName = oneNpyPath.split('/')[-1]
            noduleName = oneNpyPath.split('/')[-1].split('.')[0]
            label = oneNpyPath.split('_')[-1].split('.')[0]
            #先复制一份训练集
            copy(oneNpyPath, augTrainPath + npyName)
            npy = np.load(oneNpyPath)

            #恶性结节做5组增强，良性做3组
            if label == '1':
                afterRotat180Npy = randomRotation(npy, 180)
                np.save(augTrainPath + noduleName + '_rotate180.npy', afterRotat180Npy)
                afterRotat90Npy = randomRotation(npy, 90)
                np.save(augTrainPath + noduleName + '_rotate90.npy', afterRotat90Npy)
                afterFlip = randomFlip(npy)
                np.save(augTrainPath + noduleName + '_Flip.npy', afterFlip)
                afterCrop = randomCrop(npy)
                np.save(augTrainPath + noduleName + '_Crop.npy', afterCrop)
                afterGuassian = randomGaussian(npy)
                np.save(augTrainPath + noduleName + '_Guassian.npy', afterGuassian)
            elif label == '0':
                afterRotat180Npy = randomRotation(npy, 180)
                np.save(augTrainPath + noduleName + '_rotate180.npy', afterRotat180Npy)
                # afterCrop = randomCrop(npy)
                # np.save(augTrainPath + noduleName + '_Crop.npy', afterCrop)
                afterGuassian = randomGaussian(npy)
                np.save(augTrainPath + noduleName + '_Guassian.npy', afterGuassian)
        
        #测试集copy
        augTestPath = destPath + '/fold' + str(fold) + '/test' + '/'
        testNpyPathList = glob.glob(sourceFoldPath + '/test/*npy')
        for oneNpyPath in testNpyPathList:
            print(oneNpyPath)
            npyName = oneNpyPath.split('/')[-1]
            copy(oneNpyPath, augTestPath + npyName)


        get_fold_nodule_BorM_number(destPath + '/fold' + str(fold),True)        
        

    return

#对第一折文件
def rename_fold1():
    npyPathList = glob.glob('data/5fold_128<=20mm_mask/fold1/*/*npy')
    for onePath in npyPathList:
        fisrtPart = onePath[:-8]
        secondPart = onePath[-8:]
        renamePath = fisrtPart + '_' + secondPart
        print(renamePath)
        os.rename(onePath, renamePath)
    return


#统计一折结节直径分布
def nodule_fold_diameter_statistic():
    for fold in range(5):
        npyList = glob.glob('data/5fold_128<=20mm_aug/fold'+str(fold+1)+'/*/*npy')
        train_benign_diameter_list = []
        train_malignancy_diameter_list = []
        test_benign_diameter_list = []
        test_malignancy_diameter_list = []
        for oneNpyPath in npyList:
            # if fold == 0:
            #     noduleName = oneNpyPath.split('/')[-1].split('_')[0]
            #     label = oneNpyPath.split('/')[-1].split('_')[1].split('.')[0]
            # else:
            noduleName = oneNpyPath.split('/')[-1].split('_')[0] + oneNpyPath.split('/')[-1].split('_')[1]
            label = oneNpyPath.split('/')[-1].split('_')[2].split('.')[0]
            split = oneNpyPath.split('/')[-2]
            diameter = get_nodule_diameter(noduleName)
            if split == 'train':
                if label == '1':
                    train_malignancy_diameter_list.append(diameter)
                if label == '0':
                    train_benign_diameter_list.append(diameter)
            if split == 'test':
                if label == '1':
                    test_malignancy_diameter_list.append(diameter)
                if label == '0':
                    test_benign_diameter_list.append(diameter)
        group = np.arange(3,50,1)
        plt.subplot(121)
        plt.hist(train_malignancy_diameter_list, group,rwidth=0.85, color='#0504aa', label='train_malignancy', alpha=0.5)
        plt.hist(train_benign_diameter_list, group,rwidth=0.85, color='green', label='train_benign', alpha=0.5)  
        plt.title('fold_'+str(fold+1)+'_train')
        plt.legend()
        # plt.cla()

        plt.subplot(122) 
        plt.hist(test_malignancy_diameter_list, group,rwidth=0.85, color='#0504aa', label='test_malignancy', alpha=0.5)
        plt.hist(test_benign_diameter_list, group,rwidth=0.85, color='green', label='test_benign', alpha=0.5)   

        plt.title('fold_'+str(fold+1)+'_test')
        plt.legend()
        plt.savefig('data/fig/fold'+str(fold+1)+'_nodule_diameter_aug_distribute.png')    

        plt.clf()

    return

from sklearn.tree import DecisionTreeClassifier
#SVM、决策树
def traditional_feature_traditional_method_classification():
    rootPath = 'data/feature/addition_feature_mask<=20/'
    set_logger(rootPath + 'SVM_DecisionTreeClassifier.log')
    for fold in range(3,4):

        dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 3000, data_dir="data/5fold_128/fold"+str(fold+1), train_shuffle=False,fold=0)
        train_dl = dataloaders['train']
        test_dl = dataloaders['test']
        for i, (train_batch, labels_batch, file_name, _) in enumerate(train_dl):
            train_label = np.array(labels_batch)
            train_feature = getEightLabelFeature(file_name)
        for i, (train_batch, labels_batch, file_name, _) in enumerate(test_dl):
            test_label = np.array(labels_batch)
            test_name = file_name
            test_feature = getEightLabelFeature(file_name)

        
        kernel_function = ['linear', 'poly', 'rbf', 'sigmoid']
        C = [1e-2,1e-1,1,1e1,1e2] #C是对错误的惩罚
        gamma = [0.0001,0.0005,0.001,0.005,0.01,0.1] 
        max_iter = [50,100,200,500,700,1000,1500,2000]
        with tqdm(total = len(kernel_function)*len(C)*len(gamma)*len(max_iter)) as t:
            train_feature = train_feature.detach().numpy()
            test_feature = test_feature.detach().numpy()
            best_accuracy = 0
            for kf in kernel_function:
                for c in C:
                    for ga in gamma:
                        for miter in max_iter:
                            clf = SVC(kernel=kf, C=c, gamma=ga, max_iter=miter)
                            clf = clf.fit(train_feature, train_label)

                            Y_pred = clf.predict(test_feature)
                            con_matrix = confusion_matrix(test_label,Y_pred,labels=range(2))
                            
                            accuracy = (con_matrix[0][0] + con_matrix[1][1])/len(test_label)
                            if accuracy > best_accuracy:
                                idx = [i for i in range(len(Y_pred)) if Y_pred[i] != test_label[i]]
                                wrong_classify = [name for i,name in enumerate(test_name) if i in idx]
                                save_matrix = con_matrix
                                param_list = [c, ga, miter,kf]
                                best_accuracy = accuracy
                            t.update()
            
        TN = save_matrix[0][0]
        TP = save_matrix[1][1]
        FN = save_matrix[1][0]
        FP = save_matrix[0][1]
        logging.info('TN:{0}, TP:{1}, FN:{2}, FP:{3} '.format(TN, TP, FN, FP))
        logging.info('classify incorrectly nodule:')
        logging.info(wrong_classify)
        logging.info('{0} classification, kernel_function={1}, c={2}, gamma={3}, max_iter={4}, test_accuracy={5}'.format('SVM addition_feature<=20mm_mask', 
                                                                                                                                    param_list[3],
                                                                                                                                    param_list[0], 
                                                                                                                                    param_list[1], 
                                                                                                                                    param_list[2],
                                                                                                                                    best_accuracy))
        
        # criterion = ['entropy', 'gini']
        # splitter = ['best', 'random'] #C是对错误的惩罚
        # class_weight = {0:0.3,
        #                 1:0.7}
        # with tqdm(total = len(criterion)*len(splitter)) as t:
        #     best_accuracy = 0
        #     for cri in criterion:
        #         for spli in splitter:
        #             clf = DecisionTreeClassifier(criterion=cri, splitter=spli,class_weight=class_weight)
        #             clf = clf.fit(train_feature, train_label)

        #             Y_pred = clf.predict(test_feature)
        #             con_matrix = confusion_matrix(test_label,Y_pred,labels=range(2))
        #             accuracy = (con_matrix[0][0] + con_matrix[1][1])/len(test_label)
        #             if accuracy > best_accuracy:
        #                 idx = [i for i in range(len(Y_pred)) if Y_pred[i] != test_label[i]]
        #                 wrong_classify = [name for i,name in enumerate(test_name) if i in idx]
        #                 save_matrix = con_matrix
        #                 param_list = [cri, spli]
        #                 best_accuracy = accuracy
        #             t.update()

        # TN = save_matrix[0][0]
        # TP = save_matrix[1][1]
        # FN = save_matrix[1][0]
        # FP = save_matrix[0][1]
        # logging.info('TN:{0}, TP:{1}, FN:{2}, FP:{3} '.format(TN, TP, FN, FP))
        # logging.info('classify incorrectly nodule:')
        # logging.info(wrong_classify)
        # logging.info('{0} classification, criterion={1}, splitter={2}, test_accuracy={3}'.format('DecisionTreeClassifier addition_feature<=20mm_mask', 
        #                                                                                 param_list[0],
        #                                                                                 param_list[1],
        #                                                                                 best_accuracy))
        # logging.info('\n')                                                                            

        

    return

#得到8个标签的特征值
def getEightLabelFeature(noudleFileName):
    df = pd.read_csv('./data/pylidc_feature.csv')
    feature = torch.zeros((len(noudleFileName), 38))
    for index, oneNoduleFileName in enumerate(noudleFileName):
        nodule_idx = int(oneNoduleFileName.split('_')[0])
        # nodule_idx = int(oneNoduleFileName.split('_')[0] + oneNoduleFileName.split('_')[1])

        sublety = np.array([df[df["nodule_idx"]==nodule_idx]["sublety_mean"].iloc[0]])-1
        internalstructure = np.array([df[df["nodule_idx"]==nodule_idx]["internalstructure_mean"].iloc[0]])-1
        calcification = np.array([df[df["nodule_idx"]==nodule_idx]["calcification_mean"].iloc[0]])-1
        sphericity = np.array([df[df["nodule_idx"]==nodule_idx]["sphericity_mean"].iloc[0]])-1
        margin = np.array([df[df["nodule_idx"]==nodule_idx]["margin_mean"].iloc[0]])-1
        lobulation = np.array([df[df["nodule_idx"]==nodule_idx]["lobulation_mean"].iloc[0]])-1
        spiculation = np.array([df[df["nodule_idx"]==nodule_idx]["spiculation_mean"].iloc[0]])-1
        texture = np.array([df[df["nodule_idx"]==nodule_idx]["texture_mean"].iloc[0]])-1

        sublety_feature = torch.zeros((5))
        sublety_feature[sublety] = 1
        internalstructure_feature = torch.zeros((3))
        internalstructure_feature[internalstructure] = 1
        calcification_feature = torch.zeros((6))
        calcification_feature[calcification] = 1
        sphericity_feature = torch.zeros((5))
        sphericity_feature[sphericity] = 1
        margin_feature = torch.zeros((5))
        margin_feature[margin] = 1
        lobulation_feature = torch.zeros((4))
        lobulation_feature[lobulation] = 1
        spiculation_feature = torch.zeros((5))
        spiculation_feature[spiculation] = 1
        texture_feature = torch.zeros((5))
        texture_feature[texture] = 1
    
        feature[index] = torch.cat((sublety_feature, internalstructure_feature, calcification_feature, sphericity_feature, margin_feature, lobulation_feature, spiculation_feature, texture_feature))
    return feature

def getDatasetMeanAndStd():
    # dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 3000, data_dir="data/5fold_128<=20mm_aug/fold"+str(4), train_shuffle=False,fold=0)
    # train_dl = dataloaders['train']
    # test_dl = dataloaders['test']
    # for i, (train_batch, labels_batch, file_name, _) in enumerate(train_dl):
    #     train_batch = train_batch
    # for i, (test_batch, labels_batch, file_name, _) in enumerate(test_dl):
    #     test_batch = test_batch
    # total_batch = torch.cat((train_batch, test_batch), dim=0)
    # channel_1_mean = torch.mean(total_batch[:,:,0,:,:])
    # channel_2_mean = torch.mean(total_batch[:,:,1,:,:])
    # channel_3_mean = torch.mean(total_batch[:,:,2,:,:])
    # channel_4_mean = torch.mean(total_batch[:,:,3,:,:])
    # channel_5_mean = torch.mean(total_batch[:,:,4,:,:])
    # channel_6_mean = torch.mean(total_batch[:,:,5,:,:])
    # channel_7_mean = torch.mean(total_batch[:,:,6,:,:])
    # channel_8_mean = torch.mean(total_batch[:,:,7,:,:])
    # channel_1_std = torch.std(total_batch[:,:,0,:,:])
    # channel_2_std = torch.std(total_batch[:,:,1,:,:])
    # channel_3_std = torch.std(total_batch[:,:,2,:,:])
    # channel_4_std = torch.std(total_batch[:,:,3,:,:])
    # channel_5_std = torch.std(total_batch[:,:,4,:,:])
    # channel_6_std = torch.std(total_batch[:,:,5,:,:])
    # channel_7_std = torch.std(total_batch[:,:,6,:,:])
    # channel_8_std = torch.std(total_batch[:,:,7,:,:])

    mean = torch.tensor([[92.4995, 92.3400, 92.2389, 92.1519, 91.9990, 91.7405, 91.4233, 91.0724]])
    std = torch.tensor([[41.8699, 41.6722, 41.5907, 41.4606, 41.3476, 41.1476, 40.9845, 40.8674]])
    return mean,std

if __name__ == '__main__':
    caculate_five_method_predict_similarity()