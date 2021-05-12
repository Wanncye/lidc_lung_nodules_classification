import json
import logging
import os
import shutil
from matplotlib import cm
from skimage.feature import greycomatrix, greycoprops

import torch
import glob
from tqdm.std import tqdm

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
from model.threeDresnet_feature import generate_model
from model.threeDVGG_feature import vgg16_bn
from model.threeDGoogleNet_feature import googlenet
torch.cuda.set_device(0)
from matplotlib import pyplot as plt

from model.graphNet import GAT,GCN

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


def save_checkpoint(state, is_best, checkpoint, N_folder, params):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'folder.'+ str(N_folder)+ '.' +params.loss +'_alpha_'+str(params.FocalLossAlpha)+'.best.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    # torch.save(state, filepath)  #不保存最后的
    if is_best:
        torch.save(state, filepath)
        # shutil.copyfile(filepath, os.path.join(checkpoint, 'folder.'+ str(N_folder)+ '.' +params.loss +'_alpha_'+str(params.FocalLossAlpha)+'.best.pth.tar'))


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
    start_time = time.time()
    print('Started......lbp')
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
    finish_time = time.time()
    print('Finished......')
    print('using time:',finish_time-start_time)

    max_bins = int(max(feature) + 1)
    nor_feature, _ = np.histogram(feature, density=False, bins=512, range=(0, max_bins))
    nor_feature = normalization(nor_feature)
    return nor_feature
    
def HOG(cube):
    cube = cube.squeeze(0).squeeze(0)
    all_gredient = []
    start_time = time.time()
    print('Started......hog')
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
    finish_time = time.time()
    print('Finished......')
    print('using time:',finish_time-start_time)

    gredient_max = np.max(all_gredient)
    gredient_min = np.min(all_gredient)
    nor_feature, _ = np.histogram(all_gredient, density=True, bins=512, range=(gredient_min, gredient_max))
    nor_feature = normalization(nor_feature)
    return nor_feature

def GLCM(cube):
    cube = cube.squeeze(0).squeeze(0)
    feature = np.zeros(512)
    for index in range(8):
        img = torch.tensor(cube[index],dtype=torch.int)
        glcm = greycomatrix(img,            #共生矩阵（代码中glcm）为四维，前两维表示行列，后两维分别表示距离和角度
                            [1, 2, 4, 8],      #距离
                            [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],  #方向
                            256,        #灰度级别
                            symmetric=True,     #是否对称
                            normed=True)        #是否标准化
        attribute = ['contrast', 'dissimilarity', 'energy', 'correlation']
        for jndex, prop in enumerate(attribute):
            temp = greycoprops(glcm, prop).flatten()      #输出的每种特征（代码中temp）行表示距离，列表示角度。
            feature[index*64 + jndex*16 : index*64 + (jndex+1)*16] = temp
    return feature

def feature_extract():
    # 提取两种类型的特征：一种是非mask的结节特征，另一种是mask的结节特征。
    # 共提取6种特征，即resnet,vgg,googlenet,lbp,hog,glcm
    batch_size = 1
    dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = batch_size, data_dir='./data/nodules3d_128_mask_npy', train_shuffle=False)
    train_dl = dataloaders['train']
    train_len = 639
    test_dl = dataloaders['test']
    test_len = 160
    feature_len = 512
    googlenet_train_feature = torch.zeros((train_len,feature_len))
    googlenet_test_feature = torch.zeros((test_len,feature_len))
    resnet_train_feature = torch.zeros((train_len,feature_len))
    resnet_test_feature = torch.zeros((test_len,feature_len))
    vgg_train_feature = torch.zeros((train_len,feature_len))
    vgg_test_feature = torch.zeros((test_len,feature_len))
    hog_train_feature = torch.zeros((train_len,feature_len))
    hog_test_feature = torch.zeros((test_len,feature_len))
    lbp_train_feature = torch.zeros((train_len,feature_len))
    lbp_test_feature = torch.zeros((test_len,feature_len))
    glcm_train_feature = torch.zeros((train_len,feature_len))
    glcm_test_feature = torch.zeros((test_len,feature_len))
    train_label = torch.zeros((train_len))
    test_label = torch.zeros((test_len))


    # lbp提取特征
    with tqdm(total=len(train_dl)+len(test_dl)) as t:
        for i, (x, target, _) in enumerate(train_dl):
            feature = torch.from_numpy(LBP(x))
            lbp_train_feature[i] = feature
            t.update()
        for i, (x, target, _) in enumerate(test_dl):
            feature = torch.from_numpy(LBP(x))
            lbp_test_feature[i] = feature
            t.update()
            
    # hog提取特征
    with tqdm(total=len(train_dl)+len(test_dl)) as t:
        for i, (x, target, _) in enumerate(train_dl):
            feature = torch.from_numpy(HOG(x))
            hog_train_feature[i] = feature
            t.update()
        for i, (x, target, _) in enumerate(test_dl):
            feature = torch.from_numpy(HOG(x))
            hog_test_feature[i] = feature
            t.update()

    # glcm提取特征
    with tqdm(total=len(train_dl)+len(test_dl)) as t:
        for i, (x, target, _) in enumerate(train_dl):
            feature = torch.from_numpy(GLCM(x))
            glcm_train_feature[i] = feature
            t.update()
        for i, (x, target, _) in enumerate(test_dl):
            feature = torch.from_numpy(GLCM(x))
            glcm_test_feature[i] = feature
            t.update()

    # #googlenet提取特征
    model = googlenet()
    #strict=False 使得预训练模型参数中和新模型对应上的参数会被载入，对应不上或没有的参数被抛弃
    model.load_state_dict(torch.load('./experiments/GoogleNet_no_mask/folder.0.FocalLoss_alpha_0.25.best.pth.tar'), strict=False)
    with tqdm(total=len(train_dl)+len(test_dl)) as t:
        for i, (x, target, _) in enumerate(train_dl):
            feature = model(x)
            googlenet_train_feature[(i*batch_size):((i+1)*batch_size), :] = feature.detach()
            train_label[i*batch_size:(i+1)*batch_size] = target.detach()
            t.update()
        for i, (x, target, _) in enumerate(test_dl):
            feature = model(x)
            googlenet_test_feature[(i*batch_size):((i+1)*batch_size), :] = feature.detach()
            test_label[i*batch_size:(i+1)*batch_size] = target.detach()
            t.update()

    # #resnet50提取特征
    model = generate_model(50)
    #strict=False 使得预训练模型参数中和新模型对应上的参数会被载入，对应不上或没有的参数被抛弃
    model.load_state_dict(torch.load('./experiments/resnet50_no_mask/folder.0.FocalLoss_alpha_0.25.best.pth.tar'), strict=False)
    with tqdm(total=len(train_dl)+len(test_dl)) as t:
        for i, (x, target, _) in enumerate(train_dl):
            feature = model(x)
            resnet_train_feature[(i*batch_size):((i+1)*batch_size), :] = feature.detach()
            t.update()
        for i, (x, target, _) in enumerate(test_dl):
            feature = model(x)
            resnet_test_feature[(i*batch_size):((i+1)*batch_size), :] = feature.detach()
            t.update()

    # #resnet50提取特征
    model = vgg16_bn(0.5)
    #strict=False 使得预训练模型参数中和新模型对应上的参数会被载入，对应不上或没有的参数被抛弃
    model.load_state_dict(torch.load('./experiments/VGG16_no_mask/folder.0.FocalLoss_alpha_0.25.best.pth.tar'), strict=False)
    with tqdm(total=len(train_dl)+len(test_dl)) as t:
        for i, (x, target, _) in enumerate(train_dl):
            feature = model(x)
            vgg_train_feature[(i*batch_size):((i+1)*batch_size), :] = feature.detach()
            t.update()
        for i, (x, target, _) in enumerate(test_dl):
            feature = model(x)
            vgg_test_feature[(i*batch_size):((i+1)*batch_size), :] = feature.detach()
            t.update()

    torch.save(googlenet_train_feature,'./data/mask_feature/googlenet_train_feature.pt')
    torch.save(googlenet_test_feature,'./data/mask_feature/googlenet_test_feature.pt')
    torch.save(resnet_train_feature,'./data/mask_feature/resnet_train_feature.pt')
    torch.save(resnet_test_feature,'./data/mask_feature/resnet_test_feature.pt')
    torch.save(vgg_train_feature,'./data/mask_feature/vgg_train_feature.pt')
    torch.save(vgg_test_feature,'./data/mask_feature/vgg_test_feature.pt')
    torch.save(hog_train_feature,'./data/mask_feature/hog_train_feature.pt')
    torch.save(hog_test_feature,'./data/mask_feature/hog_test_feature.pt')
    torch.save(lbp_train_feature,'./data/mask_feature/lbp_train_feature.pt')
    torch.save(lbp_test_feature,'./data/mask_feature/lbp_test_feature.pt')
    torch.save(glcm_train_feature,'./data/mask_feature/glcm_train_feature.pt')
    torch.save(glcm_test_feature,'./data/mask_feature/glcm_test_feature.pt')
    torch.save(train_label,'./data/mask_feature/train_label.pt')
    torch.save(test_label,'./data/mask_feature/test_label.pt')

def numpy_to_tensor_and_save():
    hog_train_feature = torch.from_numpy(np.load('./data/feature/hog_train_feature.pt.npy'))
    hog_test_feature = torch.from_numpy(np.load('./data/feature/hog_test_feature.pt.npy'))
    lbp_train_feature = torch.from_numpy(np.load('./data/feature/lbp_train_feature.pt.npy'))
    lbp_test_feature = torch.from_numpy(np.load('./data/feature/lbp_test_feature.pt.npy'))
    torch.save(hog_train_feature,'./data/feature/hog_train_feature.pt')
    torch.save(hog_test_feature,'./data/feature/hog_test_feature.pt')
    torch.save(lbp_train_feature,'./data/feature/lbp_train_feature.pt')
    torch.save(lbp_test_feature,'./data/feature/lbp_test_feature.pt')

def svm_classification():
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    import warnings
    import logging
    warnings.filterwarnings('ignore')

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
                            clf = SVC(kernel=kf, C=c, gamma=ga, max_iter=miter)
                            clf = clf.fit(train_feature, train_label)

                            Y_pred = clf.predict(test_feature)
                            con_matrix = confusion_matrix(test_label,Y_pred,labels=range(2))
                            
                            accuracy = (con_matrix[0][0] + con_matrix[1][1])/160
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
            logging.info('{0} svm classification, kernel_function={1}, c={2}, gamma={3}, max_iter={4}, test_accuracy={5}'.format(feature_type, 
                                                                                                                                    param_list[3],
                                                                                                                                    param_list[0], 
                                                                                                                                    param_list[1], 
                                                                                                                                    param_list[2],
                                                                                                                                    best_accuracy))
                    
#使用训练的GCN模型来提取中间层特征用于svm分类和直方图观察
def gcn_feature():
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

if __name__ == '__main__':
    gcn_feature()
    # feature_extract()
    # numpy_to_tensor_and_save()
    # 制作5折交叉验证数据集
    # data_path = './data/nodules3d_128_npy'
    # dest_path = './data/nodules3d_128_npy_5_folders'
    # split_data_to_5folders(data_path,dest_path)
    # 画结果图
    # log_file = './experiments/VGG16_no_mask_5folders/train_epoch_why.log'
    # png_dir = './experiments/VGG16_no_mask_5folders/Visualize'
    # log_file = './experiments/resnet50_no_mask_5folders/train.log'
    # png_dir = './experiments/resnet50_no_mask_5folders/Visualize'
    # log_file = './experiments/VGG16_no_mask_5folders/train_FocalLoss_alpha_0.25_correct-alpha.log'
    # png_dir = './experiments/VGG16_no_mask_5folders/Visualize'
    # plot_figure(png_dir, log_file)
    