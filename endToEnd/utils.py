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

import visdom
import time
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt 
import re
from torch.autograd import Variable

import torch 
from tqdm import tqdm
# torch.cuda.set_device(0)
from matplotlib import pyplot as plt

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
    ?????????visdom?????????????????????????????????????????????`self.vis.function`
    ???????????????visdom??????
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # ???????????????????????????????????????
        # ????????????loss',23??? ???loss??????23??????
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        ??????visdom?????????
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        ??????plot??????
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
        ?????????don???t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~?????????
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
    ## ?????????????????????????????????????????????5??????????????????5????????????????????????????????????????????????5???
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

    #???5???TrainLoss??????????????????
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

    #??????50???epoch??????5???????????????????????????????????????50?????????50epoch??????????????????????????????????????????
    # GoogLeNet 50 ??????
    # ResNet 43 ??????
    # VGG16  0  ??????????????????????????????????????????0??????
    coverage_epoch = 20
    average_dic = {}
    for key in metric_dic['0']: #???????????????????????????????????????
        var_name = 'average_' + key
        average_dic[var_name] = [0 for i in range(100-coverage_epoch)]
        for index in folder: #?????????????????????????????????????????????
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

#????????????????????????
def binary_to_decimal(array):
    my_str = ''
    for element in array:
        my_str += str(element)
    return int(my_str, 2)

#????????????????????????
def normalization(array):
    array = np.array(array, dtype=np.float)
    array_max = np.max(array)
    array_min = np.min(array)
    distribute = array_max - array_min
    for index in range(len(array)):
        array[index] = float(array[index] - array_min) / float(distribute)
    return array

#???????????????3d LBP??????
def LBP(cube):
    cube = cube.squeeze(0).squeeze(0)
    feature=[]
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            for k in range(cube.shape[2]):
                middle_pixel = cube[i, j, k]
                binary_code = np.zeros(26, dtype=int)
                if i-1 < 0 or j-1 < 0 or k-1 < 0 or i+1 > cube.shape[0]-1 or j+1 > cube.shape[1]-1 or k+1 > cube.shape[2]-1:
                    #??????cube???????????????,?????????????????????????????????????????????????????????????????????
                    if i-1 < 0:
                        if j-1 < 0 or k-1 < 0  or j+1 > cube.shape[1]-1 or k+1 > cube.shape[2]-1:
                            pass
                        else:
                            #???????????????
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
                            #??????????????????
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
                    #??????cube???????????????,?????????????????????????????????????????????????????????????????????
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
                else:#???????????????,????????????????????????????????????????????????????????????????????????????????????????????????LBP??????
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
        # glcm = greycomatrix(img,            #????????????????????????glcm???????????????????????????????????????????????????????????????????????????
        #                     [1, 2, 4, 8],      #??????
        #                     [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],  #??????
        #                     256,        #????????????
        #                     symmetric=True,     #????????????
        #                     normed=True)        #???????????????
        # attribute = ['contrast', 'dissimilarity', 'energy', 'correlation']

        glcm = greycomatrix(img,            #????????????????????????glcm???????????????????????????????????????????????????????????????????????????
                            [1, 2],      #??????
                            [0, np.pi / 2],  #??????
                            256,        #????????????
                            symmetric=True,     #????????????
                            normed=True)        #???????????????
        attribute = ['contrast', 'correlation']

        for jndex, prop in enumerate(attribute):
            temp = greycoprops(glcm, prop).flatten()      #?????????????????????????????????temp???????????????????????????????????????
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
    # ??????????????????????????????????????????mask??????????????????????????????mask??????????????????
    # ?????????6???????????????resnet,vgg,googlenet,lbp,hog,glcm
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

    # # #googlenet????????????
    # model = googlenet()
    # #strict=False ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
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

#???6???????????????svm????????????,????????????????????????????????????csv?????????
def svm_classification():
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    

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

    #glcm????????????????????????
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
    #glcm max_iter=500 ????????????
    set_logger('./experiments/svm/log_mask.log')
    kernel_function = ['linear', 'poly', 'rbf', 'sigmoid']
    C = [1e-2,1e-1,1,1e1,1e2] #C?????????????????????
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
                                #???????????????????????????????????????csv
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
                    
#???????????????GCN????????????????????????????????????svm????????????????????????
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

#??????????????????GCN?????????64??????2???????????????????????????
def gcn_feature_line():
    train_middle_feature_64 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/train_middle_feature_64.pt')
    train_middle_feature_2 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/train_middle_feature_2.pt')
    test_middle_feature_64 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/test_middle_feature_64.pt')
    test_middle_feature_2 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/test_middle_feature_2.pt')
    train_label = torch.load('./data/mask_feature/train_label.pt')
    test_label = torch.load('./data/mask_feature/test_label.pt')

    # ????????? 64
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
        plt.clf()#?????????

    #????????? 2
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
        plt.clf()#?????????

    #????????? 64
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
        plt.clf()#?????????

    #????????? 2
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
        plt.clf()#?????????

#??????????????????GCN?????????64??????2???????????????????????????
def gcn_feature_histogram():
    train_middle_feature_64 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/train_middle_feature_64.pt')
    train_middle_feature_2 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/train_middle_feature_2.pt')
    test_middle_feature_64 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/test_middle_feature_64.pt')
    test_middle_feature_2 = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/test_middle_feature_2.pt')
    train_label = torch.load('./data/mask_feature/train_label.pt')
    test_label = torch.load('./data/mask_feature/test_label.pt')

    # ????????? 64
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
        plt.clf()#?????????

    #????????? 2
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
        plt.clf()#?????????

    #????????? 64
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
        plt.clf()#?????????

    #????????? 2
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
        plt.clf()#?????????

#??????svm???gcn??????????????????????????????
def svm_classification_gcn_middle_feature():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    
    gcn_train_feature = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/train_middle_feature_64.pt')[:,0,:]   #639*64
    gcn_test_feature = torch.load('./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar_feature/test_middle_feature_64.pt')[:,0,:]    #160*64
    train_label = torch.load('./data/mask_feature/train_label.pt')
    test_label = torch.load('./data/mask_feature/test_label.pt')
    
    
    #SVM
    #glcm max_iter=500 ????????????
    set_logger('./experiments/gcn/gcm_middle_feature_svm_classification_log.log')
    kernel_function = ['linear', 'poly', 'rbf', 'sigmoid']
    C = [1e-2,1e-1,1,1e1,1e2] #C?????????????????????
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
        logging.info('{0} svm classification, kernel_function={1}, c={2}, gamma={3}, max_iter={4}, test_accuracy={5}'.format('gcn', 
                                                                                                                                param_list[3],
                                                                                                                                param_list[0], 
                                                                                                                                param_list[1], 
                                                                                                                                param_list[2],
                                                                                                                                best_accuracy))

#??????6???????????????????????????????????????????????????????????????
def caculate_six_method_predict_similarity():
    #?????????6??????????????????????????????????????????????????????????????????????????????
    csv_reader = csv.reader(open('./data/mask_feature/glcm_testset_result.csv'))
    header = next(csv_reader)
    glcm_predict_list = []
    for row in csv_reader:
        glcm_predict_list.append(row[2])
    
    csv_reader = csv.reader(open('./data/mask_feature/googlenet_testset_result.csv'))
    header = next(csv_reader)
    googlenet_predict_list = []
    for row in csv_reader:
        googlenet_predict_list.append(row[2])

    csv_reader = csv.reader(open('./data/mask_feature/vgg_testset_result.csv'))
    header = next(csv_reader)
    vgg_predict_list = []
    for row in csv_reader:
        vgg_predict_list.append(row[2])

    csv_reader = csv.reader(open('./data/mask_feature/resnet_testset_result.csv'))
    header = next(csv_reader)
    resnet_predict_list = []
    for row in csv_reader:
        resnet_predict_list.append(row[2])
    
    csv_reader = csv.reader(open('./data/mask_feature/lbp_testset_result.csv'))
    header = next(csv_reader)
    lbp_predict_list = []
    for row in csv_reader:
        lbp_predict_list.append(row[2])

    csv_reader = csv.reader(open('./data/mask_feature/hog_testset_result.csv'))
    header = next(csv_reader)
    hog_predict_list = []
    for row in csv_reader:
        hog_predict_list.append(row[2])
    
    all_predict_list = [googlenet_predict_list, resnet_predict_list, vgg_predict_list, hog_predict_list, lbp_predict_list, glcm_predict_list]
    sim_matrix = np.zeros((6,6))
    cord = 0
    for i_predict_list in all_predict_list:
        for j_predict_list in all_predict_list:
            correct = 0
            for index in range(160):
                if i_predict_list[index] == j_predict_list[index]:
                    correct += 1
            row = int(cord / 6)
            coloum = cord % 6
            sim_matrix[row][coloum] = correct
            cord += 1
    return sim_matrix/160


#??????15????????????10????????????5???????????????5??????????????????512??????????????????txt?????????
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
        if train_label[i] == 0: # ?????????
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
        if train_label[i] == 1: # ?????????
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

#????????????????????????????????????
def get_matrix_similarity(_matrixA, _matrixB):
    _matrixA_matrixB = np.dot(_matrixA, _matrixB.transpose())
    _matrixA_norm = np.sqrt(np.multiply(_matrixA,_matrixA).sum(axis=1))
    _matrixB_norm = np.sqrt(np.multiply(_matrixB,_matrixB).sum(axis=1))
    return np.divide(_matrixA_matrixB, _matrixA_norm * _matrixB_norm.transpose())

def search_different_resnet_feature_correlation():
    matirx_a = torch.load('data/feature/resnet18_test.pt').numpy()
    matirx_b = torch.load('data/feature/resnet34_test.pt').numpy()
    print(matirx_b[29])
    print(matirx_b[0])
    sim = get_matrix_similarity(matirx_b, matirx_b)
    for i in range(160):
        print(sim[i][i], np.max(sim[i]), np.argmax(sim[i]))

#??????????????????????????????????????????????????????????????????????????????????????????
def get_dataset_label_pt():
    for index in range(5):
        dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 650, data_dir="data/5fold_128/fold"+str(index+1), train_shuffle=False)
        train_dl = dataloaders['train']
        test_dl = dataloaders['test']
        for i, (train_batch, labels_batch, _) in enumerate(train_dl):
            torch.save(labels_batch,'./data/feature/fold_'+str(index)+'_train_label.pt')
        for i, (train_batch, labels_batch, _) in enumerate(test_dl):
            torch.save(labels_batch,'./data/feature/fold_'+str(index)+'_test_label.pt')

#???????????????????????????????????????????????????????????????????????????????????????csv???
def converage_all_result():
    best_json = glob.glob(r'./experiments/*_nomask/*best*')
    print(len(best_json))

import json
#???????????????????????????????????????????????????gcn
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
    wanted_path.sort() #???????????????
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
        header = next(csv_reader) #csv???
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

#??????????????????????????????2d?????????????????????
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


#???????????????????????????????????????????????????????????????
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

#?????????????????????4???????????????????????????????????????????????????????????????
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
        print("????????????0??? {0}???1??? {1}???????????????0???{2}???1???{3}".format(train_negative, train_positive, test_negative, test_positive))
        print('='*15)

    #??????????????????????????????????????????????????????????????????????????????
    return

#8???2??? ????????????????????????????????????????????????????????????????????????????????????  
import pandas as pd
def get_various_feature():
    df = pd.read_csv('./data/pylidc_feature.csv')
    for fold in range(1, 5):
        dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 1, data_dir="data/5fold_128_mask/fold"+str(fold+1), train_shuffle=False, fold= fold)
        train_dl = dataloaders['train']
        test_dl = dataloaders['test']
        #255 = 64 * 3 + 56 +7
        feature_train = torch.zeros((len(train_dl),255))
        feature_test = torch.zeros((len(test_dl),255))
        for i, (cube, _, file_name, _) in enumerate(train_dl):
            print('---fold ' + str(fold) + '---extracting ' + file_name[0])
            if fold == 0: #??????5?????????????????????????????????????????????????????????
                patient = file_name[0].split('_')[0][:4]
                nodule = file_name[0].split('_')[0][4:]
            else:
                patient = file_name[0].split('_')[0]
                nodule = file_name[0].split('_')[1]
            nodule_idx = int(patient + nodule)
            diameter = np.array([df[df["nodule_idx"]==nodule_idx]["diameter"].iloc[0]])
            surface_area = np.array([df[df["nodule_idx"]==nodule_idx]["surface_area"].iloc[0]])
            volume = np.array([df[df["nodule_idx"]==nodule_idx]["volume"].iloc[0]])

            hu = HU(cube)     #56?????????
            glcm = GLCM(cube) #64??????????????????????????????
            lbp = LBP(cube)   #64?????????
            hog = HOG(cube)   #64?????????

            cube = np.squeeze(cube).numpy()
            mean = np.array([np.mean(cube)])   #?????????????????????????????????????????????????????????
            variance = np.array([np.var(cube)])
            pd_cube = pd.Series(cube.flatten())
            skewness = np.array([pd_cube.skew()])
            kurtosis = np.array([pd_cube.kurt()])
            temp_feature = np.concatenate((hu, glcm, lbp, hog, diameter, surface_area, volume, mean, variance, skewness, kurtosis))
            
            feature_train[i] = torch.from_numpy(temp_feature)

        for i, (cube, _, file_name, _) in enumerate(test_dl):
            if fold == 0: #??????5?????????????????????????????????????????????????????????
                patient = file_name[0].split('_')[0][:4]
                nodule = file_name[0].split('_')[0][4:]
            else:
                patient = file_name[0].split('_')[0]
                nodule = file_name[0].split('_')[1]
            nodule_idx = int(patient + nodule)
            diameter = np.array([df[df["nodule_idx"]==nodule_idx]["diameter"].iloc[0]])
            surface_area = np.array([df[df["nodule_idx"]==nodule_idx]["surface_area"].iloc[0]])
            volume = np.array([df[df["nodule_idx"]==nodule_idx]["volume"].iloc[0]])

            hu = HU(cube)     #56?????????
            glcm = GLCM(cube) #64??????????????????????????????
            lbp = LBP(cube)   #64?????????
            hog = HOG(cube)   #64?????????

            cube = np.squeeze(cube).numpy()
            mean = np.array([np.mean(cube)])   #?????????????????????????????????????????????????????????
            variance = np.array([np.var(cube)])
            pd_cube = pd.Series(cube.flatten())
            skewness = np.array([pd_cube.skew()])
            kurtosis = np.array([pd_cube.kurt()])
            temp_feature = np.concatenate((hu, glcm, lbp, hog, diameter, surface_area, volume, mean, variance, skewness, kurtosis))
            
            feature_test[i] = torch.from_numpy(temp_feature)

        torch.save(feature_train, './data/feature/addition_feature_mask/fold_' + str(fold) + '_train_addition_feature.pt')
        torch.save(feature_test, './data/feature/addition_feature_mask/fold_' + str(fold) + '_test_addition_feature.pt')
            
    return


from sklearn.manifold import TSNE 
def t_SNE():
    fold = 0
    dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 650, data_dir="data/5fold_128/fold"+str(fold+1), train_shuffle=False, fold=0)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']
    for i, (train_batch, labels_batch, _, _) in enumerate(train_dl):
        train_label = labels_batch
    for i, (train_batch, labels_batch, _, _) in enumerate(test_dl):
        test_label = labels_batch
    train = torch.load('data/feature/addition_feature/fold_0_train_addition_feature.pt')
    test = torch.load('data/feature/addition_feature/fold_0_test_addition_feature.pt')
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
    plt.title('add_0_train_t-sne')
    plt.savefig('data/feature/vis_feature/add_0_train_t-sne.png')
    plt.cla()

    tsne = TSNE(n_components=2).fit_transform(test)
    x_min, x_max = tsne.min(0), tsne.max(0)
    tsne_norm = (tsne - x_min) / (x_max - x_min)

    plt.plot(tsne_norm[test_label == 0][:,0], tsne_norm[test_label == 0][:,1], 'r.', label='benign')
    plt.plot(tsne_norm[test_label == 1][:,0], tsne_norm[test_label == 1][:,1], 'b.', label='maligancy')
    plt.legend()
    plt.title('add_0_test_t-sne')
    plt.savefig('data/feature/vis_feature/add_0_test_t-sne.png')
    plt.cla()

    
    
    return


#??????128*128?????????5????????????????????????
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
        
        print((fold+1), 'npy????????????npy_mask???????????????', list(set(npy_mask).difference(set(npy))))
    return

