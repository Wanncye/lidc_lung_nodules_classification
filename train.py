"""Train the model"""
# -*- coding: utf-8 -*-
import argparse
import logging
import os

from thop import profile

import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm
from torchsummary import summary
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import StepLR,MultiStepLR,ExponentialLR,CosineAnnealingLR
import torch.nn.functional as F

import utils
import model.net as net
import time
import torch.nn as nn
import model.data_loader as data_loader
import csv
from sklearn.metrics import confusion_matrix

from model.threeDresnet import generate_model
from model.threeDGoogleNet import googlenet
from model.threeDVGG import vgg16_bn, vgg11_bn, vgg13_bn, vgg19_bn
from model.threeDDensenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201
from model.threeDAlexnet import alexnet
from model.threeDLenet5 import lenet5
from model.Attention import attention56, attention92
from model.InceptionV3 import inceptionv3
from model.InceptionV4 import inceptionv4, inception_resnet_v2
from model.mobilenet import mobilenet
from model.mobilenetv2 import mobilenetv2
from model.preactresnet import preactresnet18,preactresnet34,preactresnet50,preactresnet101,preactresnet152
from model.resnext import resnext50,resnext101,resnext152
from model.ResnetInResnet import resnet_in_resnet
from model.senet import senet18,senet34,senet50,senet101,senet152
from model.shufflenet import shufflenet
from model.squeezenet import squeezenet
from model.wideresidual import wideresnet
from model.xception import xception

import netron     
import torch.onnx

'''
注意事项：
不加入其他特征时add_middle_feature=False,此时会保存模型提取出的特征
加入中间特征指的是GCN特征和传统特征
每次实验得修改descripe，用于生成不同的保存CSV预测结果的文件夹以及模型和准确率json文件
'''

#是否加入中间特征(包括GCN，传统，统计特征)
add_middle_feature = True
if add_middle_feature:
    #是否保存模型中间特征
    save_model_feature = False
else:
    save_model_feature = True
    save_model_dir = '10fold_model_feature_noNorm'

#设置生成的json文件、预测结果的描述，每次实验都不一样
# descripe = '_<=20mm_nodule_gcn_traditional_addEightLabelFeature_norInput_testZero_para1_10fold'
# descripe = '_para1_10fold_noNorm_add_gcn_traditional'
# descripe = '_para1_10fold_noNorm_add_gcn_adj_fc_5feature_512_graphSAGE_cat_traditional'
# descripe = 'para1_10fold_noNorm_only_add_gcn_adj_1-similarity_norm_5feature_512_cat'
# descripe = '_para1_10fold_noNorm_add_gcn_adj_1-similarity_norm_5feature_512_cat_traditional_BCELoss'
# descripe = '_para1_10fold_noNorm_add_gcn_adj_1-similarity_norm_4feature_512_cat_traditional' #对比试验4个特征FocalLoss

#GCN特征的文件加名
# gcn_feature_path = '10fold_gcn_feature_noNorm_adj_fc_addGoogleNet_grapgSAGE_mean' #用graphSAGE，mean聚合函数
# gcn_feature_path = '10fold_gcn_feature_noNorm_1-similarity_adj_diag_0_512_norm_addGoogleNet' #5个特征，最好模型
# gcn_feature_path = '10fold_gcn_feature_noNorm_1-similarity_adj_diag_0_512_norm_4feature' #4个特征FocalLoss

descripe = '_para1_10fold_noNorm_add_gcn_adj_1-similarity_norm_7feature_512_cat_traditional'
gcn_feature_path = '10fold_gcn_feature_noNorm_1-similarity_adj_diag_0_512_norm_7feature' #7个特征FocalLoss

#加特征之后全连接层的特征维度
feature_fusion_method = 'cat'
# feature_fusion_method = 'avg'
# feature_fusion_method = 'cat'
if feature_fusion_method == 'cat':
    fc_feature_dim = 512 + 512 + 38 + 255
    # fc_feature_dim = 512 + 38 + 512
elif feature_fusion_method == 'add' or feature_fusion_method == 'avg':
    fc_feature_dim = 512  + 38 + 255

#设置GPU的编号
torch.cuda.set_device(2)

#数据集文件夹名
# data_fold = '5fold_128<=20mm_aug'
data_fold = '10fold'

#要训练的模型
# model_list = ['alexnet','vgg13','resnet34','attention56','googlenet','shufflenet','mobilenet']
# model_list = ['alexnet','vgg13','resnet34','attention56','googlenet','shufflenet']
# model_list = ['alexnet','vgg13','resnet34','attention56','googlenet']
# model_list = ['alexnet','vgg13','resnet34','attention56']
# model_list = ['googlenet']
# model_list = ['resnet34','attention56','googlenet'] #VGG还没有训练完第8，9，10折
# model_list = ['attention56','googlenet']
# model_list = ['resnet34','attention56','googlenet','shufflenet']
model_list = ['shufflenet','mobilenet',]
# model_list = ['alexnet']
# model_list = ['googlenet']
# model_list = ['vgg13']
# model_list = ['resnet34']
# model_list = ['attention56']

#分两张卡训练，指定要训练的fold
foldList = [3,4,5,6,7,8,9]
# foldList = [0,1,2]

descripe = '_<=20mm_nodule_gcn_traditional'


def train(model, optimizer, loss_fn, dataloader, metrics, params, epoch, vis, N_folder, scheduler, model_name, lmbda):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
        ------- git graph test ------
    """

    # set model to training mode
    model.train()
    # print(model)

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    
    
    ground_truch_list = []
    predict_list = []
    # Use tqdm for progress bar
    datasetMean, datasetStd = utils.getDatasetMeanAndStd()
    datasetMean = datasetMean.T.expand(8,128).unsqueeze(-1).expand(8,128,128)
    datasetStd = datasetStd.T.expand(8,128).unsqueeze(-1).expand(8,128,128)
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch, file_name, one_feature) in enumerate(dataloader):

            # train_batch = (train_batch-datasetMean)/datasetStd
            
            egithFeature = utils.getEightLabelFeature(file_name)
            one_feature = torch.cat((one_feature, egithFeature), axis = 1)

            if params.cuda:
                train_batch, labels_batch, one_feature = train_batch.cuda(), labels_batch.cuda(), one_feature.cuda()
            #将载入的数据变成tensor
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            #将载入的数据输入3DResNet,得到结果
            output_batch, _ = model(train_batch, one_feature, add_middle_feature, feature_fusion_method)

            loss = loss_fn(output_batch, labels_batch)
            #将梯度初始化为0
            optimizer.zero_grad()
            #反向传播求权重梯度
            loss.backward()
            #更新所有参数
            optimizer.step()
            

            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()
            predict_batch = np.argmax(output_batch, axis=1)
            for i in range(len(labels_batch)):
                ground_truch_list.append(labels_batch[i])
                predict_list.append(predict_batch[i])

            # compute all metrics on this batch
            summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                for metric in metrics}
            summary_batch['loss'] = loss.item()
            summary_batch['epoch'] = epoch+1
            # summary_batch['iter'] = i
            summ.append(summary_batch)

            vis.plot(model_name + '_train_loss_folder_' + str(N_folder), summary_batch['loss'], 1)

            #每迭代一次，如果batchsize设置为16，则一次有16个图片输入网络以及输出
            loss_avg.update(loss.item())
            #set_postfix方法设置进度条显示信息，计算的loss均值为每一个batchsize的loss均值
            t.set_postfix(loss='{:5.3f}'.format(loss_avg()))
            #更新进度条
            t.update()

    class_1_number = np.sum(ground_truch_list)
    class_0_number = len(ground_truch_list) - class_1_number
    print("class 0 : {0}, class 1 : {1}".format(class_0_number,class_1_number))
    cMtric = confusion_matrix(ground_truch_list, predict_list)
    print(cMtric)
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    vis.plot(model_name + '_train_acc_folder_' + str(N_folder), metrics_mean['accuracy'] , 2)
    vis.plot(model_name + '_train_loss_epoch_folder_' + str(N_folder), metrics_mean['loss'] , 3)

def evaluate(model, loss_fn, dataloader, metrics, params,epoch, model_dir, vis, N_folder, model_name):

    ## 解决测试的时候显存不够的问题，加上下面这一句
    ground_truch_list = []
    predict_list = []
    with torch.no_grad():
        # set model to evaluation mode
        model.eval()

        # summary for current eval loop
        summ = []
        print("witer csv-------------------------")
        result_dir = os.path.join(model_dir, 'result'+descripe)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        predict_csv = open(result_dir+'/' + 'folder_' + str(N_folder) + '_result_'+str(epoch)+'.csv','w',encoding='utf-8')
        csv_writer = csv.writer(predict_csv)
        csv_writer.writerow(["filename","truth_label","predict_label","probability","is_right", "confidence"])
        # compute metrics over the dataset
        predict_prob = torch.zeros(len(dataloader.dataset))
        target = torch.zeros(len(dataloader.dataset))
        datasetMean, datasetStd = utils.getDatasetMeanAndStd()
        datasetMean = datasetMean.T.expand(8,128).unsqueeze(-1).expand(8,128,128)
        datasetStd = datasetStd.T.expand(8,128).unsqueeze(-1).expand(8,128,128)
        for dataloader_index, (data_batch, labels_batch, filename, one_feature) in enumerate(dataloader):
            # data_batch = (data_batch-datasetMean)/datasetStd

            egithFeature = utils.getEightLabelFeature(filename)
            egithFeature = torch.zeros_like(egithFeature)
            one_feature = torch.cat((one_feature, egithFeature), axis = 1)

            # move to GPU if available
            if params.cuda:
                data_batch, labels_batch, one_feature = data_batch.cuda(), labels_batch.cuda(), one_feature.cuda()
            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            
            # compute model output
            output_batch, _ = model(data_batch, one_feature, add_middle_feature, feature_fusion_method)
            loss = loss_fn(output_batch, labels_batch)

            m = nn.Softmax(dim=1)
            probability = m(output_batch)
            

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()

            predict = np.argmax(output_batch, axis=1)

            

            # 因为要得到指阳性标签的预测概率值，所以有下面一段代码
            st = dataloader_index * params.batch_size
            predict_prob[st:st + params.batch_size] = probability[:, 1]
            target[st:st + params.batch_size] = labels_batch

            labels_batch = labels_batch.data.cpu().numpy()
            for i in range(len(labels_batch)):
                ground_truch_list.append(labels_batch[i])
                predict_list.append(predict[i])


            for index,(name, truth_label, predict_label,probability) in enumerate(zip(filename,labels_batch,predict,probability)):
                # for index,(name, truth_label, predict_label,probability,mconfidence) in enumerate(zip(filename,labels_batch,predict,probability, confidence)):
                is_right = True if truth_label == predict_label else False
                # data = [name,truth_label,predict_label,probability[predict_label].item(),is_right, mconfidence.item()]
                data = [name,truth_label,predict_label,probability[predict_label].item(),is_right]
                csv_writer.writerow(data)
            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                            for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
            vis.plot(model_name + '_val_loss_folder_' + str(N_folder), summary_batch['loss'], 1)

        # compute mean of all metrics in summary
        class_1_number = np.sum(ground_truch_list)
        class_0_number = len(ground_truch_list) - class_1_number
        print("class 0 : {0}, class 1 : {1}".format(class_0_number,class_1_number))
        cMtric = confusion_matrix(ground_truch_list, predict_list)
        print(cMtric)
        print('confusion matrix acc: ' + str((cMtric[0,0]+cMtric[1,1])/(class_1_number+class_0_number)))
        for metric in summ[0]:
            print(metric)
        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)
        vis.plot(model_name + '_val_acc_folder_' + str(N_folder),metrics_mean['accuracy'] , 2)
        vis.plot(model_name + '_val_loss_epoch_folder_' + str(N_folder), metrics_mean['loss'] , 3)
        predict_csv.close()

        predict_prob = predict_prob.data.cpu().numpy()
        target = target.data.cpu().numpy()
    
    return metrics_mean, target, predict_prob

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir, N_folder, scheduler, model_name, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """

    # 可视化训练过程
    vis = utils.Visualizer(params.vis_env)

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir,  f'{restore_file}.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    is_best = 1
    val_acc_list = []
    params.num_epochs = 50
    for epoch in range(params.num_epochs):
        # Run one epoch
        print("模型：{0}，第{1}折".format(model_name,N_folder))
        print("第%d个epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']))
        vis.plot('lr',optimizer.param_groups[0]['lr'],2)
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        start_time = time.time()

        train(model, optimizer, loss_fn, train_dataloader, metrics, params, epoch, vis, N_folder, scheduler, model_name, lmbda = 0.1)
        scheduler.step()

        # Evaluate for one epoch on validation set
        val_metrics, truth_label, predict_label = evaluate(model, loss_fn, val_dataloader, metrics, params, epoch, model_dir, vis, N_folder, model_name)

        # 得到ROC指标
        prec, recall, auc = utils.get_metrics(truth_label, predict_label)
        specificity = utils.get_specificity(truth_label, predict_label)
        logging.info("- precision: " + str(prec) + " ; recall(or sensitivity): " + str(recall) + " ; auc: " + str(auc) + " ; specificity(or 1-FPR): " + str(specificity))
        vis.plot(model_name + '_val_sensitivity_epoch_folder_' + str(N_folder), recall , 3)
        vis.plot(model_name + '_val_specificity_epoch_folder_' + str(N_folder), specificity , 3)
        val_acc = val_metrics['accuracy']
        is_best = val_acc>best_val_acc

        # Save weights
        # if params.save_weight == 1:
        #     utils.save_checkpoint({'epoch': epoch + 1,
        #                         'state_dict': model.state_dict(),
        #                         'optim_dict' : optimizer.state_dict()},
        #                         is_best=is_best,
        #                         checkpoint=model_dir,
        #                         N_folder=N_folder,
        #                         params=params,
        #                         descript=descripe)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            # Save best val metrics in a json file in the model directory
            # best_json_path = os.path.join(model_dir, 'folder.'+ str(N_folder) + '.' +params.loss +'_alpha_'+str(params.FocalLossAlpha) + ".metrics_val_best_weights_"+descripe+".json")
            best_json_path = os.path.join(model_dir, 'folder.'+ str(N_folder) + '.' +params.loss +'_alpha_'+str(params.FocalLossAlpha) + descripe+".metrics_val_best_weights.json")
            val_metrics['epoch'] = epoch + 1
            utils.save_dict_to_json(val_metrics, best_json_path)

            #用最好的模型来提取512维特征
            dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = params.batch_size, data_dir="data/"+data_fold+"/fold"+str(N_folder+1), train_shuffle=False, fold= N_folder, gcn_feature_path = gcn_feature_path, add_middle_feature=add_middle_feature)
            train_dl_save = dataloaders['train']
            test_dl_save = dataloaders['test']
            if save_model_feature:
                print('------save feature-----')
                with torch.no_grad():
                    model.eval()
                    train_feature = torch.zeros((len(train_dl_save.dataset),512))
                    test_feature = torch.zeros((len(test_dl_save.dataset),512))
                    for i, (x, target, _, gcn_middle_feature) in enumerate(train_dl_save):
                        _, feature = model(x.cuda(), gcn_middle_feature.cuda(), add_middle_feature)
                        train_feature[(i*params.batch_size):((i+1)*params.batch_size), :] = feature.detach()
                    for i, (x, target, _, gcn_middle_feature) in enumerate(test_dl_save):
                        _, feature = model(x.cuda(), gcn_middle_feature.cuda(), add_middle_feature)
                        test_feature[(i*params.batch_size):((i+1)*params.batch_size), :] = feature.detach()
                    torch.save(train_feature,'./data/feature/'+save_model_dir+'/fold_' + str(N_folder) + '_' + model_name + '_train.pt')
                    torch.save(test_feature,'./data/feature/'+save_model_dir+'/fold_' + str(N_folder) + '_' + model_name + '_test.pt')
        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, 'folder.'+ str(N_folder) + '.' +params.loss + '_alpha_'+str(params.FocalLossAlpha) + descripe +".metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # 计算剩余时间
        finish_time = time.time()
        used_time = finish_time-start_time
        print('used: ' + str(used_time) + ' seconds. ')
        eta = ((params.num_epochs - 1 - epoch) + (9 - N_folder) * params.num_epochs) * used_time / 60
        print(model_name + ' eta: ' + str(eta) + ' minutes. = ' + str(eta/60) + 'hs')
        print("\n")

        val_acc_list.append(val_acc)

if __name__ == '__main__':
    total_time_start = time.time()
    for model_name in model_list:
        print(model_name)
        print('\n')
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_dir', default='experiments/' + model_name + '_nomask', help="Directory containing params.json")

        all_time_start = time.time()
        # Load the parameters from json file
        args = parser.parse_args()
        json_path = os.path.join(args.model_dir, 'params.json')
        print('params file path: '+ json_path)
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        params = utils.Params(json_path)

        # use GPU if available
        params.cuda = torch.cuda.is_available()

        torch.cuda.empty_cache()


        # 设置随机种子，种子相同，随机初始化相同，两次跑的结果相同
        init_seed = 230
        np.random.seed(init_seed)
        torch.manual_seed(init_seed) # cpu
        torch.cuda.manual_seed_all(init_seed)  # gpu
        torch.cuda.manual_seed(init_seed)  # gpu
        torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
        # torch.backends.cudnn.benchmark = False

        # 设置logger
        print('train file path:',os.path.join(args.model_dir, 'train_'+params.loss+'_alpha_'+str(params.FocalLossAlpha)+'_correct-alpha.log'))
        utils.set_logger(os.path.join(args.model_dir, 'train_'+params.loss+'_alpha_'+str(params.FocalLossAlpha)+descripe+'_correct-alpha.log'))


        for N_folder in foldList:
            print(N_folder)
            logging.info("------------------folder " + str(N_folder) + "------------------")
            logging.info("Loading the datasets...")

            #5折交叉验证
            dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = params.batch_size, data_dir="data/"+data_fold+"/fold"+str(N_folder+1), train_shuffle=True, gcn_feature_path = gcn_feature_path, fold= N_folder, add_middle_feature=add_middle_feature)
            # dataloaders = data_loader.fetch_N_folders_dataloader(test_folder=N_folder, types = ["train", "test"], batch_size = params.batch_size, data_dir=params.data_dir)
            train_dl = dataloaders['train']
            test_dl = dataloaders['test']
            logging.info("- done.")
            if model_name == 'resnet10':
                model = generate_model(params.net_depth).cuda()
                print('Using ResNet'+str(params.net_depth))
            elif model_name == 'resnet18':
                model = generate_model(params.net_depth).cuda()
                print('Using ResNet'+str(params.net_depth))
            elif model_name == 'resnet34':
                model = generate_model(params.net_depth, fc_feature_dim=fc_feature_dim).cuda()
                print('Using ResNet'+str(params.net_depth))
            elif model_name == 'resnet50':
                model = generate_model(params.net_depth).cuda()
                print('Using ResNet'+str(params.net_depth))
            elif model_name == 'resnet100':
                model = generate_model(params.net_depth).cuda()
                print('Using ResNet'+str(params.net_depth))
            elif model_name == 'resnet152':
                model = generate_model(params.net_depth).cuda()
                print('Using ResNet'+str(params.net_depth))
            elif model_name == 'resnet200':
                model = generate_model(params.net_depth).cuda()
                print('Using ResNet'+str(params.net_depth))
            elif model_name == 'googlenet':
                model = googlenet(fc_feature_dim=fc_feature_dim).cuda()
                print('Using GoogLeNet')
            elif model_name == 'vgg11':
                model = vgg11_bn(params.dropout_rate).cuda()
                print('Using VGG_11')
            elif model_name == 'vgg13':
                model = vgg13_bn(params.dropout_rate,fc_feature_dim).cuda()
                print('Using VGG_13')
            elif model_name == 'vgg16':
                model = vgg16_bn(params.dropout_rate).cuda()
                print('Using VGG_16')
            elif model_name == 'vgg19':
                model = vgg19_bn(params.dropout_rate).cuda()
                print('Using VGG_19')
            elif model_name == 'densenet121':
                model = DenseNet121().cuda()
                print('Using densenet121')
            elif model_name == 'densenet161':
                model = DenseNet161().cuda()
                print('Using densenet161')
            elif model_name == 'densenet169':
                model = DenseNet169().cuda()
                print('Using densenet169')
            elif model_name == 'densenet201':
                model = DenseNet201().cuda()
                print('Using densenet201')
            elif model_name == 'alexnet':
                model = alexnet(fc_feature_dim=fc_feature_dim).cuda()
                print('Using alexnet')
            elif model_name == 'lenet5':
                model = lenet5().cuda()
                print('Using lenet5')
            elif model_name == 'attention56':
                model = attention56(fc_feature_dim=fc_feature_dim).cuda()
                print('Using attention56')
            elif model_name == 'attention92':
                model = attention92().cuda()
                print('Using attention92')
            elif model_name == 'inceptionv3':
                model = inceptionv3().cuda()
                print('Using inceptionv3')
            elif model_name == 'inceptionv4':
                model = inceptionv4().cuda()
                print('Using inceptionv4')
            elif model_name == 'inception_resnet_v2':
                model = inception_resnet_v2().cuda()
                print('Using inception_resnet_v2')
            elif model_name == 'mobilenet':
                model = mobilenet(fc_feature_dim=fc_feature_dim).cuda()
                print('Using mobilenet')
            elif model_name == 'mobilenetv2':
                model = mobilenetv2().cuda()
                print('Using mobilenetv2')
            elif model_name == 'preactresnet18':
                model = preactresnet18().cuda()
                print('Using preactresnet18')
            elif model_name == 'preactresnet34':
                model = preactresnet34().cuda()
                print('Using preactresnet34')
            elif model_name == 'preactresnet50':
                model = preactresnet50().cuda()
                print('Using preactresnet50')
            elif model_name == 'preactresnet101':
                model = preactresnet101().cuda()
                print('Using preactresnet101')
            elif model_name == 'preactresnet152':
                model = preactresnet152().cuda()
                print('Using preactresnet152')
            elif model_name == 'resnext50':
                model = resnext50().cuda()
                print('Using resnext50')
            elif model_name == 'resnext101':
                model = resnext101().cuda()
                print('Using resnext101')
            elif model_name == 'resnext152':
                model = resnext152().cuda()
                print('Using resnext152')
            elif model_name == 'resnet_in_resnet':
                model = resnet_in_resnet().cuda()
                print('Using resnet_in_resnet')
            elif model_name == 'senet18':
                model = senet18().cuda()
                print('Using senet18')
            elif model_name == 'senet34':
                model = senet34().cuda()
                print('Using senet34')
            elif model_name == 'senet50':
                model = senet50().cuda()
                print('Using senet50')
            elif model_name == 'senet101':
                model = senet101().cuda()
                print('Using senet101')
            elif model_name == 'senet152':
                model = senet152().cuda()
                print('Using senet152')
            elif model_name == 'shufflenet':
                model = shufflenet(fc_feature_dim=fc_feature_dim).cuda()
                print('Using shufflenet')
            elif model_name == 'squeezenet':
                model = squeezenet().cuda()
                print('Using squeezenet')
            elif model_name == 'wideresidual':
                model = wideresnet().cuda()
                print('Using wideresnet')
            elif model_name == 'xception':
                model = xception().cuda()
                print('Using xception')

            # print('# model parameters:', sum(param.numel() for param in model.parameters()))
            # input = torch.randn(1, 1, 8, 128, 128).cuda()
            # fake_feature = torch.randn(1,56*4).cuda()
            # fake_add_gcn_feature = False
            # flops_num, params_num = profile(model, inputs=(input, fake_feature, fake_add_gcn_feature))
            # print('# flops:', flops_num)
            # print('# params:', params_num)

            # 在pytorch中，输入数据的维数可以表示为（N,C,D,H,W），其中：N为batch_size，C为输入的通道数，D为深度（D这个维度上含有时序信息），H和W分别是输入图像的高和宽。
            #可视化网络结构
            # vis_x = Variable(torch.FloatTensor(16,1, 8, 128, 128)).cuda()
            # vis_y = model(vis_x)
            # onnx_path = "onnx_model_name.onnx"
            # torch.onnx.export(model, vis_x, onnx_path)
            # netron.start(onnx_path)
            if model_name in ['shufflenet','mobilenet']:
                weight_decay = 0.01
            else:
                weight_decay = 0.0001
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=weight_decay)
            scheduler = MultiStepLR(optimizer, milestones=[20,50,80], gamma=0.5)
            # scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.8)
            # scheduler = ExponentialLR(optimizer, gamma=0.90)
            # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

            # fetch loss function and metrics
            print(params.loss)
            if params.loss == 'BCE':
                loss_fn = net.loss_fn_BCE   #交叉熵损失
            elif params.loss == 'FocalLoss':
                loss_fn = net.FocalLoss(alpha=params.FocalLossAlpha,gamma=params.FocalLossGamma)       #focalLoss损失
            else:
                print("- No this type of loss!")
            loss_fn = net.loss_fn_BCE
            metrics = net.metrics

            # Train the model
            logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
            for split, dl in dataloaders.items():
                logging.info("Number of %s examples: %s" % (split, str(len(dl.dataset))))
            
            # train(model, optimizer, loss_fn, train_dl, metrics, params)
            print('lr:',params.learning_rate)
            train_and_evaluate(model, train_dl, test_dl, optimizer, loss_fn, metrics, params, args.model_dir, N_folder, scheduler, model_name)
            # train_and_evaluate(model, train_dl, test_dl, optimizer, loss_fn, metrics, params, args.model_dir, N_folder, scheduler, model_name, restore_file="folder.0.FocalLoss_alpha_0.25.best")
        all_time_finish = time.time()
        all_used_time = all_time_finish - all_time_start
        print(model_name + ' used: ' + str(all_used_time/60) + ' mins.  =' + str(all_used_time/3600) + 'hs')
    total_time_finish = time.time()
    total_used_time = total_time_finish-total_time_start
    print('total used: ' + str(all_used_time/60) + ' mins.  =' + str(all_used_time/3600) + 'hs')
