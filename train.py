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
from torch.optim.lr_scheduler import StepLR,MultiStepLR

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

#是否保存模型中间特征
save_model_feature = True
#是否加入中间特征(包括GCN，传统，统计特征)
add_middle_feature = False

def train(model, optimizer, loss_fn, dataloader, metrics, params, epoch, vis, N_folder, scheduler, model_name):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
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
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch, file_name, one_feature) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch, one_feature = train_batch.cuda(), labels_batch.cuda(), one_feature.cuda()
            #将载入的数据变成tensor
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            #将载入的数据输入3DResNet,得到结果
            output_batch, _ = model(train_batch, one_feature, add_middle_feature)
            #计算网络输出结果和目标值之间的损失

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
        result_dir = os.path.join(model_dir, 'result')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        predict_csv = open(result_dir+'/' + 'folder_' + str(N_folder) + '_result_'+str(epoch)+'.csv','w',encoding='utf-8')
        csv_writer = csv.writer(predict_csv)
        csv_writer.writerow(["filename","truth_label","predict_label","probability","is_right"])
        # compute metrics over the dataset
        predict_prob = torch.zeros(len(dataloader.dataset))
        target = torch.zeros(len(dataloader.dataset))
        for dataloader_index, (data_batch, labels_batch, filename, one_feature) in enumerate(dataloader):

            # move to GPU if available
            if params.cuda:
                data_batch, labels_batch, one_feature = data_batch.cuda(), labels_batch.cuda(), one_feature.cuda()
            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
            
            # compute model output
            output_batch, _ = model(data_batch, one_feature, add_middle_feature)
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
                is_right = True if truth_label == predict_label else False
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
    for epoch in range(params.num_epochs):
        # Run one epoch
        print("第%d个epoch的学习率：%f" % (epoch+1, optimizer.param_groups[0]['lr']))
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        start_time = time.time()

        train(model, optimizer, loss_fn, train_dataloader, metrics, params, epoch, vis, N_folder, scheduler, model_name)
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
        is_best = val_acc>=best_val_acc

        # Save weights
        if params.save_weight == 1:
            utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optim_dict' : optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=model_dir,
                                N_folder=N_folder,
                                params=params)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, 'folder.'+ str(N_folder) + '.' +params.loss +'_alpha_'+str(params.FocalLossAlpha) + ".metrics_val_best_weights.json")
            val_metrics['epoch'] = epoch + 1
            utils.save_dict_to_json(val_metrics, best_json_path)

            #用最好的模型来提取512维特征
            dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = params.batch_size, data_dir="data/5fold_128<=20mm_aug/fold"+str(N_folder+1), train_shuffle=False, fold= N_folder, add_middle_feature=add_middle_feature)
            train_dl_save = dataloaders['train']
            test_dl_save = dataloaders['test']
            if save_model_feature:
                with torch.no_grad():
                    model.eval()
                    train_feature = torch.zeros((len(train_dl_save.dataset),512))
                    test_feature = torch.zeros((len(test_dl_save.dataset),512))
                    for i, (x, target, _, gcn_middle_feature) in enumerate(train_dl_save):
                        _, feature = model(x.cuda(), gcn_middle_feature, add_middle_feature)
                        train_feature[(i*params.batch_size):((i+1)*params.batch_size), :] = feature.detach()
                    for i, (x, target, _, gcn_middle_feature) in enumerate(test_dl_save):
                        _, feature = model(x.cuda(), gcn_middle_feature, add_middle_feature)
                        test_feature[(i*params.batch_size):((i+1)*params.batch_size), :] = feature.detach()
                    torch.save(train_feature,'./data/feature/5fold_128<=20mm_aug/fold_' + str(N_folder) + '_' + model_name + '_train.pt')
                    torch.save(test_feature,'./data/feature/5fold_128<=20mm_aug/fold_' + str(N_folder) + '_' + model_name + '_test.pt')
        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, 'folder.'+ str(N_folder) + '.' +params.loss + '_alpha_'+str(params.FocalLossAlpha) + ".metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # 计算剩余时间
        finish_time = time.time()
        used_time = finish_time-start_time
        print('used: ' + str(used_time) + ' seconds. ')
        eta = ((params.num_epochs - 1 - epoch) + (0 - N_folder) * params.num_epochs) * used_time / 60
        print('eta: ' + str(eta) + ' minutes. = ' + str(eta/60) + 'hs')
        print("\n")

        val_acc_list.append(val_acc)
        # if epoch > 10:
        #     if val_acc_list[epoch] == val_acc_list[epoch-1] and \
        #         val_acc_list[epoch] == val_acc_list[epoch-2] and \
        #         val_acc_list[epoch] == val_acc_list[epoch-3] and \
        #         val_acc_list[epoch] == val_acc_list[epoch-4] and \
        #         val_acc_list[epoch] == val_acc_list[epoch-5] and\
        #         val_acc_list[epoch] == val_acc_list[epoch-6] and\
        #         val_acc_list[epoch] == val_acc_list[epoch-7]:
        #         logging.info("- early stop because 5 epochs had the same accuracy.")
        #         break

if __name__ == '__main__':

    # model_list=['vgg11',  'vgg13', 'vgg16', 'vgg19', 
    #             'googlenet', 
    #             'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200',
    #             'alexnet']

    # model_list = ['attention56', 'attention92', 'mobilenet', 'mobilenetv2', 'shufflenet', 'squeezenet', 'preactresnet18', 'preactresnet34', 'preactresnet50', 'preactresnet101', 'preactresnet152',]
    # model_list = [  'alexnet', 'vgg13','resnet34', 'densenet201', ]
    # model_list = ['densenet201']
    # model_list = ['resnet34']
    # model_list=['lenet5']                         #有问题 50%
    # model_list=['alexnet']                        #86.88%
    model_list=['attention56']                    #83.75%
    # model_list=['attention92']                    #81.25%
    # model_list=['resnet50']                    

    # model_list=['mobilenet']                      #69.75%
    # model_list=['mobilenetv2']                    #77.63%

    # model_list=['shufflenet']                     #76.25%
    # model_list=['squeezenet']                     #88.75%

    # model_list=['preactresnet18']                 #84.38%
    # model_list=['preactresnet34']                 #81.88%
    # model_list=['preactresnet50']                 #73.13%       
    # model_list=['preactresnet101']                #72.50%        
    # model_list=['preactresnet152']                #81.88%  

    # model_list=['densenet161']                    #85.63%  
    # model_list=['densenet201']                    #84.38%
    # model_list=['densenet169']                    #85.00%
    # model_list=['densenet121']                    #82.50%

    # model_list=['inceptionv3']                    #78.13%     
    # model_list=['inceptionv4']                    #有问题 50%
    # model_list=['inception_resnet_v2']            #有问题 50%
    
    # model_list = ['resnet_in_resnet',
    # 'senet18', 'senet34', 'senet50', 'senet101', 'senet152', 'xception', 'wideresidual','inceptionv3']

    # model_list=['resnext50']                      #83.13%  
    # model_list=['resnext101']                     #72.50%
    # model_list=['resnext152']                     #74.38%
    # model_list=['resnet_in_resnet']               #81.88%
    # model_list=['senet18']                        #85.00%
    # model_list=['senet34']                        #86.25%
    # model_list=['senet50']                        #78.75%
    # model_list=['senet101']                       #80.63%
    # model_list=['senet152']                       #81.88%
    # model_list=['wideresidual']                   
    # model_list=['xception']                       #85.63%
    # model_list=['xception', 'wideresidual']
        
    for model_name in model_list:

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

        #使用第二块gpu
        torch.cuda.set_device(0)
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
        utils.set_logger(os.path.join(args.model_dir, 'train_'+params.loss+'_alpha_'+str(params.FocalLossAlpha)+'_correct-alpha.log'))

        # 五折交叉验证
        for N_folder in range(2,5):
            print(N_folder)
            logging.info("------------------folder " + str(N_folder) + "------------------")
            logging.info("Loading the datasets...")
            # 得到训练测试数据
            # dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = params.batch_size, data_dir="data/nodules3d_128_npy_no_same_patient_in_two_dataset", train_shuffle=False)
            
            #5折交叉验证
            dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = params.batch_size, data_dir="data/5fold_128<=20mm_aug/fold"+str(N_folder+1), train_shuffle=True, fold= N_folder, add_middle_feature=add_middle_feature)
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
                model = generate_model(params.net_depth).cuda()
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
                model = googlenet().cuda()
                print('Using GoogLeNet')
            elif model_name == 'vgg11':
                model = vgg11_bn(params.dropout_rate).cuda()
                print('Using VGG_11')
            elif model_name == 'vgg13':
                model = vgg13_bn(params.dropout_rate).cuda()
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
                model = alexnet().cuda()
                print('Using alexnet')
            elif model_name == 'lenet5':
                model = lenet5().cuda()
                print('Using lenet5')
            elif model_name == 'attention56':
                model = attention56().cuda()
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
                model = mobilenet().cuda()
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
                model = shufflenet().cuda()
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

            print('# model parameters:', sum(param.numel() for param in model.parameters()))
            input = torch.randn(1, 1, 8, 128, 128).cuda()
            fake_feature = torch.randn(1,56*4).cuda()
            fake_add_gcn_feature = False
            flops_num, params_num = profile(model, inputs=(input, fake_feature, fake_add_gcn_feature))
            print('# flops:', flops_num)
            print('# params:', params_num)

            # 在pytorch中，输入数据的维数可以表示为（N,C,D,H,W），其中：N为batch_size，C为输入的通道数，D为深度（D这个维度上含有时序信息），H和W分别是输入图像的高和宽。
            #可视化网络结构
            # vis_x = Variable(torch.FloatTensor(16,1, 8, 128, 128)).cuda()
            # vis_y = model(vis_x)
            # onnx_path = "onnx_model_name.onnx"
            # torch.onnx.export(model, vis_x, onnx_path)
            # netron.start(onnx_path)
            
            optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=0.0001)
            scheduler = MultiStepLR(optimizer, milestones=[20,50,80], gamma=0.5)

            # fetch loss function and metrics
            print(params.loss)
            if params.loss == 'BCE':
                loss_fn = net.loss_fn_BCE   #交叉熵损失
            elif params.loss == 'FocalLoss':
                loss_fn = net.FocalLoss(alpha=params.FocalLossAlpha,gamma=params.FocalLossGamma)       #focalLoss损失
            else:
                print("- No this type of loss!")
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
        print('used: ' + str(all_used_time/60) + ' mins.  =' + str(all_used_time/3600) + 'hs')

