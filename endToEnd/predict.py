import torch.nn as nn
import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from graphNet import GAT,GCN
from alexnet import alexnet
from resnet import generate_model
from vgg import vgg13_bn
from densenet import DenseNet201
from attention import attention56

from data_loader import fetch_dataloader
import numpy as np
import time

from utils import *
from net import loss_fn_BCE, accuracy

from sklearn.metrics import confusion_matrix

class endToend(nn.Module):
    '''
    使用resnet、vgg、alexnet、densenet做成一个端到端的训练模型
    '''
    def __init__(self, resnet, vgg, alexnet, densenet, gcn, adj=None):
        super(endToend, self).__init__()
        self.resnet = resnet
        self.vgg = vgg
        self.alexnet = alexnet
        self.densenet = densenet
        if adj == None:
            adj = torch.ones((4,4))
            np.random.seed(123)
            for i in range(4):
                for j in range(4):
                    random_num = np.random.rand()
                    if random_num > 0.5:
                        adj[i,j] = 0
            print('adj: ', adj)
            self.adj = adj.cuda()
            
        else:
            self.adj = adj.cuda()

        self.gcn = gcn
        self.fc1 = nn.Linear(512*4, 56*4).cuda()
        self.relu = nn.ReLU(inplace=True).cuda()
        self.dropout = nn.Dropout(0.5).cuda()
        self.fc2 = nn.Linear(56*4*2, 2).cuda()

    def forward(self, input):
        alexnetOutput, alexnetFeature = self.alexnet(input)
        vggOutput, vggFeature = self.vgg(input)
        resnetOutput, resnetFeature = self.resnet(input)
        densenetOutput, densenetFeature = self.densenet(input)
        cnnFeature = torch.cat(
            (alexnetFeature.unsqueeze(1),
            vggFeature.unsqueeze(1),
            resnetFeature.unsqueeze(1),
            densenetFeature.unsqueeze(1)),
            axis = 1
        )

        cnnFcOutput = self.fc1(cnnFeature.view(cnnFeature.size(0),-1))
        cnnFcOutput = self.relu(cnnFcOutput)
        cnnFcOutput = self.dropout(cnnFcOutput)
        
        gcnFeature, gcnOutput = self.gcn(cnnFeature, self.adj)

        totalFeature = torch.cat(
            (cnnFcOutput,gcnFeature),
            axis = 1
        )

        output = self.fc2(totalFeature)
        
        return [alexnetOutput, vggOutput, resnetOutput, densenetOutput], gcnOutput, output


if __name__ == '__main__':
    
    foldList = [5]
    for fold in foldList:

        init_seed = 230
        np.random.seed(init_seed)
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed_all(init_seed)
        torch.cuda.manual_seed(init_seed) 
        torch.backends.cudnn.deterministic = True

        resnet = generate_model(34).cuda()
        vgg = vgg13_bn(0.5).cuda()
        alexnet = alexnet().cuda()
        densenet = attention56().cuda()
        gcn = GCN(nfeat=512,
            nhid=64,
            nclass=2,
            fc_num=2,
            dropout=0.6,
            ft=4).cuda()

        endToEndModel = endToend(resnet, vgg, alexnet, densenet, gcn)
        optimizer = optim.Adam(endToEndModel.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = MultiStepLR(optimizer, milestones=[10,20,40], gamma=0.5)
        loadCheckpoint = True

        if loadCheckpoint:
            print("---加载模型---")
            model_CKPT = torch.load('checkpoint/finalEpochWeight_firstweight_fold4.pth')
            endToEndModel.load_state_dict(model_CKPT['state_dict'])
            print("---epoch:{0}".format(model_CKPT['epoch']))
            optimizer.load_state_dict(model_CKPT['optim_dict'])
            print("该模型的学习率为{0}".format(optimizer.param_groups[0]['lr']))
        print("数据集路径：{0}".format("../data/5fold_128<=20mm_aug/fold"+str(fold)))
        dataloaders = fetch_dataloader(types = ["train", "test"], batch_size = 3, data_dir="../data/5fold_128<=20mm_aug/fold"+str(fold), train_shuffle=True)
        train_dl = dataloaders['train']
        test_dl = dataloaders['test']


        with torch.no_grad():
            endToEndModel.eval()
            summ = []
            loss_epoch = []
            ground_truch = []
            predict = []
            with tqdm(total=len(test_dl)) as t:
                for dataloader_index, (test_batch, labels_batch, filename) in enumerate(test_dl):
                    test_batch, labels_batch = test_batch.cuda(), labels_batch.cuda()
                    test_batch, labels_batch = Variable(test_batch), Variable(labels_batch)
                    modelOutput, gcnOutput, finalOutput = endToEndModel(test_batch)

                    finalLoss = loss_fn_BCE(finalOutput, labels_batch)
                    totalLoss =  finalLoss

                    finalOutput = finalOutput.data.cpu().numpy()
                    labels_batch = labels_batch.data.cpu().numpy()

                    predict_batch = np.argmax(finalOutput, axis=1)

                    for i in range(len(labels_batch)):
                        ground_truch.append(labels_batch[i])
                        predict.append(predict_batch[i])

                    loss_epoch.append(totalLoss.item())

                    summary_batch = {
                        'accuracy': None,
                        'loss': None
                    }
                    summary_batch['accuracy'] = accuracy(finalOutput, labels_batch)
                    summary_batch['loss'] = totalLoss.item()
                    summ.append(summary_batch)

                    t.set_postfix(loss='{:5.3f}'.format(totalLoss.item()))
                    t.update()

            class_1_number = np.sum(ground_truch)
            class_0_number = len(ground_truch) - class_1_number
            print("class 0 : {0}, class 1 : {1}".format(class_0_number,class_1_number))
            cMtric = confusion_matrix(ground_truch, predict)
            print(cMtric)
            TN = cMtric[0][0]
            FP = cMtric[0][1]
            FN = cMtric[1][0]
            TP = cMtric[1][1]

            metrics_mean = {
                'accuracy' : (TN+TP)/(TP+TN+FP+FN),
                'loss' : np.mean(loss_epoch),
                'sensitivity' : TP/(TP+FN),
                'specitivity' : TN/(TN+FP)
            }
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
            print("- Eval metrics : " + metrics_string)
            print('\n\n')
