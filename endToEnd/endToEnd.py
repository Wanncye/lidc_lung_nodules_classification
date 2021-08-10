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

from data_loader import fetch_dataloader
import numpy as np
import time

from utils import *
from net import loss_fn_BCE, metrics

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
    resnet = generate_model(34).cuda()
    vgg = vgg13_bn(0.5).cuda()
    alexnet = alexnet().cuda()
    densenet = DenseNet201().cuda()
    gcn = GCN(nfeat=512,
        nhid=64,
        nclass=2,
        fc_num=2,
        dropout=0.6,
        ft=4).cuda()

    endToEndModel = endToend(resnet, vgg, alexnet, densenet, gcn)
    optimizer = optim.Adam(endToEndModel.parameters(), lr=0.0001, weight_decay=0)
    scheduler = MultiStepLR(optimizer, milestones=[20,50,80], gamma=0.5)

    
    dataloaders = fetch_dataloader(types = ["train", "test"], batch_size = 3, data_dir="../data/5fold_128/fold1", train_shuffle=True, fold= 1)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    vis = Visualizer('endToEndVis')

    for epoch in range(100):
        
        endToEndModel.train()
        summ = []
        with tqdm(total=len(train_dl)) as t:
            for i, (train_batch, labels_batch, file_name) in enumerate(train_dl):
                train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda() 
                
                train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
                modelOutput, gcnOutput, finalOutput = endToEndModel(train_batch)
                alexnetLoss = loss_fn_BCE(modelOutput[0], labels_batch)

                vggLoss = loss_fn_BCE(modelOutput[1], labels_batch)

                resnetLoss = loss_fn_BCE(modelOutput[2], labels_batch)

                densenetLoss = loss_fn_BCE(modelOutput[3], labels_batch)


                gcnLoss = loss_fn_BCE(gcnOutput, labels_batch)

                finalLoss = loss_fn_BCE(finalOutput, labels_batch)

                totalLoss = (alexnetLoss + vggLoss + resnetLoss + densenetLoss) + gcnLoss + finalLoss

                optimizer.zero_grad()
                totalLoss.backward()
                optimizer.step()

                finalOutput = finalOutput.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()


                summary_batch = {metric:metrics[metric](finalOutput, labels_batch)
                                    for metric in metrics}

                summary_batch['loss'] = totalLoss.item()
                summary_batch['epoch'] = epoch+1
                summ.append(summary_batch)

                vis.plot('train_loss_iter', summary_batch['loss'], 1)

                
                t.set_postfix(loss='{:5.3f}'.format(totalLoss.item()))
                t.update()

        metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        print("- Train metrics: " + metrics_string)
        vis.plot('train_acc', metrics_mean['accuracy'] , 2)
        vis.plot('train_loss_epoch', metrics_mean['loss'] , 3)


        with torch.no_grad():
            endToEndModel.eval()
            summ = []
            with tqdm(total=len(test_dl)) as t:
                for dataloader_index, (test_batch, labels_batch, filename) in enumerate(test_dl):
                    test_batch, labels_batch = test_batch.cuda(), labels_batch.cuda()
                    test_batch, labels_batch = Variable(test_batch), Variable(labels_batch)
                    modelOutput, gcnOutput, finalOutput = endToEndModel(test_batch)

                    alexnetLoss = loss_fn_BCE(modelOutput[0], labels_batch)
                    vggLoss = loss_fn_BCE(modelOutput[1], labels_batch)
                    resnetLoss = loss_fn_BCE(modelOutput[2], labels_batch)
                    densenetLoss = loss_fn_BCE(modelOutput[3], labels_batch)
                    gcnLoss = loss_fn_BCE(gcnOutput, labels_batch)
                    finalLoss = loss_fn_BCE(finalOutput, labels_batch)
                    totalLoss = (alexnetLoss + vggLoss + resnetLoss + densenetLoss) + gcnLoss + finalLoss

                    finalOutput = finalOutput.data.cpu().numpy()
                    labels_batch = labels_batch.data.cpu().numpy()

                    summary_batch = {metric: metrics[metric](finalOutput, labels_batch)
                                for metric in metrics}

                    summary_batch['loss'] = totalLoss.item()
                    summ.append(summary_batch)
                    vis.plot('val_loss_iter', summary_batch['loss'], 1)

                    t.set_postfix(loss='{:5.3f}'.format(totalLoss.item()))
                    t.update()

            metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
            print("- Eval metrics : " + metrics_string)
            vis.plot('val_acc' ,metrics_mean['accuracy'] , 2)
            vis.plot('val_loss_epoch', metrics_mean['loss'], 3)







        
