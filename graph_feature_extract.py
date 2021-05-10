from model.graphNet import GAT,GCN
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np


def accuracy(pred, target):
    correct = pred.eq(target).double()
    correct = correct.sum()
    return correct / len(target)


# model = GCN(nfeat=512, 
#             nhid=64, 
#             nclass=2, 
#             fc_num=128,
#             dropout=0.6)
model = GAT(nfeat=512, 
            nhid=64, 
            nclass=2, 
            fc_num=128,
            dropout=0.6, 
            nheads=8, 
            alpha=0.2)
optimizer = optim.Adam(model.parameters(), 
                       lr=1e-4, 
                       weight_decay=5e-2)
train_len = 639
test_len = 160
feature_len = 512

googlenet_train_feature = torch.load('./data/feature/googlenet_train_feature.pt')
googlenet_test_feature = torch.load('./data/feature/googlenet_test_feature.pt')
resnet_train_feature = torch.load('./data/feature/resnet_train_feature.pt')
resnet_test_feature = torch.load('./data/feature/resnet_test_feature.pt')
vgg_train_feature = torch.load('./data/feature/vgg_train_feature.pt')
vgg_test_feature = torch.load('./data/feature/vgg_test_feature.pt')
hog_train_feature = torch.load('./data/feature/hog_train_feature.pt')
hog_test_feature = torch.load('./data/feature/hog_test_feature.pt')
lbp_train_feature = torch.load('./data/feature/lbp_train_feature.pt')
lbp_test_feature = torch.load('./data/feature/lbp_test_feature.pt')
glcm_train_feature = torch.load('./data/feature/glcm_train_feature.pt')
glcm_test_feature = torch.load('./data/feature/glcm_test_feature.pt')
train_label = torch.load('./data/feature/train_label.pt')
test_label = torch.load('./data/feature/test_label.pt')
adj = Variable(torch.ones((5, 5)))

for epoch in range(100):
    loss_train_list = []
    pre_train_list = torch.zeros(len(train_label))

    loss_test_list = []
    pre_test_list = torch.ones(len(test_label))
    #训练
    for index, one_nodule_feature in enumerate(zip(googlenet_train_feature, 
                                                    resnet_train_feature, 
                                                    vgg_train_feature, 
                                                    hog_train_feature,
                                                    lbp_train_feature)):
        temp = torch.zeros((len(one_nodule_feature),512))
        for i, feature in enumerate(one_nodule_feature):
            temp[i] = feature
        one_nodule_feature = temp
        
        one_nodule_feature = torch.from_numpy(np.array(one_nodule_feature))
        features = Variable(one_nodule_feature)

        model.train()
        optimizer.zero_grad()

        output = model(features, adj)
        one_label = train_label[index].unsqueeze(0).long()
        pre_train_list[index] = output.max(1)[1].type_as(one_label)
        loss_train = F.nll_loss(output,one_label)
        loss_train_list.append(loss_train.item())

        loss_train.backward()
        optimizer.step()
    acc_train = accuracy(pre_train_list, train_label)

    #测试
    for index, one_nodule_feature in enumerate(zip(googlenet_test_feature, 
                                                    resnet_test_feature, 
                                                    vgg_test_feature, 
                                                    hog_test_feature,
                                                    lbp_train_feature)):
        temp = torch.zeros((len(one_nodule_feature),512))
        for i, feature in enumerate(one_nodule_feature):
            temp[i] = feature
        one_nodule_feature = temp

        adj = torch.ones((len(one_nodule_feature), len(one_nodule_feature)))
        one_nodule_feature = torch.from_numpy(np.array(one_nodule_feature))
        features, adj = Variable(one_nodule_feature), Variable(adj)

        model.eval()
        output = model(features, adj)
        one_label = test_label[index].unsqueeze(0).long()
        pre_test_list[index] = output.max(1)[1].type_as(one_label)
        loss_test_list.append(F.nll_loss(output,one_label).item())
        
    acc_test = accuracy(pre_test_list, test_label)
    print('epoch:{:d}'.format(epoch) 
        , ', train loss:{:.4f}'.format(np.mean(loss_train_list)) 
        , ', train acc:{:.6f}'.format(acc_train.item()) 
        , ', test loss:{:.4f}'.format(np.mean(loss_test_list)) 
        , ', test acc:{:.6f}'.format(acc_test.item()))