from model.graphNet import GAT,GCN
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def accuracy(pred, target):
    correct = pred.eq(target).double()
    correct = correct.sum()
    con_matrix = confusion_matrix(target,pred,labels=range(2))
    TN = con_matrix[0][0]
    TP = con_matrix[1][1]
    FN = con_matrix[1][0]
    FP = con_matrix[0][1]
    return correct / len(target),[TN,TP,FN,FP]


model = GCN(nfeat=512,
            nhid=64,
            nclass=2,
            fc_num=2,
            dropout=1)
# model = GAT(nfeat=512,
#             nhid=64,
#             nclass=2,
#             fc_num=128,
#             dropout=0.6,
#             nheads=8,
#             alpha=0.2)
optimizer = optim.Adam(model.parameters(), 
                       lr=1e-4, 
                       weight_decay=5e-2)
train_len = 639
test_len = 160
feature_len = 512

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

adj = Variable(torch.ones((4, 4)))

best_test_acc = 0
best_epoc = 0
for epoch in range(100):
    loss_train_list = []
    pre_train_list = torch.zeros(len(train_label))

    loss_test_list = []
    pre_test_list = torch.ones(len(test_label))
    #训练
    for index, one_nodule_feature in enumerate(zip(googlenet_train_feature, 
                                                    resnet_train_feature, 
                                                    vgg_train_feature,
                                                    hog_train_feature)):
        temp = torch.zeros((len(one_nodule_feature),512))
        for i, feature in enumerate(one_nodule_feature):
            temp[i] = feature
        one_nodule_feature = temp
        
        one_nodule_feature = torch.from_numpy(np.array(one_nodule_feature))
        features = Variable(one_nodule_feature)

        model.train()
        optimizer.zero_grad()

        _ , _, output = model(features, adj)
        one_label = train_label[index].unsqueeze(0).long()
        pre_train_list[index] = output.max(1)[1].type_as(one_label)
        loss_train = F.nll_loss(output,one_label)
        loss_train_list.append(loss_train.item())

        loss_train.backward()
        optimizer.step()
    acc_train,_ = accuracy(pre_train_list, train_label)

    #测试
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
        _ , _, output = model(features, adj)
        one_label = test_label[index].unsqueeze(0).long()
        pre_test_list[index] = output.max(1)[1].type_as(one_label)
        loss_test_list.append(F.nll_loss(output,one_label).item())
        
    acc_test,conf_mat = accuracy(pre_test_list, test_label)
    if acc_test >= best_test_acc:
        best_test_acc = acc_test
        best_epoc = epoch
        #最好准确率时保存模型
        # torch.save({
        #     'epoch' : epoch,
        #     'state_dict': model.state_dict(),
        #     'optim_dict' : optimizer.state_dict()
        # },'./experiments/gcn/fc_2_feature_4_wdecay_5e-2.best.pth.tar')
    print('epoch:{:d}'.format(epoch) 
        , ', train loss:{:.4f}'.format(np.mean(loss_train_list)) 
        , ', train acc:{:.6f}'.format(acc_train.item()) 
        , ', test loss:{:.4f}'.format(np.mean(loss_test_list)) 
        , ', test acc:{:.6f}'.format(acc_test.item()))
print('best test acc:{:.4f}, epoch:{:d}, TN:{:d}, TP:{:d}, FN:{:d}, FP:{:d}'.format(best_test_acc, best_epoc, conf_mat[0], conf_mat[1], conf_mat[2], conf_mat[3]))