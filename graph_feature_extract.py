from torch._C import dtype
from model.graphNet import GAT,GCN
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from utils import caculate_six_method_predict_similarity

def normalize_features(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diagflat(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(pred, target):
    correct = pred.eq(target).double()
    correct = correct.sum()
    con_matrix = confusion_matrix(target,pred,labels=range(2))
    TN = con_matrix[0][0]
    TP = con_matrix[1][1]
    FN = con_matrix[1][0]
    FP = con_matrix[0][1]
    return correct / len(target),[TN,TP,FN,FP]

f = open('./experiments/gcn/random_adj/random_adj_43_feature_result.txt', 'w')
best_acc_list = []
for out_index in range(1000):
    input_dim = 512
    node_num = 43
    model = GCN(nfeat=input_dim,
                nhid=64,
                nclass=2,
                fc_num=2,
                dropout=0.6,
                ft=node_num)
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
    alexnet_train_feature = torch.load('./data/feature/alexnet_train.pt')
    alexnet_test_feature = torch.load('./data/feature/alexnet_test.pt')
    attention56_train_feature = torch.load('./data/feature/attention56_train.pt')
    attention56_test_feature = torch.load('./data/feature/attention56_test.pt')
    attention92_train_feature = torch.load('./data/feature/attention92_train.pt')
    attention92_test_feature = torch.load('./data/feature/attention92_test.pt')
    mobilenet_train_feature = torch.load('./data/feature/mobilenet_train.pt')
    mobilenet_test_feature = torch.load('./data/feature/mobilenet_test.pt')
    mobilenetv2_train_feature = torch.load('./data/feature/mobilenetv2_train.pt')
    mobilenetv2_test_feature = torch.load('./data/feature/mobilenetv2_test.pt')
    shufflenet_train_feature = torch.load('./data/feature/shufflenet_train.pt')
    shufflenet_test_feature = torch.load('./data/feature/shufflenet_test.pt')
    squeezenet_train_feature = torch.load('./data/feature/squeezenet_train.pt')
    squeezenet_test_feature = torch.load('./data/feature/squeezenet_test.pt')
    preactresnet18_train_feature = torch.load('./data/feature/preactresnet18_train.pt')
    preactresnet18_test_feature = torch.load('./data/feature/preactresnet18_test.pt')

    preactresnet34_train_feature = torch.load('./data/feature/preactresnet34_train.pt')
    preactresnet34_test_feature = torch.load('./data/feature/preactresnet34_test.pt')
    inceptionv3_train_feature = torch.load('./data/feature/inceptionv3_train.pt')
    inceptionv3_test_feature = torch.load('./data/feature/inceptionv3_test.pt')
    densenet121_train_feature = torch.load('./data/feature/densenet121_train.pt')
    densenet121_test_feature = torch.load('./data/feature/densenet121_test.pt')
    densenet161_train_feature = torch.load('./data/feature/densenet161_train.pt')
    densenet161_test_feature = torch.load('./data/feature/densenet161_test.pt')
    densenet169_train_feature = torch.load('./data/feature/densenet169_train.pt')
    densenet169_test_feature = torch.load('./data/feature/densenet169_test.pt')
    densenet201_train_feature = torch.load('./data/feature/densenet201_train.pt')
    densenet201_test_feature = torch.load('./data/feature/densenet201_test.pt')
    
    preactresnet50_train_feature = torch.load('./data/feature/preactresnet50_train.pt')
    preactresnet50_test_feature = torch.load('./data/feature/preactresnet50_test.pt')
    preactresnet101_train_feature = torch.load('./data/feature/preactresnet101_train.pt')
    preactresnet101_test_feature = torch.load('./data/feature/preactresnet101_test.pt')
    preactresnet152_train_feature = torch.load('./data/feature/preactresnet152_train.pt')
    preactresnet152_test_feature = torch.load('./data/feature/preactresnet152_test.pt')
    resnext50_train_feature = torch.load('./data/feature/resnext50_train.pt')
    resnext50_test_feature = torch.load('./data/feature/resnext50_test.pt')
    resnext101_train_feature = torch.load('./data/feature/resnext101_train.pt')
    resnext101_test_feature = torch.load('./data/feature/resnext101_test.pt')
    resnext152_train_feature = torch.load('./data/feature/resnext152_train.pt')
    resnext152_test_feature = torch.load('./data/feature/resnext152_test.pt')
    resnet_in_resnet_train_feature = torch.load('./data/feature/resnet_in_resnet_train.pt')
    resnet_in_resnet_test_feature = torch.load('./data/feature/resnet_in_resnet_test.pt')
    senet18_train_feature = torch.load('./data/feature/senet18_train.pt')
    senet18_test_feature = torch.load('./data/feature/senet18_test.pt')
    senet34_train_feature = torch.load('./data/feature/senet34_train.pt')
    senet34_test_feature = torch.load('./data/feature/senet34_test.pt')
    senet50_train_feature = torch.load('./data/feature/senet50_train.pt')
    senet50_test_feature = torch.load('./data/feature/senet50_test.pt')
    senet101_train_feature = torch.load('./data/feature/senet101_train.pt')
    senet101_test_feature = torch.load('./data/feature/senet101_test.pt')
    senet152_train_feature = torch.load('./data/feature/senet152_train.pt')
    senet152_test_feature = torch.load('./data/feature/senet152_test.pt')
    xception_train_feature = torch.load('./data/feature/xception_train.pt')
    xception_test_feature = torch.load('./data/feature/xception_test.pt')
    wideresidual_train_feature = torch.load('./data/feature/wideresidual_train.pt')
    wideresidual_test_feature = torch.load('./data/feature/wideresidual_test.pt')

    train_label = torch.load('./data/feature/train_label.pt')
    test_label = torch.load('./data/feature/test_label.pt')

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

    # np.random.seed(2021)
    # adj = torch.from_numpy(caculate_six_method_predict_similarity()).float()
    adj = Variable(torch.ones((node_num, node_num)))
    # #将邻接矩阵改一改,随机置1
    for i in range(node_num):
        for j in range(node_num):
            random_num = np.random.rand()
            if out_index < 500: 
                if random_num > 0.5:
                    adj[i,j] = 0
            elif out_index >= 500: #大于200的循环随机数，而不是0-1矩阵
                adj[i,j] = random_num

    best_test_acc = 0
    best_epoc = 0
    for epoch in range(100):
        loss_train_list = []
        pre_train_list = torch.zeros(len(train_label))

        loss_test_list = []
        pre_test_list = torch.ones(len(test_label))
        #训练
        for index, one_nodule_feature in enumerate(zip(
            googlenet_train_feature,
            resnet10_train_feature,
            vgg16_train_feature,
            hog_train_feature,
            lbp_train_feature,
            glcm_train_feature,
            resnet18_train_feature,
            resnet34_train_feature,
            resnet50_train_feature,
            resnet101_train_feature,
            resnet152_train_feature,
            resnet200_train_feature,
            vgg11_train_feature,
            vgg13_train_feature,
            vgg19_train_feature,
            alexnet_train_feature,
            attention56_train_feature,
            attention92_train_feature,
            mobilenet_train_feature,
            mobilenetv2_train_feature,
            shufflenet_train_feature,
            squeezenet_train_feature,
            preactresnet18_train_feature,
            preactresnet34_train_feature,
            inceptionv3_train_feature,
            densenet121_train_feature,
            densenet161_train_feature,
            densenet169_train_feature,
            densenet201_train_feature,
            preactresnet50_train_feature,
            preactresnet101_train_feature,
            preactresnet152_train_feature,
            resnext50_train_feature,
            resnext101_train_feature,
            resnext152_train_feature,
            resnet_in_resnet_train_feature,
            senet18_train_feature,
            senet34_train_feature,
            senet50_train_feature,
            senet101_train_feature,
            senet152_train_feature,
            xception_train_feature,
            wideresidual_train_feature
        )):  #必须得在这里用zip才行，好家伙
            temp = torch.zeros((len(one_nodule_feature),512))
            for i, feature in enumerate(one_nodule_feature):
                temp[i] = feature
            one_nodule_feature = temp  #512*6  尝试改成512个节点，每个节点6维特征

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
        for index, one_nodule_feature in enumerate(zip(
            googlenet_test_feature,
            resnet10_test_feature,
            vgg16_test_feature,
            hog_test_feature,
            lbp_test_feature,
            glcm_test_feature,
            resnet18_test_feature,
            resnet34_test_feature,
            resnet50_test_feature,
            resnet101_test_feature,
            resnet152_test_feature,
            resnet200_test_feature,
            vgg11_test_feature,
            vgg13_test_feature,
            vgg19_test_feature,
            alexnet_test_feature,
            attention56_test_feature,
            attention92_test_feature,
            mobilenet_test_feature,
            mobilenetv2_test_feature,
            shufflenet_test_feature,
            squeezenet_test_feature,
            preactresnet18_test_feature,
            preactresnet34_test_feature,
            inceptionv3_test_feature,
            densenet121_test_feature,
            densenet161_test_feature,
            densenet169_test_feature,
            densenet201_test_feature,
            preactresnet50_test_feature,
            preactresnet101_test_feature,
            preactresnet152_test_feature,
            resnext50_test_feature,
            resnext101_test_feature,
            resnext152_test_feature,
            resnet_in_resnet_test_feature,
            senet18_test_feature,
            senet34_test_feature,
            senet50_test_feature,
            senet101_test_feature,
            senet152_test_feature,
            xception_test_feature,
            wideresidual_test_feature
        )):
            temp = torch.zeros((len(one_nodule_feature),512))
            for i, feature in enumerate(one_nodule_feature):
                temp[i] = feature
            one_nodule_feature = temp

            # adj = torch.ones((len(one_nodule_feature), len(one_nodule_feature)))
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
            , ', train loss:{:.8f}'.format(np.mean(loss_train_list)) 
            , ', train acc:{:.6f}'.format(acc_train.item()) 
            , ', test loss:{:.8f}'.format(np.mean(loss_test_list)) 
            , ', test acc:{:.6f}'.format(acc_test.item()))
    #这里输出的混淆矩阵值是最后一个epoch的
    print('best test acc:{:.4f}, epoch:{:d}, TN:{:d}, TP:{:d}, FN:{:d}, FP:{:d}'.format(best_test_acc, best_epoc, conf_mat[0], conf_mat[1], conf_mat[2], conf_mat[3]))
    best_acc_list.append(best_test_acc)
    for adj_i in range(43):
        f.write(str(adj[adj_i]) + '\n')
    f.write('test_acc:' + str(best_test_acc) + '\n\n')
    f.flush()
f.write('best_acc_list:'+ str(best_acc_list))

    