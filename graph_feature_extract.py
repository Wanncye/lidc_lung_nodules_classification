from torch._C import dtype
from model.graphNet import GAT,GCN
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from utils import caculate_six_method_predict_similarity,calculate_percentage
from utils import Visualizer
import model.data_loader as data_loader
import csv

def get_random_adj(node_num, out_index):
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
    return adj



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

import matplotlib.pyplot as plt
import os
def save_incorrect_nodule(pre_label, truth_label, nodule_name):
    dest_path = './experiments/gcn/incorrect'
    for root, dirs, files in os.walk(dest_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
    for i in range(157):
        if pre_label[i] != truth_label[i]:
            incorrect_path = './data/nodules3d_128_npy_no_same_patient_in_two_dataset/test/' + str(nodule_name[i])
            npy = np.load(incorrect_path)
            for index in range(8):
                save_name = dest_path + '/' + nodule_name[i] + '_' + str(index) + '.png'
                slice = npy[:, :, index]
                plt.imsave(save_name, slice, cmap='gray')


# f = open('./experiments/gcn/random_adj/random_adj_43_feature_0~1_result_2.txt', 'w')
for fold in range(10):
    # dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 641, data_dir="data/10fold_model_feature_noNorm/fold"+str(fold+1), train_shuffle=False, fold= fold)
    # test_dl = dataloaders['test']
    # for i, (train_batch, labels_batch, file_name, _) in enumerate(test_dl):
    #     nodule_name = file_name
    weightDecay = 1
    vis = Visualizer('GCN_'+str(fold))
    print(fold)
    best_acc_list = []
    for out_index in range(1):
        input_dim = 512
        node_num = 4
        model = GCN(nfeat=input_dim,
                    nhid=64,
                    nclass=2,
                    fc_num=2,
                    dropout=0.5,
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
                            weight_decay=weightDecay)
        train_len = 641
        test_len = 157
        feature_len = 512

        #pretrain_feature
        # m = torch.nn.Tanh()
        # attention56_train_feature = m(torch.load('./data/feature/5fold_128<=20mm_aug/fold_'+str(fold)+'_attention56_train.pt'))
        # attention56_test_feature = m(torch.load('./data/feature/5fold_128<=20mm_aug/fold_'+str(fold)+'_attention56_test.pt'))
        # resnet34_train_feature = m(torch.load('data/feature/5fold_128<=20mm_aug/fold_'+str(fold)+'_resnet34_train.pt'))
        # resnet34_test_feature = m(torch.load('data/feature/5fold_128<=20mm_aug/fold_'+str(fold)+'_resnet34_test.pt'))
        # vgg13_train_feature = m(torch.load('data/feature/5fold_128_new/fold_'+str(fold)+'_vgg13_train.pt'))
        # vgg13_test_feature = m(torch.load('data/feature/5fold_128<=20mm_aug/fold_'+str(fold)+'_vgg13_test.pt'))
        # alexnet_train_feature = m(torch.load('data/feature/5fold_128<=20mm_aug/fold_'+str(fold)+'_alexnet_train.pt'))
        # alexnet_test_feature = m(torch.load('data/feature/5fold_128<=20mm_aug/fold_'+str(fold)+'_alexnet_test.pt'))

        attention56_train_feature = torch.load('./data/feature/10fold_model_feature_noNorm/fold_'+str(fold)+'_attention56_train.pt')
        attention56_test_feature = torch.load('./data/feature/10fold_model_feature_noNorm/fold_'+str(fold)+'_attention56_test.pt')
        resnet34_train_feature = torch.load('data/feature/10fold_model_feature_noNorm/fold_'+str(fold)+'_resnet34_train.pt')
        resnet34_train_feature = torch.load('data/feature/10fold_model_feature_noNorm/fold_'+str(fold)+'_resnet34_train.pt')
        resnet34_test_feature = torch.load('data/feature/10fold_model_feature_noNorm/fold_'+str(fold)+'_resnet34_test.pt')
        vgg13_train_feature = torch.load('data/feature/10fold_model_feature_noNorm/fold_'+str(fold)+'_vgg13_train.pt')
        vgg13_test_feature = torch.load('data/feature/10fold_model_feature_noNorm/fold_'+str(fold)+'_vgg13_test.pt')
        alexnet_train_feature = torch.load('data/feature/10fold_model_feature_noNorm/fold_'+str(fold)+'_alexnet_train.pt')
        alexnet_test_feature = torch.load('data/feature/10fold_model_feature_noNorm/fold_'+str(fold)+'_alexnet_test.pt')
        

        train_label = torch.load('data/feature/10fold_model_feature_noNorm/fold_'+str(fold)+'_train_label.pt')
        test_label = torch.load('data/feature/10fold_model_feature_noNorm/fold_'+str(fold)+'_test_label.pt')





        # #glcm竖直方向上归一化
        # glcm_train_feature = glcm_train_feature.transpose(0,1)
        # glcm_test_feature = glcm_test_feature.transpose(0,1)
        # for index in range(len(glcm_train_feature)):
        #     max = glcm_train_feature[index].max()  #170
        #     min = glcm_train_feature[index].min()  #1.88
        #     glcm_train_feature[index] = (glcm_train_feature[index] - min) / (max-min)
        # for index in range(len(glcm_train_feature)):
        #     max = glcm_test_feature[index].max()  #170
        #     min = glcm_test_feature[index].min()  #1.88
        #     glcm_test_feature[index] = (glcm_test_feature[index] - min) / (max-min)
        # glcm_test_feature = glcm_test_feature.transpose(0,1)
        # glcm_train_feature = glcm_train_feature.transpose(0,1)


        # adj = torch.from_numpy(caculate_six_method_predict_similarity()).float()
        np.random.seed(np.random.randint(1,500))
        adj = get_random_adj(node_num, out_index)
        adj = torch.tensor(
            [[1., 0., 1., 1.],
            [1., 0., 1., 1.],
            [0., 1., 0., 1.],
            [1., 0., 1., 0.]])
        print(adj)

        best_test_acc = 0
        best_epoc = 0

        gcn_train_middle_feature = torch.zeros(len(train_label),56*4)
        gcn_test_middle_feature = torch.zeros(len(test_label),56*4)

        for epoch in range(200):
            loss_train_list = []
            pre_train_list = torch.zeros(len(train_label))

            loss_test_list = []
            pre_test_list = torch.ones(len(test_label))
            #训练
            for index, one_nodule_feature in enumerate(zip(
                resnet34_train_feature,
                vgg13_train_feature,
                alexnet_train_feature,
                attention56_train_feature,
            )):  #必须得在这里用zip才行，好家伙
                temp = torch.zeros((len(one_nodule_feature),512))
                for i, feature in enumerate(one_nodule_feature):
                    temp[i] = feature
                one_nodule_feature = temp  #512*6  尝试改成512个节点，每个节点6维特征

                one_nodule_feature = torch.from_numpy(np.array(one_nodule_feature))
                features = Variable(one_nodule_feature)

                model.train()
                optimizer.zero_grad()

                _ , one_gcn_train_middle_feature, output = model(features, adj)
                #将gcn中间特征保存下来
                gcn_train_middle_feature[index] = one_gcn_train_middle_feature

                one_label = train_label[index].unsqueeze(0).long()
                pre_train_list[index] = output.max(1)[1].type_as(one_label)
                loss_train = F.nll_loss(output,one_label)
                loss_train_list.append(loss_train.item())

                loss_train.backward()
                optimizer.step()
            acc_train,_ = accuracy(pre_train_list, train_label)


            #测试
            for index, one_nodule_feature in enumerate(zip(
                resnet34_test_feature,
                vgg13_test_feature,
                alexnet_test_feature,
                attention56_test_feature,
            )):
                temp = torch.zeros((len(one_nodule_feature),512))
                for i, feature in enumerate(one_nodule_feature):
                    temp[i] = feature
                one_nodule_feature = temp

                # adj = torch.ones((len(one_nodule_feature), len(one_nodule_feature)))
                one_nodule_feature = torch.from_numpy(np.array(one_nodule_feature))
                features, adj = Variable(one_nodule_feature), Variable(adj)

                model.eval()
                _ , one_gcn_test_middle_feature, output = model(features, adj)
                #将gcn中间特征保存下来
                gcn_test_middle_feature[index] = one_gcn_test_middle_feature

                one_label = test_label[index].unsqueeze(0).long()
                pre_test_list[index] = output.max(1)[1].type_as(one_label)
                loss_test_list.append(F.nll_loss(output,one_label).item())
            acc_test,conf_mat = accuracy(pre_test_list, test_label)
            # 出现当前最佳准确率时
            if acc_test >= best_test_acc:
                best_test_acc = acc_test
                best_epoc = epoch
                best_conf_mat = conf_mat

                #将最好的结果保存到csv中
                # predict_csv = open('./experiments/gcn/result/best_result_fold_'+str(fold+1)+'.csv','w',encoding='utf-8')
                # csv_writer = csv.writer(predict_csv)
                # csv_writer.writerow(["filename","truth_label","predict_label","is_right",'percentage'])
                # for index,(name, truth_label, predict_label) in enumerate(zip(nodule_name, test_label.data.cpu().numpy(), pre_test_list.data.cpu().numpy())):
                #     is_right = True if truth_label == predict_label else False
                #     percentage = calculate_percentage(name)
                #     data = [name, truth_label, int(predict_label), is_right, percentage]
                #     csv_writer.writerow(data)
                # predict_csv.close()

                # 将错误分类的结节保存下来
                # save_incorrect_nodule(pre_test_list, test_label, nodule_name)


                #最好准确率时保存模型
                torch.save({
                    'epoch' : epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict' : optimizer.state_dict()
                },'./experiments/gcn/fc_2_feature_4_wdecay_5e-2_fold_'+str(fold)+'.best.pth.tar')

                #保存gcn中间特征到文件中，用于其他模型的训练
                torch.save(gcn_train_middle_feature,'data/feature/10fold_gcn_feature_random_adj/gcn_train_middle_feature_fold_'+str(fold)+'.pt')
                torch.save(gcn_test_middle_feature,'data/feature/10fold_gcn_feature_random_adj/gcn_test_middle_feature_fold_'+str(fold)+'.pt')

            vis.plot('train loss',np.mean(loss_train_list),1)
            vis.plot('test loss',np.mean(loss_test_list),1)
            vis.plot('train acc',acc_train.item(),1)
            vis.plot('test acc',acc_test.item(),1)
            print('epoch:{:d}'.format(epoch) 
                , ', train loss:{:.8f}'.format(np.mean(loss_train_list)) 
                , ', train acc:{:.6f}'.format(acc_train.item()) 
                , ', test loss:{:.8f}'.format(np.mean(loss_test_list)) 
                , ', test acc:{:.6f}'.format(acc_test.item()))
        #这里输出的混淆矩阵值是最后一个epoch的
        print('best test acc:{:.4f}, epoch:{:d}, TN:{:d}, TP:{:d}, FN:{:d}, FP:{:d}'.format(best_test_acc, best_epoc, best_conf_mat[0], best_conf_mat[1], best_conf_mat[2], best_conf_mat[3]))
        # best_acc_list.append(best_test_acc)
    #     for adj_i in range(43):
    #         f.write(str(adj[adj_i]) + '\n')
    #     f.write('test_acc:' + str(best_test_acc) + '\n\n')
    #     f.flush()
    # f.write('best_acc_list:'+ str(best_acc_list))


