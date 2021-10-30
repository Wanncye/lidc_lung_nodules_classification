from sklearn.tree import DecisionTreeClassifier
from utils import set_logger, get_matrix_similarity
import torch
import logging
import model.data_loader as data_loader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
#SVM、决策树
rootPath = 'data/feature/5fold_128/'
set_logger(rootPath + 'SVM_DecisionTreeClassifier.log')
for fold in range(2,3):
    logging.info('fold {0}'.format(fold+1))
    train_feature = torch.load(rootPath + 'fold_'+str(fold)+'_alexnet_train.pt')
    test_feature = torch.load(rootPath + 'fold_'+str(fold)+'_alexnet_test.pt')

    features = torch.cat((train_feature,test_feature), axis=0)
    similarity = torch.from_numpy(get_matrix_similarity(np.array(features),np.array(features)))
    similarity = similarity - torch.diag_embed(torch.diag(similarity))
    similarity, idx = similarity.sort(descending=True)
    
    val_idx = torch.zeros((test_feature.shape[0])).long()
    
    train_idx = torch.zeros(train_feature.shape[0]-val_idx.shape[0]).long()
    for i in range(train_feature.shape[0], features.shape[0]):
        for j in idx[i]:
            if j >=  train_feature.shape[0]:
                continue
            else:
                if j in val_idx:
                    continue
                else:
                    val_idx[i-train_feature.shape[0]] = j
                    break

    train_i = 0
    for i in range(train_feature.shape[0]):
        if i not in val_idx:
            train_idx[train_i] = i
            train_i += 1
            
    # print(train_feature[train_idx])
    # print(train_feature[val_idx])
    # print(train_idx)
    # print(val_idx)
    

    dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 3000, data_dir="data/5fold_128/fold"+str(fold+1), train_shuffle=False,fold=0)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']
    for i, (train_batch, labels_batch, file_name, _) in enumerate(train_dl):
        train_label = np.array(labels_batch)
        train_name = file_name
    for i, (train_batch, labels_batch, file_name, _) in enumerate(test_dl):
        test_label = np.array(labels_batch)
        test_name = file_name

    kernel_function = ['linear', 'poly', 'rbf', 'sigmoid']
    C = [1e-2,1e-1,1,1e1,1e2] #C是对错误的惩罚
    gamma = [0.0001,0.0005,0.001,0.005,0.01,0.1] 
    max_iter = [50,100,200,500,700,1000,1500,2000]
    with tqdm(total = len(kernel_function)*len(C)*len(gamma)*len(max_iter)) as t:
        train_feature = train_feature.detach().numpy()
        test_feature = test_feature.detach().numpy()
        best_accuracy = 0
        for kf in kernel_function:
            for c in C:
                for ga in gamma:
                    for miter in max_iter:
                        clf = SVC(kernel=kf, C=c, gamma=ga, max_iter=miter, probability=True)
                        clf = clf.fit(train_feature[train_idx], train_label[train_idx])

                        train_pred = clf.predict(train_feature[train_idx])
                        train_con_matrix = confusion_matrix(train_label[train_idx],train_pred,labels=range(2))
                        print(train_con_matrix)
                        train_accuracy = (train_con_matrix[0][0] + train_con_matrix[1][1])/len(train_label[train_idx])
                        print('train_accuracy--{0}'.format(train_accuracy))

                        val_pred = clf.predict(train_feature[val_idx])
                        val_con_matrix = confusion_matrix(train_label[val_idx],val_pred,labels=range(2))
                        print(val_con_matrix)
                        val_accuracy = (val_con_matrix[0][0] + val_con_matrix[1][1])/len(train_label[val_idx])
                        sensitivity = val_con_matrix[1][1] / (val_con_matrix[1][1] + val_con_matrix[1][0])
                        specitivity = val_con_matrix[0][0] / (val_con_matrix[0][1] + val_con_matrix[0][0])
                        print('val_accuracy--{0}'.format(val_accuracy))

                        Y_pred = clf.predict(test_feature)
                        # Y_proba = clf.predict_proba(test_feature) * [sensitivity, specitivity]
                        # print('test_prob--{0}'.format(Y_proba))
                        # Y_pred = np.argmax(Y_proba, axis=1)

                        con_matrix = confusion_matrix(test_label,Y_pred,labels=range(2))
                        print(con_matrix)
                        test_accuracy = (con_matrix[0][0] + con_matrix[1][1])/len(test_label)
                        print('test_accuracy--{0}'.format(test_accuracy))
                        print('\n')
                        if test_accuracy > best_accuracy:
                            idx = [i for i in range(len(Y_pred)) if Y_pred[i] != test_label[i]]
                            wrong_classify = [name for i,name in enumerate(test_name) if i in idx]
                            save_matrix = con_matrix
                            param_list = [c, ga, miter,kf]
                            best_accuracy = test_accuracy
                        t.update()
        
    TN = save_matrix[0][0]
    TP = save_matrix[1][1]
    FN = save_matrix[1][0]
    FP = save_matrix[0][1]
    logging.info('TN:{0}, TP:{1}, FN:{2}, FP:{3} '.format(TN, TP, FN, FP))
    logging.info('classify incorrectly nodule:')
    logging.info(wrong_classify)
    logging.info('{0} classification, kernel_function={1}, c={2}, gamma={3}, max_iter={4}, test_accuracy={5}'.format('SVM addition_feature<=20mm_mask', 
                                                                                                                                param_list[3],
                                                                                                                                param_list[0], 
                                                                                                                                param_list[1], 
                                                                                                                                param_list[2],
                                                                                                                                best_accuracy))