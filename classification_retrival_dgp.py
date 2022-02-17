import model.dgi_encoder as model
import model.data_loader as data_loader
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


encoder = model.Encoder()
encoder.load_state_dict(torch.load('./experiments/dgi/encoder_feature.256_epoch.81.wgt'))
batch_size = 16
dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = batch_size, data_dir='./data/nodules3d_128_npy', train_shuffle=False)
train_dl = dataloaders['train']
train_len = 639
test_dl = dataloaders['test']
test_len = 160
feature_len = 256

#提取训练集和测试集的特征
with tqdm(total=len(train_dl)) as t:
    train_feature = np.zeros((train_len,feature_len))
    train_label = np.zeros((train_len))
    for i, (x, target, _) in enumerate(train_dl):
        feature, _ = encoder(x)
        train_feature[(i*batch_size):((i+1)*batch_size), :] = feature.detach().numpy()
        train_label[i*batch_size:(i+1)*batch_size] = target.detach().numpy()
        t.update()

with tqdm(total=len(test_dl)) as t:
    test_feature = np.zeros((test_len,feature_len))
    test_label = np.zeros((test_len))
    for i, (x, target, _) in enumerate(test_dl):
        feature, _ = encoder(x)        
        test_feature[i*batch_size:(i+1)*batch_size, :] = feature.detach().numpy()
        test_label[i*batch_size:(i+1)*batch_size] = target.detach().numpy()
        t.update()

# 在训练集中检索
print('retrival in train set...')
cos_sim = cosine_similarity(test_feature,train_feature)
sort_cos_sim = np.sort(cos_sim)
index_sort_cos_sim = np.argsort(cos_sim)
top_n = [1,3,5,7,9,10,15,20]
for n in top_n:
    correct_num = 0
    for i in range(test_len):
        true_label = test_label[i]
        big_than_half = 0
        for j in index_sort_cos_sim[i,-1*n:]:
            retrival_label = train_label[j]
            if true_label == retrival_label:
                big_than_half += 1
        if big_than_half >= n/2:
            correct_num += 1
    acc = correct_num/test_len
    print('top_'+str(n)+'_acc = '+str(acc))

# 在测试集中检索
print('retrival in test set...')
cos_sim = cosine_similarity(test_feature,test_feature)
sort_cos_sim = np.sort(cos_sim)
index_sort_cos_sim = np.argsort(cos_sim)
top_n = [1,3,5,7,9,10,15,20]
for n in top_n:
    correct_num = 0
    for i in range(test_len):
        true_label = test_label[i]
        big_than_half = 0
        for j in index_sort_cos_sim[i,-1*n:]:
            retrival_label = test_label[j]
            if true_label == retrival_label:
                big_than_half += 1
        if big_than_half >= n/2:
            correct_num += 1
    acc = correct_num/test_len
    print('top_'+str(n)+'_acc = '+str(acc))
