import torchvision.models as models
import model.data_loader as data_loader

import torch

from model_pretrain.resnet import ResNet34
from model_pretrain.vgg import vgg13_bn
from model_pretrain.alexnet import AlexNet
from model_pretrain.densenet import DenseNet201

batch_size = 1
model_name_list = ['densenet201','alexnet','vgg13','resnet34',]
for model_name in model_name_list:
    #加载模型
    if model_name == 'resnet34':
        print('Now is turn to resnet34-----')
        resnet34_dict = models.resnet34(pretrained=True).state_dict()
        model = ResNet34().cuda()
        model.load_state_dict(resnet34_dict, strict=False)
    if model_name == 'vgg13':
        print('Now is turn to vgg13-----')
        vgg13_dict = models.vgg13_bn(pretrained=True).state_dict()
        model = vgg13_bn().cuda()
        model.load_state_dict(vgg13_dict, strict=False)
    if model_name == 'alexnet':
        print('Now is turn to alexnet-----')
        alexnet_dict = models.alexnet(pretrained=True).state_dict()
        model = AlexNet().cuda()
        model.load_state_dict(alexnet_dict, strict=False)
    if model_name == 'densenet201':
        print('Now is turn to densenet201-----')
        densenet201_dict = models.densenet201(pretrained=True).state_dict()
        model = DenseNet201().cuda()
        model.load_state_dict(densenet201_dict, strict=False)

    for fold in range(5):
        #加载数据
        dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = batch_size, data_dir="data/5fold_128/fold"+str(fold+1), train_shuffle=False)
        train_dl = dataloaders['train']
        test_dl = dataloaders['test']

        #提取特征
        with torch.no_grad():
            model.eval()
            train_feature = torch.zeros((len(train_dl),512))
            test_feature = torch.zeros((len(test_dl),512))
            for i, (x, target, _) in enumerate(train_dl):
                _, feature = model(x.cuda())
                train_feature[(i*batch_size):((i+1)*batch_size), :] = feature.detach()
            for i, (x, target, _) in enumerate(test_dl):
                _, feature = model(x.cuda())
                test_feature[(i*batch_size):((i+1)*batch_size), :] = feature.detach()
            torch.save(train_feature,'./data/pretrain_feature/fold' + str(fold+1) + '/' + model_name + '_train.pt')
            torch.save(test_feature,'./data/pretrain_feature/fold'+ str(fold+1) + '/' + model_name + '_test.pt')

        if model_name == 'densenet201':
            dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = 700, data_dir="data/5fold_128/fold"+str(fold+1), train_shuffle=False)
            train_dl = dataloaders['train']
            test_dl = dataloaders['test']
            for i, (train_batch, labels_batch, _) in enumerate(train_dl):
                torch.save(labels_batch,'./data/pretrain_feature/fold' + str(fold+1) + '/train_label.pt')
            for i, (train_batch, labels_batch, _) in enumerate(test_dl):
                torch.save(labels_batch,'./data/pretrain_feature/fold' + str(fold+1) + '/test_label.pt')



    