import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import statistics as stats
import model.dgi_encoder as models
from pathlib import Path
import model.data_loader as data_loader
import logging
import utils
import os
from torch.optim.lr_scheduler import StepLR,MultiStepLR


def precision(confusion):
    correct = confusion * torch.eye(confusion.shape[0])
    incorrect = confusion - correct
    correct = correct.sum(0)
    incorrect = incorrect.sum(0)
    # precision = correct / (correct + incorrect)
    total_correct = correct.sum().item()
    total_incorrect = incorrect.sum().item()
    acc = total_correct / (total_correct + total_incorrect)
    precision = (confusion[1][1]/(confusion[1][0]+confusion[1][1])).item()
    return acc,precision


if __name__ == '__main__':
    utils.set_logger(os.path.join('./experiments/dgi', 'classification_feature.256.log'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 80
    num_classes = 2
    fully_supervised = False
    epochs = 300
    data_dir = './data/nodules3d_128_npy'
    reload = None

    # image size 3, 32, 32
    # batch size must be an even number
    # shuffle must be True
    dataloaders = data_loader.fetch_dataloader(types = ["train", "test"], batch_size = batch_size, data_dir=data_dir)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    if fully_supervised:
        classifier = nn.Sequential(
            models.Encoder(),
            models.Classifier()
        ).to(device)
    else:
        classifier = models.DeepInfoAsLatent('./experiments/dgi/encoder_feature.256_epoch.81.wgt').to(device)
        if reload is not None:
            classifier = torch.load(f'./experiments/dgi/classification_{reload}.mdl')

    optim = Adam(classifier.parameters(), lr=1e-4)
    scheduler = MultiStepLR(optim, milestones=[50,100,20], gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        ll = []
        with tqdm(total=len(train_dl)) as t:
            for i, (x, target, _) in enumerate(train_dl):
                batch_ll=[]
                x = x.to(device)
                target = target.to(device)

                optim.zero_grad()
                y = classifier(x)
                loss = criterion(y, target)
                ll.append(loss.detach().item())
                batch_ll.append(loss.detach().item())
                # t.set_description(f'{epoch} Train Loss: {stats.mean(ll)}')
                t.set_description(f'{epoch} Train Loss: {stats.mean(batch_ll)}')
                loss.backward()
                optim.step()
                t.update()

        confusion = torch.zeros(num_classes, num_classes)
        ll = []
        total = len(test_dl)
        with tqdm(total=len(test_dl)) as t:
            for i, (x, target, _) in enumerate(test_dl):
                batch_ll=[]
                x = x.to(device)
                target = target.to(device)

                y = classifier(x)
                loss = criterion(y, target)
                ll.append(loss.detach().item())
                batch_ll.append(loss.detach().item())
                # t.set_description(f'{epoch} Test Loss: {stats.mean(ll)}')   #哪有给整个loss去均值的loss方法？这样能看出下降了？
                t.set_description(f'{epoch} Test Loss: {stats.mean(batch_ll)}')

                _, predicted = y.detach().max(1)

                for item in zip(predicted, target):
                    confusion[item[0], item[1]] += 1

                t.update()

        acc,precis = precision(confusion)
        logging.info('epoch:'+str(epoch)+' acc:'+str(acc)+' precision:'+str(precis))
        print('\n')
        scheduler.step()
        if epoch % 100 == 0:
            classifier_save_path = Path('./experiments/dgi' + '/classification_feature.256_epoch.' + str(epoch) + '.mdl')
            torch.save(classifier, str(classifier_save_path))
