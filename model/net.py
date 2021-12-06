"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1) #就是将多行的tensor拉平成1行

def Net(params):
    model = models.resnet18(pretrained=False)
    ## add final FC layer
    # model = torch.nn.Sequential(model, torch.nn.Linear(1000, 2))
    # model.ap = torch.nn.AdaptiveAvgPool2d(output_size=1)
    # model.bn1 = torch.nn.BatchNorm1d(1000, eps = 1e-05, momentum=.1, affine=True, track_running_stats=True)
    # model.do1 = torch.nn.Dropout(params.dropout_rate, inplace=True)
    num_features = model.fc.in_features
    num_feat2 = int(num_features / 2)
    model.fc = torch.nn.Linear(num_features, num_feat2)
    model.add_module("flatten", Flatten())
    tail = torch.nn.Sequential(
        torch.nn.ReLU(inplace = True),
        torch.nn.BatchNorm2d(num_feat2),
        # torch.nn.Dropout(params.dropout_rate, inplace=True),
        torch.nn.Dropout(.25, inplace=True),
        torch.nn.Linear(num_feat2, 2)
        )
    model.add_module("tail", tail)
    # model = torch.nn.Sequential(*model.children(), *tail)
    # # model.add_module("tail", tail)
    # out0  = model(im0[0].unsqueeze(0))
    # out0.shape
    # model.fc = torch.nn.Linear(num_features, num_feat2)
    # model.add_module("fc_rl1", torch.nn.ReLU(inplace = True))
    # model.add_module("fc_bn1", torch.nn.BatchNorm2d(num_feat2))
    # model.add_module("fc_do1", torch.nn.Dropout(params.dropout_rate, inplace=True))
    # model.add_module("fc_fc2", torch.nn.Linear(num_feat2, 2))

    return model



def loss_fn_BCE(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_class = 2
    loss = nn.BCEWithLogitsLoss()
    N = len(labels)
    target = torch.zeros(N, num_class).cuda()
    target.scatter_(dim=1,index=labels.unsqueeze(dim=1),src=torch.ones(N, num_class).cuda())
    target = target.to(torch.float32).cuda()
    return loss(outputs, target)

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi铿乪d examples (p > .5),
                                   putting more focus on hard, misclassi铿乪d examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num=2, alpha=0.25, gamma=2, size_average=False):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = torch.FloatTensor([alpha])
                # self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1).cuda()

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        if self.alpha.shape == torch.Tensor([1]).shape:
            alpha = torch.ones(P.shape[0],P.shape[1]).cuda()
            alpha[:,0] = alpha[:,0] * (1-self.alpha)
            alpha[:,1] = alpha[:,1] * self.alpha
            alpha = (alpha * class_mask).sum(dim=1).view(-1,1)
        else:

            alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}


if __name__ == "__main__":
    alpha = torch.rand(21, 1)
    print(alpha)
    FL = FocalLoss(class_num=5, gamma=5, alpha=0.75)
    CE = nn.CrossEntropyLoss()
    N = 4
    C = 5
    inputs = torch.rand(N, C)
    targets = torch.LongTensor(N).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())

    inputs_ce = Variable(inputs.clone(), requires_grad=True)
    targets_ce = Variable(targets.clone())
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)

    fl_loss = FL(inputs_fl, targets_fl)
    ce_loss = CE(inputs_ce, targets_ce)
    print('ce = {}, fl ={}'.format(ce_loss.item(), fl_loss.item()))
    fl_loss.backward()
    ce_loss.backward()
    # print(inputs_fl.grad.data)
    print(inputs_ce.grad.data)
