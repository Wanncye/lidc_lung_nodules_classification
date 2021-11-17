import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool3d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x): #16,96,2,32,32
        if self.ACROSS_CHANNELS:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class AlexNet(nn.Module):

    def __init__(self, fc_feature_dim=512, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 96, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # LRN(local_size=3, alpha=0.0001, beta=0.75),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(96, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # LRN(local_size=3, alpha=0.0001, beta=0.75),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=2),
            nn.Conv3d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=2),
        )
        self.linear1 = nn.Linear(256 * 7 * 7, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(4096, 512)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.linear3 = nn.Linear(fc_feature_dim, num_classes)
        self.linear4 = nn.Linear(512, num_classes)

    def forward(self, x, gcn_feature, add_gcn_middle_feature):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        feature = self.linear2(x)
        x = self.relu2(feature)
        x = self.dropout2(x)
        if add_gcn_middle_feature:
            x = torch.cat((x,gcn_feature),axis=1)
            output = self.linear3(x)
        else:
            output = self.linear4(x)
        return output, feature


def alexnet(fc_feature_dim, pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(fc_feature_dim, **kwargs)
    if pretrained:
        model_path = 'model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model