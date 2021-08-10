import torch
import torch.nn as nn
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size*growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))

    # 重载forward函数
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            _DenseLayer(in_channels+growth_rate*i,
                                        growth_rate, bn_size))


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet_BC(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6,12,24,16),
                 bn_size=4, theta=0.5, num_classes=2):
        super(DenseNet_BC, self).__init__()

        # 初始的卷积为filter:2倍的growth_rate
        num_init_feature = 2 * growth_rate

        # 表示LIDC
        if num_classes == 2:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv3d(1, num_init_feature,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False)),
            ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm3d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool3d((1, 1, 1)))

        self.fc1 = nn.Linear(num_feature, 512)
        self.fc2 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.bias, 0)
                
    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        feature = self.fc1(out)
        out = self.fc2(feature)
        return out, feature


# DenseNet_BC for ImageNet
def DenseNet121():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=2)

def DenseNet169():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=2)

def DenseNet201():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=2)

def DenseNet161():
    return DenseNet_BC(growth_rate=48, block_config=(6, 12, 36, 24), num_classes=2)

