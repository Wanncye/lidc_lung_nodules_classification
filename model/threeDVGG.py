import torch.nn as nn
import math
import torch


class VGG(nn.Module):

    def __init__(self, features, dropout_rate, fc_feature_dim=512, num_classes=2, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*1*4*4, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
        )
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(fc_feature_dim, num_classes)
        self.fc3 = nn.Linear(512, num_classes)

        self.confidence = nn.Linear(512, 1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, gcn_feature, add_gcn_middle_feature):
        x = self.features(x)
        x = x.view(x.size(0), -1)  #16*8192
        x = self.classifier(x)
        feature = self.fc1(x)    #得到的512维特征
        if add_gcn_middle_feature:
            x1 = torch.cat((feature,gcn_feature),axis=1)
            x2 = self.fc2(x1)
        else:
            x2 = self.fc3(feature)

        # confidence = self.confidence(feature)

        # return x2, feature, confidence
        return x2, feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=3, stride=2, padding=1)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(fc_feature_dim, **kwargs):
    model = VGG(make_layers(cfg['A']), fc_feature_dim, **kwargs)
    return model


def vgg11_bn(fc_feature_dim, dropout_rate,**kwargs):
    model = VGG(make_layers(cfg['A'], batch_norm=True), dropout_rate, fc_feature_dim, **kwargs)
    return model


def vgg13(fc_feature_dim, **kwargs):
    model = VGG(make_layers(cfg['B']), fc_feature_dim, **kwargs)
    return model


def vgg13_bn(dropout_rate,fc_feature_dim,**kwargs):
    model = VGG(make_layers(cfg['B'], batch_norm=True), dropout_rate, fc_feature_dim, **kwargs)
    return model


def vgg16(fc_feature_dim, **kwargs):
    model = VGG(make_layers(cfg['D']),fc_feature_dim,  **kwargs)
    return model


def vgg16_bn(fc_feature_dim, dropout_rate,**kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), dropout_rate, fc_feature_dim, **kwargs)
    return model


def vgg19(fc_feature_dim, **kwargs):
    model = VGG(make_layers(cfg['E']), fc_feature_dim, **kwargs)
    return model


def vgg19_bn(fc_feature_dim, dropout_rate,**kwargs):
    model = VGG(make_layers(cfg['E'], batch_norm=True), dropout_rate, fc_feature_dim, **kwargs)
    return model


if __name__ == '__main__':
    # 'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'
    # Example
    net11 = vgg11()
    print(net11)

