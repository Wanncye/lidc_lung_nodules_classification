"""mobilenet in pytorch
[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
"""

import torch
import torch.nn as nn


class DepthSeperabelConv3d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv3d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            nn.BatchNorm3d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicConv3d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.conv = nn.Conv3d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MobileNet(nn.Module):

    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, class_num=2, fc_feature_dim=512):
       super().__init__()

       alpha = width_multiplier
       self.stem = nn.Sequential(
           BasicConv3d(1, int(32 * alpha), 3, padding=1, bias=False),
           DepthSeperabelConv3d(
               int(32 * alpha),
               int(64 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv1 = nn.Sequential(
           DepthSeperabelConv3d(
               int(64 * alpha),
               int(128 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv3d(
               int(128 * alpha),
               int(128 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv2 = nn.Sequential(
           DepthSeperabelConv3d(
               int(128 * alpha),
               int(256 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv3d(
               int(256 * alpha),
               int(256 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv3 = nn.Sequential(
           DepthSeperabelConv3d(
               int(256 * alpha),
               int(512 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),

           DepthSeperabelConv3d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv3d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv3d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv3d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv3d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv4 = nn.Sequential(
           DepthSeperabelConv3d(
               int(512 * alpha),
               int(1024 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv3d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       self.fc1 = nn.Linear(int(1024 * alpha), 512)
       self.relu = nn.ReLU(inplace=True)
       self.fc2 = nn.Linear(512, class_num)
    #    self.fc3 = nn.Linear(512 + 512+ 255 + 38, class_num)
       self.fc3 = nn.Linear(fc_feature_dim, class_num)
       self.avg = nn.AdaptiveAvgPool3d(1)

    def forward(self, x, gcn_feature, add_gcn_middle_feature, feature_fusion_method):
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        feature = self.fc1(x)
        if add_gcn_middle_feature:
            if feature_fusion_method == 'cat':
                catX = torch.cat((feature,gcn_feature),axis=1)
            elif feature_fusion_method == 'add':
                sub_feature_1 = gcn_feature[:,:512]
                sub_feature_2 = gcn_feature[:,512:]
                x1 = feature + sub_feature_1
                catX = torch.cat((x1,sub_feature_2),axis=1)
            elif feature_fusion_method == 'avg':
                sub_feature_1 = gcn_feature[:,:512]
                sub_feature_2 = gcn_feature[:,512:]
                x1 = (feature + sub_feature_1)/2
                catX = torch.cat((x1,sub_feature_2),axis=1)
            x = self.fc3(catX)
        else:
            x = self.fc2(feature)
        return x,feature


def mobilenet(alpha=1, class_num=2, fc_feature_dim=512):
    return MobileNet(alpha, class_num, fc_feature_dim=fc_feature_dim)