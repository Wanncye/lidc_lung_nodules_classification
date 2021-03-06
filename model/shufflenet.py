"""shufflenet in pytorch
[1] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun.
    ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
    https://arxiv.org/abs/1707.01083v2
"""

from functools import partial

import torch
import torch.nn as nn


class BasicConv3d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, depth, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        #"""suppose a convolutional layer with g groups whose output has
        #g x n channels; we first reshape the output channel dimension
        #into (g, n)"""
        x = x.view(batchsize, self.groups, channels_per_group, depth, height, width)

        #"""transposing and then flattening it back as the input of next layer."""
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, depth, height, width)

        return x

class DepthwiseConv3d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size, **kwargs),
            nn.BatchNorm3d(output_channels)
        )

    def forward(self, x):
        return self.depthwise(x)

class PointwiseConv3d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, 1, **kwargs),
            nn.BatchNorm3d(output_channels)
        )

    def forward(self, x):
        return self.pointwise(x)

class ShuffleNetUnit(nn.Module):

    def __init__(self, input_channels, output_channels, stage, stride, groups):
        super().__init__()

        #"""Similar to [9], we set the number of bottleneck channels to 1/4
        #of the output channels for each ShuffleNet unit."""
        self.bottlneck = nn.Sequential(
            PointwiseConv3d(
                input_channels,
                int(output_channels / 4),
                groups=groups
            ),
            nn.ReLU(inplace=True)
        )

        #"""Note that for Stage 2, we do not apply group convolution on the first pointwise
        #layer because the number of input channels is relatively small."""
        if stage == 2:
            self.bottlneck = nn.Sequential(
                PointwiseConv3d(
                    input_channels,
                    int(output_channels / 4),
                    groups=groups
                ),
                nn.ReLU(inplace=True)
            )

        self.channel_shuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv3d(
            int(output_channels / 4),
            int(output_channels / 4),
            3,
            groups=int(output_channels / 4),
            stride=stride,
            padding=1
        )

        self.expand = PointwiseConv3d(
            int(output_channels / 4),
            output_channels,
            groups=groups
        )

        self.relu = nn.ReLU(inplace=True)
        self.fusion = self._add
        self.shortcut = nn.Sequential()

        #"""As for the case where ShuffleNet is applied with stride,
        #we simply make two modifications (see Fig 2 (c)):
        #(i) add a 3 ?? 3 average pooling on the shortcut path;
        #(ii) replace the element-wise addition with channel concatenation,
        #which makes it easy to enlarge channel dimension with little extra
        #computation cost.
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.AvgPool3d((1,3,3), stride=(2,2,2), padding=(0,1,1))

            self.expand = PointwiseConv3d(
                int(output_channels / 4),
                output_channels - input_channels,
                groups=groups
            )
            

            self.fusion = self._cat

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        shortcut = self.shortcut(x)

        shuffled = self.bottlneck(x)
        shuffled = self.channel_shuffle(shuffled)
        shuffled = self.depthwise(shuffled)
        shuffled = self.expand(shuffled)

        output = self.fusion(shortcut, shuffled)
        output = self.relu(output)

        return output

class ShuffleNet(nn.Module):

    def __init__(self, num_blocks, num_classes=2, groups=3, fc_feature_dim=512):
        super().__init__()

        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.conv1 = BasicConv3d(1, out_channels[0], 3, padding=1, stride=1)
        self.input_channels = out_channels[0]

        self.stage2 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[0],
            out_channels[1],
            stride=2,
            stage=2,
            groups=groups
        )

        self.stage3 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[1],
            out_channels[2],
            stride=2,
            stage=3,
            groups=groups
        )

        self.stage4 = self._make_stage(
            ShuffleNetUnit,
            num_blocks[2],
            out_channels[3],
            stride=2,
            stage=4,
            groups=groups
        )

        self.avg = nn.AdaptiveAvgPool3d((1))
        self.fc1 = nn.Linear(out_channels[3], 512)
        self.fc2 = nn.Linear(512, num_classes)
        # self.fc3 = nn.Linear(512 + 512+ 255 + 38, num_classes)
        self.fc3 = nn.Linear(fc_feature_dim, num_classes)

    def forward(self, x, gcn_feature, add_gcn_middle_feature, feature_fusion_method):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
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

    def _make_stage(self, block, num_blocks, output_channels, stride, stage, groups):
        """make shufflenet stage
        Args:
            block: block type, shuffle unit
            out_channels: output depth channel number of this stage
            num_blocks: how many blocks per stage
            stride: the stride of the first block of this stage
            stage: stage index
            groups: group number of group convolution
        Return:
            return a shuffle net stage
        """
        strides = [stride] + [1] * (num_blocks - 1)

        stage = []

        for stride in strides:
            stage.append(
                block(
                    self.input_channels,
                    output_channels,
                    stride=stride,
                    stage=stage,
                    groups=groups
                )
            )
            self.input_channels = output_channels

        return nn.Sequential(*stage)

def shufflenet(fc_feature_dim):
    return ShuffleNet([4, 8, 4], fc_feature_dim=fc_feature_dim)