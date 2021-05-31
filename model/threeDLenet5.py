from torch.nn import Module
from torch import nn


class lenet5(Module):
    def __init__(self):
        super(lenet5, self).__init__()
        self.conv1 = nn.Conv3d(1, 6, (3, 5, 5))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d((2,3,3))
        self.conv2 = nn.Conv3d(6, 16, (3, 5, 5))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d((1,3,3))
        self.fc1 = nn.Linear(2304, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(512, 2)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout3d(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = self.dropout(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        feature = self.relu4(y)
        y = self.fc3(feature)
        y = self.relu5(y)
        return y, feature