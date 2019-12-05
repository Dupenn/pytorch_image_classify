# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class simpleconv3(nn.Module):
    def __init__(self):
        super(simpleconv3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=24)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=48)
        self.fc1 = nn.Linear(in_features=48 * 5 * 5, out_features=1200)
        self.fc2 = nn.Linear(in_features=1200, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        # print "bn1 shape",x.shape
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 48 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    from visualize import make_dot

    x = Variable(torch.randn(1, 3, 48, 48))
    model = simpleconv3()
    y = model(x)
    g = make_dot(y)
    g.view()
