import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Net(nn.Module):
    '''

    自定义的CNN网络，3个卷积层，包含batch norm。2个pool,
    3个全连接层，包含Dropout
    输入：28x28x1s
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.feature = nn.Sequential(
            OrderedDict(
                [
                    ('conv1', nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=2)),
                    ('relu1', nn.ReLU()),
                    ('bn1', nn.BatchNorm2d(num_features=12)),

                    ('conv2', nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)),
                    ('relu2', nn.ReLU()),
                    ('bn2', nn.BatchNorm2d(num_features=24)),
                    ('pool1', nn.MaxPool2d(kernel_size=2)),

                    ('conv3', nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, stride=1, padding=1)),
                    ('relu3', nn.ReLU()),
                    ('bn3', nn.BatchNorm2d(num_features=48)),
                    ('pool2', nn.MaxPool2d(kernel_size=2)),

                ]
            )
        )

        self.classifier = nn.Sequential(

            OrderedDict(
                [
                    ('fc1', nn.Linear(in_features=48 * 5 * 5, out_features=1200)),
                    ('dropout1', nn.Dropout2d(p=0.5)),
                    ('fc2', nn.Linear(in_features=1200, out_features=128)),
                    ('dropout2', nn.Dropout2d(p=0.6)),
                    ('fc3', nn.Linear(in_features=128, out_features=2))
                ]
            )

        )

    def forward(self, x):
        out = self.feature(x)
        print('x:', x.shape())
        print('out:', out.shape())
        out = out.view(-1, 48 * 5 * 5)
        out = self.classifier(out)
        return out
