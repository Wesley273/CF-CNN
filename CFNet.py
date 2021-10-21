import torch.nn as nn


class CFNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CFNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 36, 3, padding=1),
                                   nn.BatchNorm2d(36), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(36, 36, 3, padding=1),
                                   nn.BatchNorm2d(36), nn.PReLU())
        self.pool1 = nn.Sequential(nn.MaxPool2d(2), nn.BatchNorm2d(36),
                                   nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(36, 48, 3, padding=1),
                                   nn.BatchNorm2d(48), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1),
                                   nn.BatchNorm2d(48), nn.PReLU())
        self.pool2 = nn.Sequential(nn.MaxPool2d(2), nn.BatchNorm2d(48),
                                   nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(48, 68, 3, padding=1),
                                   nn.BatchNorm2d(68), nn.PReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(68, 68, 3, padding=1),
                                   nn.BatchNorm2d(68), nn.PReLU())
        self.fc7 = nn.Sequential(nn.Linear(9 * 9 * 68, 300),
                                 nn.BatchNorm1d(300), nn.PReLU())
        self.fc8 = nn.Sequential(nn.Linear(300, out_ch))

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(self.pool1(c2))
        c4 = self.conv4(c3)
        c5 = self.conv5(self.pool2(c4))
        c6 = self.conv6(c5)
        f7 = self.fc7(c6)
        f8 = self.fc8(f7)
        out = nn.Sigmoid()(f8)
        return out