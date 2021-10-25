import torch
import torch.nn as nn


class Branch3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch3D, self).__init__()
        # tensor的格式为[B,C,H,W]
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 36, 3, padding=1), nn.BatchNorm2d(36), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(36, 36, 3, padding=1), nn.BatchNorm2d(36), nn.PReLU())
        # 注意ceil_mode的设置
        self.pool1 = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), nn.BatchNorm2d(36), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(36, 48, 3, padding=1), nn.BatchNorm2d(48), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.PReLU())
        self.pool2 = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), nn.BatchNorm2d(48), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(48, 68, 3, padding=1), nn.BatchNorm2d(68), nn.PReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(68, 68, 3, padding=1), nn.BatchNorm2d(68), nn.PReLU())
        self.fc7 = nn.Sequential(nn.Linear(9 * 9 * 68, out_ch), nn.BatchNorm1d(out_ch), nn.PReLU())

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(self.pool1(c2))
        c4 = self.conv4(c3)
        c5 = self.conv5(self.pool2(c4))
        c6 = self.conv6(c5)
        # view()函数用于将tensor除了Batch维，其他维压缩在一起
        f7 = self.fc7(c6.view(c6.size(0), -1))

        return f7


class Branch2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch2D, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, 36, 3, padding=1), nn.BatchNorm2d(36), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(36, 36, 3, padding=1), nn.BatchNorm2d(36), nn.PReLU())
        self.pool1 = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), nn.BatchNorm2d(36), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(36, 48, 3, padding=1), nn.BatchNorm2d(48), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.PReLU())
        self.pool2 = nn.Sequential(nn.MaxPool2d(2, ceil_mode=True), nn.BatchNorm2d(48), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(48, 68, 3, padding=1), nn.BatchNorm2d(68), nn.PReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(68, 68, 3, padding=1), nn.BatchNorm2d(68), nn.PReLU())
        self.fc7 = nn.Sequential(nn.Linear(9 * 9 * 68, out_ch), nn.BatchNorm1d(out_ch), nn.PReLU())

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(self.pool1(c2))
        c4 = self.conv4(c3)
        c5 = self.conv5(self.pool2(c4))
        c6 = self.conv6(c5)
        f7 = self.fc7(c6.view(c6.size(0), -1))

        return f7


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.branch_3d = Branch3D(3, 300)
        self.branch_2d = Branch2D(2, 300)
        self.fc8 = nn.Sequential(nn.Linear(600, 35*35), nn.BatchNorm1d(35*35), nn.PReLU())

    def forward(self, x, y):
        f7_concat = torch.cat((self.branch_3d(x), self.branch_2d(y)), 1)
        out = torch.reshape(self.fc8(f7_concat), (-1, 1, 35, 35))

        return nn.Sigmoid()(out)
