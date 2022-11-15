import os

import linger
import numpy as np
import torch
import torch.nn as nn
from linger.utils import PlatFormQuant

if not os.path.exists('data.ignore'):
    os.mkdir('data.ignore')


def test_group_fuse_bn():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.transpose = nn.ConvTranspose2d(2, 2, 5, 5, 2, 4, 2, True, 2)
            self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=1,
                                  padding=1, bias=True, groups=2)
            self.bn = nn.BatchNorm2d(2)
            self.fc = nn.Linear(8, 100)

        def forward(self, x):
            x = self.transpose(x)
            x = self.conv(x)
            x = self.bn(x)
            n, c, h, w = x.shape
            x = x.view(n, c*h*w)
            x = self.fc(x)
            return x

    class Net1(nn.Module):
        def __init__(self):
            super(Net1, self).__init__()
            self.transpose = nn.ConvTranspose2d(2, 2, 5, 5, 2, 4, 2, True, 2)
            self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=1,
                                  padding=1, bias=True, groups=2)
            self.bn = nn.BatchNorm2d(2)
            self.fc = nn.Linear(8, 100)

        def forward(self, x):
            x = self.transpose(x)
            x = self.conv(x)
            x = self.bn(x)
            n, c, h, w = x.shape
            x = x.view(n, c*h*w)
            x = self.fc(x)
            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    torch.cuda.set_device(0)
    dummy_input = torch.randn(1, 2, 2, 2).cuda()

    target = torch.ones(1, 100).cuda()
    # criterion = nn.MSELoss()
    replace_tuple = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    net1 = Net().cuda()
    net1.train()
    net1(dummy_input)
    torch.save(net1.state_dict(), 'data.ignore/model.pt.ignore')
    net1.eval()
    out1 = net1(dummy_input)

    net2 = Net1().cuda()
    print(net2)
    linger.trace_layers(net2, net2, dummy_input)
    print(net2)
    linger.SetPlatFormQuant(platform_quant=PlatFormQuant.luna_quant)
    net3 = linger.init(net2, quant_modules=replace_tuple)
    net3.load_state_dict(torch.load('data.ignore/model.pt.ignore'))
    # net3.cuda()
    out3 = net3(dummy_input)
    print(out1.sum() - out3.sum())

    assert out1.sum() - out3.sum() < 1
