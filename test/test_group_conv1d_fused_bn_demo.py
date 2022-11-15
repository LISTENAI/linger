#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from linger.utils import PlatFormQuant
import torch
import torch.nn as nn
import linger
import numpy as np
import os
if not os.path.exists('data.ignore'):
    os.mkdir('data.ignore')
def test_group_fuse_bn1d():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv1d(10, 10, kernel_size=3, stride=1,
                         padding=1, bias=True, groups=2)
            self.bn = nn.BatchNorm1d(10)
            self.fc = nn.Linear(10*50, 100)
    
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            n, c, l = x.shape
            x = x.view(n, c*l)
            x = self.fc(x)
            return x
    
    class Net1(nn.Module):
        def __init__(self):
            super(Net1, self).__init__()
            self.conv = nn.Conv1d(10, 10, kernel_size=3, stride=1,
                         padding=1, bias=True, groups=2)
            self.bn = nn.BatchNorm1d(10)
            self.fc = nn.Linear(10*50, 100)
    
        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            n, c, l = x.shape
            x = x.view(n, c*l)
            x = self.fc(x)
            return x
    
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    torch.cuda.set_device(0)
    aa = torch.randn(10, 10, 50).cuda()
    
    replace_tuple=(nn.Conv1d, nn.ConvTranspose2d, nn.Linear)
    net1 = Net().cuda()
    net1.eval()
    print(net1)
    net1(aa)
    torch.save(net1.state_dict(), 'data.ignore/model1d.pt.ignore')
    net1.train()
    out1 = net1(aa)
    
    net2 = Net1().cuda()
    
    linger.trace_layers(net2,net2, aa)
    linger.SetPlatFormQuant(platform_quant=PlatFormQuant.luna_quant)
    net3 = linger.init(net2, quant_modules=replace_tuple)
    #net3 = net2
    net3.load_state_dict(torch.load('data.ignore/model1d.pt.ignore'))
    out3 = net3(aa)
    
    assert out1.sum() - out3.sum() < 0.01
    
    