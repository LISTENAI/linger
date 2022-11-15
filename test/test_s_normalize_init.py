#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from linger.utils import PlatFormQuant
import torch
import torch.nn as nn
import linger
import numpy as np

def test_convbn_normalize():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                         padding=1, bias=True)
            self.bn = nn.BatchNorm2d(10)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(1000, 100)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            n, c, h, w = x.shape
            x = x.view((n, c*h*w))  
            x = self.fc(x)
            return x
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    aa = torch.randn(1, 10, 10, 10).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
    loss = None
    for i in range(200):
        optimizer.zero_grad()
        out = net(aa)
        loss = criterion(out, target)
        if i % 20 == 0:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()
    # assert loss < 1e-12, 'training loss error'
    net.eval()
    torch.save(net.state_dict(), 'data.ignore/convbn_clamp.pt')
    out1 = net(aa)
    net.train()
    # unable the last fc clamp
    linger.disable_normalize(net.fc)
    linger.trace_layers(net, net, aa)
    normalize_modules = (nn.Conv2d, nn.Linear, nn.BatchNorm2d)
    net = linger.normalize_layers(net, normalize_modules=normalize_modules, normalize_weight_value=8, normalize_bias_value=8, normalize_output_value=8)
    net.load_state_dict(torch.load('data.ignore/convbn_clamp.pt'))
    net.cuda()
    out2 = net(aa)
    torch.save(net.state_dict(), 'data.ignore/convbn_quant.pt')
    linger.disable_quant(net.fc)
    linger.SetPlatFormQuant(platform_quant=PlatFormQuant.luna_quant)
    net = linger.init(net)
    net.load_state_dict(torch.load('data.ignore/convbn_quant.pt'))
    net.cuda()
    out3 = net(aa)
    assert abs(out3.mean() - 1) < 0.02
