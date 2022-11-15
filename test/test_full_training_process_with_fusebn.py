#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import linger
import numpy as np
import os


if not os.path.exists('data.ignore'):
    os.mkdir('data.ignore')

def test_full_training():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.transpose = nn.ConvTranspose2d(2, 2, 5, 5, 2, 4, 2, True, 2)
            self.conv1 = nn.Sequential(
                                    nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False, groups=2),
                                    nn.BatchNorm2d(2),
                                    nn.ReLU(),)
            self.conv2 = nn.Sequential(
                                    nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False, groups=2),
                                    nn.BatchNorm2d(2),
                                    nn.ReLU(),)
            self.fc = nn.Linear(392, 100)

        def forward(self, x):
            x = self.transpose(x)
            x = self.conv1(x)
            x = self.conv2(x)
            n, c, h, w = x.shape
            x = x.view(n, c*h*w)
            x = self.fc(x)
            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    dummy_input = torch.randn(1, 2, 2, 2).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)

    #normal training
    for i in range(1000):
        optimizer.zero_grad()
        out = net(dummy_input)
        loss = criterion(out, target)
        if i % 100 == 99:
            print('origin loss: ', loss)
        loss.backward()
        optimizer.step()
    net.eval()
    out_ori = net(dummy_input)
    torch.save(net.state_dict(), 'data.ignore/model.pt.ignore')

    #normal finetune
    net = Net().cuda()
    criterion = nn.MSELoss()
    replace_tuple=(nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    linger.trace_layers(net,net, dummy_input)
    net = linger.init(net, quant_modules=replace_tuple, mode=linger.QuantMode.QValue)
    net.train()
    net.load_state_dict(torch.load('data.ignore/model.pt.ignore', map_location='cpu'))
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
    for i in range(150):
        optimizer.zero_grad()
        out = net(dummy_input)
        loss = criterion(out, target)
        if i % 30 == 29:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()
    net.eval()
    out = net(dummy_input)
    # save int model
    torch.save(net.state_dict(), 'data.ignore/aa.pt')
    out1 = net(dummy_input)
    assert out.sum() == out1.sum()

    #normal testing
    net1 = Net().cuda()
    net1.train()
    linger.trace_layers(net1,net1, dummy_input)
    net1 = linger.init(net1, quant_modules=replace_tuple, mode=linger.QuantMode.QValue)
    net1.eval()
    net1.load_state_dict(torch.load('data.ignore/aa.pt', map_location='cpu'))
    out2 = net1(dummy_input)

    assert out1.sum() == out2.sum()
