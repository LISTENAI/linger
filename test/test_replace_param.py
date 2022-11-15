#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import linger
import numpy as np

def test_replace_param():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                         padding=1, bias=True)
            self.fc = nn.Linear(1000, 100)

        def forward(self, x):
            x = self.conv(x)
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
    net.train()
    print(net)
    for k, v in net.named_parameters():
        print(k, v.data_ptr(), v.abs().sum())
    criterion = nn.MSELoss()
    net = linger.normalize_layers(net)
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
    
    print(net)
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
    torch.save(net.state_dict(), 'data.ignore/conv_linear.pt')
    out1 = net(aa)
    for k, v in net.named_parameters():
        print(k, v.data_ptr(), v.abs().sum())
    