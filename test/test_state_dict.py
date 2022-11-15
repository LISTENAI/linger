import os

import linger
import numpy as np
import torch
import torch.nn as nn
from linger.utils import PlatFormQuant

if not os.path.exists('data.ignore'):
    os.mkdir('data.ignore')

def test_state_dict():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.transpose = nn.ConvTranspose2d(2, 2, 5, 5, 2, 4, 2, True, 2)
            self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=1,
                                  padding=1, bias=True)
            self.bn = nn.BatchNorm2d(2)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(392, 100)

        def forward(self, x):
            x = self.transpose(x)
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            n, c, h, w = x.shape
            x = x.view(n, c*h*w)
            x = self.fc(x)
            return x
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    aa = torch.randn(1, 2, 2, 2).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    replace_tuple = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    net.train()
    linger.SetPlatFormQuant(platform_quant=PlatFormQuant.luna_quant)
    net = linger.init(net, quant_modules=replace_tuple,
                      mode=linger.QuantMode.QValue)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss = None
    for i in range(150):
        optimizer.zero_grad()
        out = net(aa)
        loss = criterion(out, target)
        if i % 30 == 29:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()
    assert loss < 1e-2, 'training loss error'
    net.eval()
    torch.save(net.state_dict(), 'data.ignore/aa.pt')
    out1 = net(aa)
    net1 = Net().cuda()
    net1.train()
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)

    net1 = linger.init(net1, quant_modules=replace_tuple,
                       mode=linger.QuantMode.QValue)
    net1.eval()
    net1.load_state_dict(torch.load('data.ignore/aa.pt', map_location='cpu'))
    out2 = net1(aa)

    assert out1.sum() == out2.sum()
