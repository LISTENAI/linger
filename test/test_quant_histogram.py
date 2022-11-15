import linger
import numpy as np
import torch
import torch.nn as nn
from linger.utils import PlatFormQuant


def test_histogram():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.transpose = nn.ConvTranspose2d(2, 2, 5, 5, 2, 4, 2, True, 2)
            self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=1,
                                  padding=1, bias=True)
            self.fc = nn.Linear(392, 100, bias=False)

        def forward(self, x):
            x = self.transpose(x)
            x = self.conv(x)
            n, c, h, w = x.shape
            x = x.view(n, c*h*w)
            x = self.fc(x)
            return x
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    torch.cuda.set_device(0)
    net1 = Net().cuda()
    net2 = Net().cuda()
    aa = torch.randn(1, 2, 2, 2).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    replace_tuple = (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)
    net1.train()
    net2.train()
    linger.SetPlatFormQuant(platform_quant=PlatFormQuant.luna_quant)
    net1 = linger.init(net1, quant_modules=replace_tuple,
                       mode=linger.QuantMode.QValue)

    net2 = linger.init(net2, quant_modules=replace_tuple,
                       mode=linger.QuantMode.QValue)
    for v1, v2 in zip(net1.parameters(), net2.parameters()):
        v2.data.copy_(v1.data)
    optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.01)
    optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.01)
    for i in range(150):
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        out1 = net1(aa)
        out2 = net2(aa)
        loss1 = criterion(out1, target)
        loss1.backward()
        optimizer1.step()
        loss2 = criterion(out2, target)
        loss2.backward()
        optimizer2.step()
        if i % 30 == 29:
            print('loss1 {}, loss2 {}'.format(loss1, loss2))
    net1.eval()
    net2.eval()
    out1 = net1(aa)
    out2 = net2(aa)

    assert criterion(out1, out2) < 0.02
