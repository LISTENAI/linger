import linger
import numpy as np
import torch
import torch.nn as nn


def test_tf_quant():
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
    dummy_input = torch.randn(1, 2, 2, 2).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    replace_tuple = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    net1.train()

    net1 = linger.init(net1, quant_modules=replace_tuple,
                       mode=linger.QuantMode.QValue)

    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    # linger.SetTFQuant(luna_quant=True)
    optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.01)
    for i in range(150):
        optimizer1.zero_grad()
        out1 = net1(dummy_input)
        loss1 = criterion(out1, target)
        loss1.backward()
        optimizer1.step()
        if i % 30 == 29:
            print('loss1 {}'.format(loss1))
    net1.eval()
    out1 = net1(dummy_input)