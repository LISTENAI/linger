import os

import linger
import numpy as np
import torch
import torch.nn as nn

if not os.path.exists('data.ignore'):
    os.mkdir('data.ignore')

def test_wb_analyse():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=1,
                                  padding=1, bias=True)
            self.conv1 = nn.Conv2d(2, 2, kernel_size=3, stride=1,
                                   padding=1, bias=True)
            self.fc = nn.Linear(8, 100)

        def forward(self, x):

            x = self.conv(x)
            x = self.conv1(x)
            n, c, h, w = x.shape
            x = x.view((n, c*h*w))
            x = self.fc(x)

            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    # random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    dummy_input = torch.ones(1, 2, 2, 2).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    replace_tuple = (nn.Conv2d, nn.Linear, nn.AvgPool2d)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss = None
    for i in range(200):
        optimizer.zero_grad()
        out = net(dummy_input)
        loss = criterion(out, target)
        if i % 20 == 0:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()
    assert loss < 1, 'training loss error'
    net.eval()
    torch.save(net.state_dict(), 'data.ignore/tool_test.pt')
    linger.wb_analyse('data.ignore/tool_test.pt', 'data.ignore/wb_anylse.log')
