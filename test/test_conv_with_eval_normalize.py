import linger
import numpy as np
import torch
import torch.nn as nn


def test_clip():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.transpose = nn.ConvTranspose2d(2, 2, 5, 5, 2, 4, 2, True, 2)
            self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=1,
                                  padding=1, bias=True)
            self.bn = nn.BatchNorm2d(2)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(392, 100, bias=True)

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
    net2 = Net().cuda()
    dummy_input = torch.randn(1, 2, 2, 2).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    replace_tuple = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    net2.train()
    net2 = linger.init(net2, quant_modules=replace_tuple,
                       mode=linger.QuantMode.QValue)

    optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.01)
    for i in range(150):
        optimizer2.zero_grad()
        out2 = net2(dummy_input)
        loss2 = criterion(out2, target)
        loss2.backward()
        optimizer2.step()
        if i % 30 == 29:
            print('loss2 {}'.format(loss2))
    torch.save(net2.state_dict(), 'data.ignore/eval_normalize.pt')
    net2.eval()
    out2 = net2(dummy_input)
    assert abs(out2.mean() - target.mean()) < 0.01
    # reset avoid other test files params be changed
