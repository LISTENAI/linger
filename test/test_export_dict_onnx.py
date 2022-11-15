import os

import linger
import numpy as np
import torch
import torch.nn as nn

if not os.path.exists('data.ignore'):
    os.mkdir('data.ignore')


def test_batchnorm_int_net():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(2, 2, kernel_size=2,
                                  stride=1, padding=1, bias=False, groups=2)
            self.bn = nn.BatchNorm2d(2)
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(2, 2, kernel_size=2,
                                   stride=1, padding=1, bias=False, groups=2)
            self.bn1 = nn.BatchNorm2d(2)
            self.relu1 = nn.ReLU()
            self.fc = nn.Linear(32, 100)

        def forward(self, x: dict):
            x = x['input']
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            n, c, h, w = x.shape
            x = x.view((n, c*h*w))
            x = self.fc(x)
            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    bb = torch.randn(1, 2, 2, 2).cuda()
    aa = {}
    aa['input'] = bb
    aa['input1'] = bb

    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    net = Net().cuda()
    criterion = nn.MSELoss()
    replace_tuple = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                     nn.BatchNorm2d, linger.NormalizeConvBN2d)
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    # linger.FuseBNIntoConv(net, aa)
    linger.trace_layers(net, net, aa, fuse_bn=False)
    net = linger.init(net, quant_modules=replace_tuple,
                      mode=linger.QuantMode.QValue)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    for i in range(150):
        optimizer.zero_grad()
        out = net(aa)
        loss = criterion(out, target)
        if i % 30 == 29:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()

    net.eval()
    out = net(aa)

    torch.save(net.state_dict(), 'data.ignore/aa.pt')
    out1 = net(aa)
    with torch.no_grad():
        torch.onnx.export(net, aa, "data.ignore/batchnormInt.onnx", export_params=True,
                          opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    out2 = net(aa)
    assert (out.mean() - out1.mean()
            ) < 0.001, print('out1: {}, out2: {}'.format(out.sum(), out1.sum()))
    assert out.abs().sum() == out2.abs().sum(), 'inconsistant for batchnormint'
