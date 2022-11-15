#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import linger
import numpy as np
import torch
import torch.nn as nn
from linger.utils import PlatFormQuant, QuantMode


def test_convbn1d_normalize():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv0 = nn.Conv1d(10, 10, kernel_size=3,
                                   stride=1, padding=1, bias=True)
            self.relu0 = nn.ReLU()
            self.conv = nn.Conv1d(10, 10, kernel_size=3, stride=1,
                                  padding=1, bias=True)
            self.bn = nn.BatchNorm1d(10)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(10*50, 100)

        def forward(self, x):
            x = self.conv0(x)
            x = self.relu0(x)
            x = self.conv(x)
            x = self.bn(x)
            # x = self.convbn(x)
            x = self.relu(x)
            n, c, l = x.shape
            x = x.view((n, c*l))
            x = self.fc(x)
            return x
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    # random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    aa = torch.randn(10, 10, 50).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss = None
    for i in range(1000):
        optimizer.zero_grad()
        out = net(aa)
        loss = criterion(out, target)
        if i % 20 == 0:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()
    # assert loss < 1e-12, 'training loss error'
    net.eval()
    torch.save(net.state_dict(), 'data.ignore/convbn_clamp1d.pt')
    out1 = net(aa)
    # print(out1)
    linger.trace_layers(net, net, aa)
    # linger.disable_normalize(net.conv0)
    net = linger.normalize_layers(net)
    net.train().cuda()
    net.load_state_dict(torch.load('data.ignore/convbn_clamp1d.pt'))
    out2 = net(aa)
    net.eval()
    out3 = net(aa)
    torch.save(net.state_dict(), 'data.ignore/convbn_quant1d.pt')
    linger.SetPlatFormQuant(platform_quant=PlatFormQuant.luna_quant)
    # linger.SetCastorBiasInt16(True)
    net = linger.init(net, mode=QuantMode.QValue)
    net.train()
    net.load_state_dict(torch.load('data.ignore/convbn_quant1d.pt'))
    net.cuda().train()
    out4 = net(aa)

    with torch.no_grad():
        torch.onnx.export(net, aa, "data.ignore/conv_bn_clamp1d.onnx", export_params=True,
                          opset_version=9, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    assert abs(out1.mean() - 1) < 0.01

