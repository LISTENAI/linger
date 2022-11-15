#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import linger
import numpy as np

def test_s_conv_linear_normalize():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 5, kernel_size=3, stride=1,
                         padding=1, bias=True)
            self.bn = nn.BatchNorm2d(5)
            self.conv1 = nn.Conv2d(10, 5, kernel_size=3, stride=1,
                         padding=1, bias=True)
            self.bn1 = nn.BatchNorm2d(5)
            self.fc = nn.Linear(1000, 100)

        def forward(self, x):
            x1 = self.conv(x)
            x1 = self.bn(x1)
            x2 = self.conv1(x)
            x2 = self.bn1(x2)
            x = torch.cat([x1, x2], dim=1)
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
    aa = torch.randn(1, 10, 10, 10).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    replace_tuple=(nn.Conv2d, nn.Linear, nn.AvgPool2d)
    net.train()
    # linger.disable_quant(net.fc)
    net = linger.init(net, quant_modules=replace_tuple, mode=linger.QuantMode.QValue)
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
    torch.save(net.state_dict(), 'data.ignore/conv_linear.pt')
    out1 = net(aa)
    #print(out1)
    with torch.no_grad():
        linger.onnx.export(net, aa,"data.ignore/conv_linear_iqcat.onnx",export_params=True,opset_version=11,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    assert abs(out1.mean() - 1) < 0.1
