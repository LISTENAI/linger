#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import linger
import numpy as np

def test_s_relu6_int():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                         padding=1, bias=True)
            self.bn = nn.BatchNorm2d(10)
            self.relu = nn.ReLU6()
            self.conv1 = nn.Conv2d(10, 10, kernel_size=1, stride=1,
                         padding=0, bias=True)
            self.fc = nn.Linear(1000, 100)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            # x = x+8
            x = self.relu(x)
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
    aa = torch.randn(1, 10, 10, 10).cuda()
    # import pickle as pk
    # with open('input.pb', 'wb') as file:
    #     pk.dump(aa, file)
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    # replace_tuple=(nn.Conv2d, nn.Linear, nn.AvgPool2d, nn.ReLU6)
    replace_tuple=(nn.ReLU6)
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
    torch.save(net.state_dict(), 'data.ignore/relu6_int.pt')
    net.load_state_dict(torch.load('data.ignore/relu6_int.pt'))
    out1 = net(aa)
    # print(out1)
    with torch.no_grad():
        torch.onnx.export(net, aa,"data.ignore/test_relu6_int.onnx",export_params=True,opset_version=12,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    assert abs(out1.mean() - 1) < 0.01
