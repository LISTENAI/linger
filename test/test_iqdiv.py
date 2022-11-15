#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import linger
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                              padding=1, bias=True)
        self.bn = nn.BatchNorm2d(10)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(10)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(10)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(1000, 100)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x1 = self.relu(x)

        x = self.conv1(input)
        x2 = self.bn1(x)
        x2 = self.relu(x2)
        # y = linger.from_torch_tensor(torch.tensor(1),0.5,8).cuda()
        x2 = x2 + 1
        # x2 = self.relu1(x)
        x = x1 / 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        n, c, h, w = x.shape
        x = x.view((n, c*h*w))
        x = self.fc(x)
        return x


def test_conv_linear():

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    # random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    aa = torch.randn(1, 10, 10, 10).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    replace_tuple = (nn.Conv2d, nn.Linear, nn.BatchNorm2d)
    net.train()
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    linger.FuseConvBNAheadRelu(
        net, aa, fused_bn=False, ahead_bn_relu=True, ahead_conv_relu=True)
    net = linger.init(net, quant_modules=replace_tuple,
                      mode=linger.QuantMode.QValue)
    print(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
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
    torch.save(net.state_dict(), 'data.ignore/conv_iqdiv.pt')
    print(net)
    out1 = net(aa)
    # print(out1)
    with torch.no_grad():
        torch.onnx.export(net, aa, "data.ignore/conv_iqdiv1.onnx", export_params=True,
                          opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    assert abs(out1.mean() - 1) < 0.1
