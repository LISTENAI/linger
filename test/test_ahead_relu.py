#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import linger
import numpy as np
import torch
import torch.nn as nn
from linger.ops.ops_names import LINGER_AHEAD_RELU


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                              padding=1, bias=True)
        self.bn = nn.BatchNorm2d(10)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(10)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(1000, 100)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x - 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn1(x)
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
    replace_tuple = (nn.Conv2d, nn.Linear)
    net.train()
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    linger.FuseConvBNAheadRelu(
        net, aa, fused_bn=False, ahead_bn_relu=True, ahead_conv_relu=True)
    net = linger.init(net, quant_modules=replace_tuple, mode=linger.QuantMode.QValue)
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
    torch.save(net.state_dict(), 'data.ignore/conv_linear.pt')
    out1 = net(aa)
    # print(out1)
    with torch.no_grad():
        torch.onnx.export(net, aa, "data.ignore/conv_linear.onnx", export_params=True,
                          opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    assert abs(out1.mean() - 1) < 0.01


def test_ahead_relu_conv_attr():
    model = Net()
    aa = torch.randn(1, 10, 10, 10)
    linger.trace_layers(model, model, aa)
    assert model.conv2.ahead_relu
    assert model.conv.ahead_relu

    model = Net()
    linger.trace_layers(model, model, aa, ahead_bn_relu=False)
    assert hasattr(model.conv1, linger.LINGER_AHEAD_RELU)
    assert getattr(model.conv1, linger.LINGER_AHEAD_RELU, False)

    model = Net()
    linger.trace_layers(model, model, aa, ahead_conv_relu=False)
    assert model.conv.ahead_relu
    assert model.conv2.ahead_relu

    model = Net()
    linger.trace_layers(
        model, model, aa, ahead_conv_relu=False, ahead_bn_relu=False)
    assert not model.conv.ahead_relu
    assert not model.conv2.ahead_relu

    model = Net()
    linger.trace_layers(model, model, aa, fuse_bn=False)

    assert hasattr(model.bn, LINGER_AHEAD_RELU)
    assert getattr(model.bn, LINGER_AHEAD_RELU, True)
    assert getattr(model.conv1, LINGER_AHEAD_RELU, False)