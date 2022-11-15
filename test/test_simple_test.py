#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from os import stat
import linger
import torch
import torch.nn as nn


def test_trace_iqtensor():

    class Add(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x, y):
            return x + y

    class iqtensorAdd(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.add = Add()

        def forward(self, x, y):
            x = linger.quant_tensor(self, x, name='x')
            y = linger.quant_tensor(self, y, name='y')
            z = self.add(x, y)
            return z
    net = iqtensorAdd()
    dummy_input = torch.randn(10, 10)
    bb = torch.randn(10, 10)
    linger.trace_layers(net, net.add, (dummy_input, bb))
    cc = net(dummy_input, bb)
    with torch.no_grad():
        torch.onnx.export(net, (dummy_input, bb), "data.ignore/add.onnx", export_params=True, opset_version=12,
                          verbose=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)


def test_iqview_onnx():
    class IQView(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 2, 1, 1)
            self.fc = nn.Linear(18, 100)

        def forward(self, x):
            x = self.conv(x)
            n, c, h, w = x.shape
            x = x.view((n, c*h*w))
            x = self.fc(x)
            return x
    dummy_input = torch.randn(1, 2, 2, 2)
    net = IQView()
    out = net(dummy_input)

    net = linger.init(net)

    out = net(dummy_input)

    with torch.no_grad():
        torch.onnx.export(net, (dummy_input,), "data.ignore/iqview.onnx", export_params=True, opset_version=12,
                          verbose=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
