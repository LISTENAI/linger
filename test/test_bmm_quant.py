#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import linger
import numpy
import torch
import torch.nn as nn


def test_bmmint_net():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear1 = nn.Linear(12, 12)
            self.linear2 = nn.Linear(12, 12)
            self.linear3 = nn.Linear(16, 16)

        @torch.no_grad()
        def forward(self, input):

            x = self.linear1(input)

            y = self.linear2(input)

            x = x.view(8, 4, 3)

            y = y.view(8, 3, 4)
            x = torch.bmm(x, y)

            x = x.view(8, 16)
            x = self.linear3(x)

            return x
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    numpy.random.seed(1)
    # aa = torch.LongTensor([[1,2,4,5],[4,3,2,9]]).cuda()
    aa = torch.randn((8, 12), requires_grad=True).cuda()
    replace_tuple = (nn.Linear)

    net = Net().cuda()

    linger.SetFunctionBmmQuant(True)  # default  false

    net = linger.init(net, quant_modules=replace_tuple, mode=linger.QuantMode.QValue)
    criterion = nn.MSELoss()
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss = None
    label = torch.ones((8, 16)).cuda()
    for i in range(100):
        optimizer.zero_grad()
        out = net(aa)

    with torch.no_grad():
        net.eval()
        torch.onnx.export(net, (aa), "data.ignore/bmm_int.onnx", export_params=True, opset_version=12,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    linger.SetFunctionBmmQuant(False)  # set default
