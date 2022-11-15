#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import linger
import numpy as np
import torch
import torch.nn as nn


def test_conv_dropout_linear():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                                  padding=1, bias=True)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(100, 100)

        def forward(self, x):
            x = self.conv(x)
            x = x.squeeze()
            x = x.permute(0, 1, 2)  # permute only support len(x.shape)<=3
            x = self.dropout(x)
            c, h, w = x.shape
            x = x.reshape((c, h*w))
            x = self.fc(x)
            return x
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    # random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    dummy_input = torch.randn(1, 10, 10, 10).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    replace_tuple = (nn.Conv2d, nn.Linear, nn.AvgPool2d)
    net = linger.init(net)
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
    # assert loss < 1e-12, 'training loss error'
    net.eval()
    torch.save(net.state_dict(), 'data.ignore/conv_permute_linear.pt')
    out1 = net(dummy_input)
    # print(out1)
    with torch.no_grad():
        torch.onnx.export(net, dummy_input, "data.ignore/conv_permute_linear.onnx", export_params=True,
                          opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    assert abs(out1.mean() - 1) < 1