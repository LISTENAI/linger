#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import linger
import numpy as np

def test_conv_linear():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                         padding=1, bias=True)
            self.bn = nn.BatchNorm2d(10)
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                         padding=1, bias=True)
            self.conv2 = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                         padding=1, bias=True)  
            self.bn1 = nn.BatchNorm2d(10)
            self.relu1 = nn.ReLU()           
            self.fc = nn.Linear(1000, 100)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = x - 1
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.bn1(x)
            x = self.relu1(x)
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
    replace_tuple=(nn.Conv2d, nn.Linear)
    net.train()
    net = linger.normalize_layers(net, normalize_modules=replace_tuple, normalize_weight_value=4, normalize_bias_value=4, normalize_output_value=4)
    net.cuda()
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
        torch.onnx.export(net, aa,"data.ignore/conv_linear.onnx",export_params=True,opset_version=11,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    assert abs(out1.mean() - 1) < 0.01