#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import linger
import numpy as np
import os


if not os.path.exists('data.ignore'):
    os.mkdir('data.ignore')

def test_trace_convtranspose_net():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1, bias=False, groups=2)
            self.bn = nn.BatchNorm2d(10)
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1, bias=False, groups=2)
            self.bn1 = nn.BatchNorm2d(10)
            self.relu1 = nn.ReLU()
            self.deconv = nn.ConvTranspose2d(10,20, kernel_size=2, stride=2)
            self.bn2 = nn.BatchNorm2d(20)
            self.relu2 = nn.ReLU()
            self.fc = nn.Linear(20*100*100, 100)
            # self.fc = nn.Linear(10*50*50, 100)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.deconv(x)
            x = self.bn2(x)
            x = self.relu2(x)
            n, c, h, w = x.shape
            x = x.view((n, c*h*w))
            x = self.fc(x)
            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    aa = torch.randn(10, 10, 50, 50).cuda()
    target = torch.ones(10, 100).cuda()
    criterion = nn.MSELoss()
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)

    net = Net().cuda()
    criterion = nn.MSELoss()
    replace_tuple=(nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d, linger.NormalizeConvBN2d, linger.NormalizeConvTransposeBN2d)
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    # torchintx.FuseBNIntoConv(net, aa)
    # import pdb; pdb.set_trace()
    # torchintx.trace_layers(net,net, aa, fuse_bn=True)
    # import pdb; pdb.set_trace()
    # net = torchintx.init(net, quant_modules=replace_tuple, data_bits=8, parameter_bits=8, out_bits=8, mode=torchintx.QuantMode.MaxValue)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
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

    torch.save(net.state_dict(), 'data.ignore/aa_convtranspose.pt')
    net1 = Net().cuda()
    linger.trace_layers(net1, net1, aa, fuse_bn=True)
    net1.eval()
    print(net1)
    net1.load_state_dict(torch.load('data.ignore/aa_convtranspose.pt'))
    out1 = net1(aa)
    # import pdb; pdb.set_trace()
    # with torch.no_grad():
    #     torch.onnx.export(net, aa,"data.ignore/convtranspose_fuse_bn.onnx",export_params=True,opset_version=11,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    # out2 = net(aa)
    # import pdb; pdb.set_trace()
    assert (out.mean() - out1.mean()).abs() < 0.001, print('out1: {}, out2: {}'.format(out.sum(), out1.sum()))
    # assert out.abs().sum() == out2.abs().sum(), 'inconsistant for tarce convtranspose'


def test_quant_convtransposebn_net():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 8, kernel_size=3, stride=1, padding=1, bias=False, groups=2)
            self.bn = nn.BatchNorm2d(8)
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False, groups=2)
            self.bn1 = nn.BatchNorm2d(8)
            self.relu1 = nn.ReLU()
            self.deconv = nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2)
            self.bn2 = nn.BatchNorm2d(1)
            self.relu2 = nn.ReLU()
            self.fc = nn.Linear(100*100, 100)
            # self.fc = nn.Linear(10*50*50, 100)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.deconv(x)
            x = self.bn2(x)
            x = self.relu2(x)
            n, c, h, w = x.shape
            x = x.view((n, c*h*w))
            x = self.fc(x)
            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    aa = torch.randn(1, 10, 50, 50).cuda()
    target = torch.ones(10, 100).cuda()
    criterion = nn.MSELoss()
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)

    net = Net().cuda()
    criterion = nn.MSELoss()
    replace_tuple=(nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d, linger.NormalizeConvBN2d, linger.NormalizeConvTransposeBN2d)
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    # torchintx.FuseBNIntoConv(net, aa)
    # import pdb; pdb.set_trace()
    # torchintx.trace_layers(net,net, aa, fuse_bn=True)
    # import pdb; pdb.set_trace()
    # net = torchintx.init(net, quant_modules=replace_tuple, data_bits=8, parameter_bits=8, out_bits=8, mode=torchintx.QuantMode.MaxValue)
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
    for i in range(150):
        optimizer.zero_grad()
        out = net(aa)
        loss = criterion(out, target)
        if i % 30 == 29:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()

    # net.eval()
    out = net(aa)

    torch.save(net.state_dict(), 'data.ignore/aa_convtranspose_float.pt')
    net1 = Net().cuda()
    linger.trace_layers(net1,net1, aa, fuse_bn=True)
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    # torchintx.disable_quant(net1.fc)
    net1 = linger.init(net1, quant_modules=replace_tuple,
                      mode=linger.QuantMode.QValue)
    print(net1)
    net1.load_state_dict(torch.load('data.ignore/aa_convtranspose_float.pt'))
    net1.train()
    out1 = net1(aa)
    assert ((out.abs().mean() - out1.abs().mean()).abs() < 0.1)
    # import pdb; pdb.set_trace()
    optimizer1 = torch.optim.SGD(net1.parameters(), lr = 0.001)
    for i in range(150):
        optimizer1.zero_grad()
        out = net1(aa)
        loss = criterion(out, target)
        if i % 30 == 29:
            print('loss: ', loss)
        loss.backward()
        optimizer1.step()
    net1.eval()
    out2 = net1(aa)
    # import pdb; pdb.set_trace()
    with torch.no_grad():
        torch.onnx.export(net1, aa,"data.ignore/convtranspose_fuse_bn.onnx",export_params=True,opset_version=11,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    out3 = net1(aa)
    # import pdb; pdb.set_trace()
    assert out2.abs().sum() == out3.abs().sum()
    # import pdb; pdb.set_trace()
    assert ((out2.abs().mean() - out1.abs().mean()).abs() < 0.1)