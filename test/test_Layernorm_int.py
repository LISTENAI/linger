import os

import linger
import numpy as np
import torch
import torch.nn as nn

if not os.path.exists('data.ignore'):
    os.mkdir('data.ignore')


def notest_layernorm_int_net():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 3, kernel_size=2,
                                  stride=1, padding=1, bias=False, groups=1)
            self.ln = nn.LayerNorm([4, 4])
            self.relu = nn.ReLU()
            self.conv1 = nn.Conv2d(3, 1, kernel_size=2,
                                   stride=1, padding=1, bias=False, groups=1)
            self.ln1 = nn.LayerNorm([5, 5])
            self.relu1 = nn.ReLU()
            self.fc = nn.Linear(25, 100)

        def forward(self, x):
            x = self.conv(x)
            x = self.ln(x)
            x = self.relu(x)
            x = self.conv1(x)
            x = self.ln1(x)
            x = self.relu1(x)
            n, c, h, w = x.shape
            x = x.view((n, c*h*w))
            # print(x.shape)
            x = self.fc(x)
            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    net = Net()#.cuda()
    dummy_input = torch.randn(1, 1, 3, 3)#.cuda()
    target = torch.ones(1, 100)#.cuda()
    criterion = nn.MSELoss()
    net.train()
    replace_tuple = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                     nn.BatchNorm2d, linger.NormalizeConvBN2d, nn.LayerNorm)
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    # # linger.FuseBNIntoConv(net, dummy_input)
    linger.trace_layers(net, net, dummy_input, fuse_bn=False)
    net = linger.init(net, quant_modules=replace_tuple,
                      mode=linger.QuantMode.QValue)
    print(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # model.to(device)
    for i in range(200):
        optimizer.zero_grad()
        out = net(dummy_input)
        loss = criterion(out, target)
        if i % 20 == 0:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()
    net.eval()
    torch.save(net.state_dict(), 'data.ignore/aa.pt')
    out1 = net(dummy_input)
    with torch.no_grad():
        torch.onnx.export(net, dummy_input, "data.ignore/layernormInt.onnx", export_params=True,
                          opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    out2 = net(dummy_input)
    assert (out1.mean() - out1.mean()
            ) < 0.001, print('out1: {}, out2: {}'.format(out1.sum(), out1.sum()))
    assert out1.abs().sum() == out2.abs().sum(), 'inconsistant for batchnormint'


def notest_monolayernorm():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.ln = nn.LayerNorm([28, 28])

        def forward(self, x):
            x = self.ln(x)
            return x

    dummy_input = torch.randn(10, 1, 28, 28).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net()
    ln_lg = Net()
    weight = torch.nn.Parameter(torch.empty(28, 28))
    nn.init.normal_(weight)
    bias = torch.nn.Parameter(torch.empty(28, 28))
    nn.init.normal_(bias)
    replace_tuple = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                     nn.BatchNorm2d, linger.NormalizeConvBN2d, nn.LayerNorm)
    ln_lg = linger.init(ln_lg, quant_modules=replace_tuple,
                        mode=linger.QuantMode.QValue)
    ln_lg.ln.weight = weight
    ln_lg.ln.bias = bias
    net.ln.weight = weight
    net.ln.bias = bias

    ln_lg.train()
    net.train()
    net = net.to(device)
    ln_lg = ln_lg.to(device)

    for _ in range(300):

        out = ln_lg(dummy_input)
        out1 = net(dummy_input)

    out = ln_lg(dummy_input)
    out1 = net(dummy_input)
    ln_lg.eval()
    net.eval()
    assert (out1.mean() - out.mean()
            ) < 0.001, print('out1: {}, out: {}'.format(out1.sum(), out.sum()))

    out2 = ln_lg(dummy_input)
    out3 = net(dummy_input)
    with torch.no_grad():
        torch.onnx.export(ln_lg, dummy_input, "data.ignore/layernorm.onnx", export_params=True,
                          opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    out4 = ln_lg(dummy_input)

    assert (out2.mean() - out3.mean()
            ) < 0.001, print('out1: {}, out2: {}'.format(out2.sum(), out3.sum()))

    assert out4.abs().sum() == out2.abs().sum(), 'inconsistant for layernormint'


notest_layernorm_int_net()
