import math

import linger
import torch
import torch.nn as nn
from linger.ops import *

torch.set_printoptions(linewidth=28*10)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        assert isinstance(x, IQTensor)
        s0 = x.scale_data
        x = self.relu1(x)
        assert isinstance(x, IQTensor)
        s1 = x.scale_data
        assert s0 == s1
        x = self.pool1(x)
        assert isinstance(x, IQTensor)
        s2 = x.scale_data
        assert s0 == s2
        x = self.conv2(x)
        assert isinstance(x, IQTensor)
        s0 = x.scale_data
        x = self.relu2(x)
        assert isinstance(x, IQTensor)
        s1 = x.scale_data
        assert s0 == s1
        x = self.pool2(x)
        assert isinstance(x, IQTensor)
        s2 = x.scale_data
        assert s0 == s2
        x = x.view(x.size(0), -1)
        assert isinstance(x, IQTensor)
        s3 = x.scale_data
        assert s0 == s3
        output = self.out(x)
        assert isinstance(output, IQTensor)
        s4 = output.scale_data
        return output


def test_convint_base():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    net = Conv2dInt(in_channels=1, out_channels=16,
                    kernel_size=5, stride=1, padding=2,)
    net.weight.data.fill_(1.0)
    net.bias.data.fill_(0)
    t = torch.ones(1, 1, 28, 28)
    r = net(t)
    assert hasattr(r, 'scale_data')
    # assert r.sum().item() == 287296
    iq = torch.ones(1, 1, 28, 28)*(-0.1)
    iq = from_torch_tensor(iq, 0.5, 8)
    riq = net(iq)
    assert hasattr(riq, 'scale_data')
    assert riq.sum().item() == 0
    iq2 = torch.ones(1, 1, 28, 28)
    iq2 = from_torch_tensor(iq2, 0.6, 8)
    riq2 = net(iq2)
    assert hasattr(riq2, 'scale_data')
    # assert abs(riq2.sum() - 478827) < 1


def test_linearint_base():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    net = LinearInt(64, 16)
    net.weight.data.fill_(1.0)
    net.bias.data.fill_(0)
    t = torch.ones(64, 64)
    r = net(t)
    assert not hasattr(r, 'scale_data')
    iq = torch.ones(64, 64) * (-0.1)
    iq = from_torch_tensor(iq, 0.5, 8)
    riq = net(iq)
    assert not hasattr(riq, 'scale_data')
    assert riq.sum().item() == 0

    iq2 = torch.ones(64, 64)
    iq2 = from_torch_tensor(iq2, 0.6, 8)
    riq2 = net(iq2)
    assert not hasattr(riq2, 'scale_data')
    # assert abs(riq2.sum().item() - 109226) < 1

def test_linearint_base_obits():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    net = LinearInt(64, 16, o_bits=8)
    net.weight.data.fill_(1.0)
    net.bias.data.fill_(0)
    t = torch.ones(64, 64)
    r = net(t)
    assert hasattr(r, 'scale_data')
    iq = torch.ones(64, 64)*(-0.1)
    iq = from_torch_tensor(iq, 0.5, 8)
    riq = net(iq)
    assert hasattr(riq, 'scale_data')
    assert riq.sum().item() == 0

    iq2 = torch.ones(64, 64)
    iq2 = from_torch_tensor(iq2, 0.6, 8)
    riq2 = net(iq2)
    assert hasattr(riq2, 'scale_data')
    # assert  abs(riq2.sum().item() - 109226 ) < 1

    net.weight.data.fill_(0.1)
    iq3 = torch.ones(64, 64)
    iq3 = from_torch_tensor(iq3, 0.6, 8)
    riq3 = net(iq3)
    assert hasattr(riq3, 'scale_data')
    # assert abs (riq3.sum().item() - 10922)< 1


def test_convint_out():
    net = CNN().cuda()
    net = linger.init(net)
    aa = torch.randn(1, 1, 28, 28).cuda()
    net(aa)


def test_view():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    a = torch.randn((1, 2, 3, 4), requires_grad=True)
    b = a.detach().clone()
    b.requires_grad_()
    c = a.detach().clone()
    c.requires_grad_()
    x = a.view((4, 3, 2, 1))
    x.sum().backward()

    b_iq = from_torch_tensor(b, 120, 8)

    y = b_iq.view(4, 3, 2, 1)
    assert isinstance(y, IQTensor)
    y.sum().backward()

    assert (b.grad - a.grad).sum() == 0
    assert b.grad.size() == a.grad.size()

    c_iq = from_torch_tensor(c, 120, 8)
    z = c_iq.view((4, 3, 2, 1))
    z.sum().backward()
    assert (c.grad - a.grad).sum() == 0
    assert c.grad.size() == a.grad.size()


def test_view_as():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    a = torch.randn((1, 2, 3, 4), requires_grad=True)
    a_target = torch.randn((3, 2, 1, 4), requires_grad=True)
    b = a.detach().clone()
    b.requires_grad_()
    c = a.detach().clone()
    c.requires_grad_()
    x = a.view_as(a_target)
    x.sum().backward()

    b_iq = from_torch_tensor(b, 120, 8)

    y = b_iq.view_as(a_target)
    assert isinstance(y, IQTensor)
    y.sum().backward()
    assert (b.grad - a.grad).sum() == 0
    assert b.grad.size() == a.grad.size()


def test_transposeconv2dint_base():
    my_convtranspose = ConvTranspose2dInt(50, 50, 5, 5, 2, 4, 2, False, 2)
    input1 = torch.ones(50, 50, 50, 50)
    out = my_convtranspose(input1)
    assert type(out) == torch.Tensor


def test_transposeconv2dint_obit():
    my_convtranspose = ConvTranspose2dInt(
        50, 50, 5, 5, 2, 4, 2, False, 2, o_bits=8)
    input1 = torch.ones(50, 50, 50, 50)
    out = my_convtranspose(input1)
    assert type(out) == IQTensor
    assert out.scale_data > 0
    assert out.bits == 8


def test_transposeconv2dint_iq_obit():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    my_convtranspose = ConvTranspose2dInt(
        50, 50, 5, 5, 2, 4, 2, False, 2, o_bits=8)
    input1 = torch.ones(50, 50, 50, 50)
    input_iq = from_torch_tensor(input1, 127/1.0, 8)
    out = my_convtranspose(input_iq)
    assert type(out) == IQTensor
    assert out.scale_data > 0
    assert out.bits == 8
    assert abs(my_convtranspose.running_x-1.0) < 0.001


def test_reshape():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    a = torch.randn((1, 2, 3, 4), requires_grad=True)
    b = a.detach().clone()
    b.requires_grad_()
    c = a.detach().clone()
    c.requires_grad_()
    d = a.detach().clone()
    d.requires_grad_()

    x = a.reshape((4, 3, 2, 1))
    x.sum().backward()

    b_iq = from_torch_tensor(b, 120, 8)

    y = b_iq.reshape(4, 3, 2, 1)
    assert isinstance(y, IQTensor)
    assert y.size() == (4, 3, 2, 1)
    y.sum().backward()
    assert b_iq.grad is None
    assert (b.grad - a.grad).sum() == 0
    assert b.grad.size() == a.grad.size()

    c_iq = from_torch_tensor(c, 120, 8)
    z = c_iq.reshape((4, 3, 2, 1))
    assert isinstance(z, IQTensor)
    assert z.size() == (4, 3, 2, 1)
    z.sum().backward()
    assert c_iq.grad is None
    assert (c.grad - a.grad).sum() == 0
    assert c.grad.size() == a.grad.size()

    d_iq = from_torch_tensor(d, 120, 8)
    z = d_iq.reshape((4, -1, 2, 1))
    assert isinstance(z, IQTensor)
    assert z.size() == (4, 3, 2, 1)
    z.sum().backward()
    assert d_iq.grad is None
    assert (d.grad - a.grad).sum() == 0
    assert d.grad.size() == a.grad.size()


def test_reshape_as():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    a = torch.randn((1, 2, 3, 4), requires_grad=True)
    shape_tensor = torch.randn((1, 3, 2, 4), requires_grad=True)
    b = a.detach().clone()
    b.requires_grad_()
    c = a.detach().clone()
    c.requires_grad_()
    d = a.detach().clone()
    d.requires_grad_()

    x = a.reshape_as(shape_tensor)
    x.sum().backward()

    b_iq = from_torch_tensor(b, 120, 8)

    y = b_iq.reshape_as(shape_tensor)
    assert isinstance(y, IQTensor)
    assert y.size() == shape_tensor.size()
    y.sum().backward()
    assert b_iq.grad is None
    assert (b.grad - a.grad).sum() == 0
    assert b.grad.size() == a.grad.size()


def test_squeeze():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    a = torch.randn((1, 2, 1, 3, 1, 5), requires_grad=True)
    b = a.detach().clone()
    b.requires_grad_()
    a1 = a.detach().clone()
    a1.requires_grad_()
    d = a.detach().clone()
    d.requires_grad_()

    x = a.squeeze()
    x.sum().backward()

    b_iq = from_torch_tensor(b, 120, 8)

    y = b_iq.squeeze()
    assert isinstance(y, IQTensor)
    assert y.size() == (2, 3, 5)
    y.sum().backward()
    assert b_iq.grad is None
    assert (b.grad - a.grad).sum() == 0
    assert b.grad.size() == a.grad.size()

    x1 = a1.squeeze(2)
    x1.sum().backward()

    c_iq = from_torch_tensor(d, 120, 8)
    z = c_iq.squeeze(2)
    assert isinstance(z, IQTensor)
    assert z.size() == (1, 2, 3, 1, 5)
    z.sum().backward()
    assert c_iq.grad is None
    assert (d.grad - a1.grad).sum() == 0
    assert d.grad.size() == a1.grad.size()


def test_unsqueeze():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    a = torch.randn((2, 3, 1, 5), requires_grad=True)
    b = a.detach().clone()
    b.requires_grad_()

    x = a.unsqueeze(0)
    x.sum().backward()

    b_iq = from_torch_tensor(b, 120, 8)
    y = b_iq.unsqueeze(0)
    assert isinstance(y, IQTensor)
    assert y.size() == (1, 2, 3, 1, 5)
    y.sum().backward()
    assert b_iq.grad is None
    assert (b.grad - a.grad).sum() == 0
    assert b.grad.size() == a.grad.size()


def test_transpose():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    a = torch.randn((2, 3, 5), requires_grad=True)
    b = a.detach().clone()
    b.requires_grad_()

    x = a.transpose(1, 2)
    sizex = x.size()
    x.sum().backward()

    b_iq = from_torch_tensor(b, 120, 8)
    y = b_iq.transpose(1, 2)
    assert isinstance(y, IQTensor)
    assert y.size() == sizex
    y.sum().backward()
    assert b_iq.grad is None
    assert (b.grad - a.grad).sum() == 0
    assert b.grad.size() == a.grad.size()

def test_getitem():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    a = torch.randn((2, 5, 6, 7), requires_grad=True)
    b = a.detach().clone()
    b.requires_grad_()

    x = a[1, :, :, :]
    sizex = x.size()
    x.sum().backward()

    b_iq = from_torch_tensor(b, 120, 8)
    y = b_iq[1, :, :, :]
    assert isinstance(y, IQTensor)
    assert y.size() == sizex
    y.sum().backward()
    assert b_iq.grad is None
    assert (b.grad - a.grad).sum() == 0
    assert b.grad.size() == a.grad.size()

def test_flatten():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    a = torch.randn((2, 3, 1, 5, 6, 7), requires_grad=True)
    b = a.detach().clone()
    b.requires_grad_()

    x = a.flatten(1, 3)
    sizex = x.size()
    x.sum().backward()

    b_iq = from_torch_tensor(b, 120, 8)
    y = b_iq.flatten(1, 3)
    assert isinstance(y, IQTensor)
    assert y.size() == sizex
    y.sum().backward()
    assert b_iq.grad is None
    assert (b.grad - a.grad).sum() == 0
    assert b.grad.size() == a.grad.size()


def test_split():
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    a = torch.randn((5, 2), requires_grad=True)
    b = a.detach().clone()
    b.requires_grad_()

    x = a.split(2)
    w = torch.cat(x, 0).sum()
    w.backward()
    print(x)

    b_iq = from_torch_tensor(b, 120, 8)
    y = b_iq.split(2)
    z = torch.cat(y, 0).sum()

    z.backward()
    print(y)

    assert isinstance(y[0], IQTensor)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                                  padding=1, bias=True)
            self.pool = nn.AvgPool2d((2, 2), (2, 2), (0, 0), False)
            self.fc = nn.Linear(250, 100)

        def forward(self, x):
            x = self.conv(x)
            n, c, h, w = x.shape
            x = self.pool(x)
            n, c, h, w = x.shape
            x = x.view((n, c*h*w))
            x = x.split(150)
            x = self.fc(x[0])
            return x

    import numpy as np
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    dummy_input = torch.randn(1, 10, 10, 10).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    replace_tuple = (nn.Conv2d, nn.Linear, nn.AvgPool2d)
    net.train()
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    # linger.disable_quant(net.fc)
    net = linger.init(net, quant_modules=replace_tuple,
                      mode=linger.QuantMode.QValue)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss = None
    for i in range(1):
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
        torch.onnx.export(net, dummy_input, "data.ignore/split.onnx", export_params=True, opset_version=11,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    print("out1.mean(): ", out1.mean())
    assert abs(out1.mean() - 1) < 2