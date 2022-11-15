import os

import linger
import torch
from linger.ops import (IQTensor, from_torch_tensor, iqadd, iqAddLayer,
                             iqmul, iqMulLayer, torch_cat)

if not os.path.exists('data.ignore'):
    os.mkdir('data.ignore')


def test_iqMul():
    x = torch.tensor([[6.0, 4.0]], device=torch.device(
        'cuda:0'), requires_grad=True)

    y = torch.tensor([[3.0, 8.0]], device=torch.device(
        'cuda:0'), requires_grad=True)
    iq_layer = iqMulLayer().cuda()
    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)

    m = iq_layer(a, b, 127.0/32)  # iqAdd.apply(x,y).sum().backward()
    assert abs(m.data[0][0].item()-18.1417) < 0.2
    assert abs(m.data[0][1].item()-32) < 0.3
    m.sum().backward()

    p = torch.tensor([[6.0, 4.0]], device=torch.device(
        'cuda:0'), requires_grad=True).cuda()
    q = torch.tensor([[3.0, 8.0]], device=torch.device(
        'cuda:0'), requires_grad=True).cuda()
    m = p*q
    m = m.sum()
    m.backward()
    err = p.grad.data - x.grad.data
    assert err.abs().sum() < 0.01
    err = q.grad.data - y.grad.data
    assert err.abs().sum() < 0.01


def test_iqMul_layer():
    x = torch.tensor([[6.0, 4.0]], device=torch.device(
        'cuda:0'), requires_grad=True)
    y = torch.tensor([[3.0, 8.0]], device=torch.device(
        'cuda:0'), requires_grad=True)

    class iqTestLayer(torch.nn.Module):
        def __init__(self):
            super(iqTestLayer, self).__init__()

        def forward(self, x, y):
            return iqmul(self, x, y, 'test')

    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)

    net = iqTestLayer().cuda()  # iqAdd.apply(x,y).sum().backward()
    m = net(a, b)
    assert abs(m.data[0][0].item()-18.1417) < 0.2
    assert abs(m.data[0][1].item()-32) < 0.3
    m.sum().backward()

    p = torch.tensor([[6.0, 4.0]], device=torch.device(
        'cuda:0'), requires_grad=True)
    q = torch.tensor([[3.0, 8.0]], device=torch.device(
        'cuda:0'), requires_grad=True)
    m = p*q
    m = m.sum()
    m.backward()
    err = p.grad.data - x.grad.data
    assert a.grad is None
    assert err.abs().sum() < 0.01
    err = q.grad.data - y.grad.data
    assert b.grad is None
    assert err.abs().sum() < 0.01


def test_iqMul_layer_cpu():
    x = torch.tensor([[6.0, 4.0]], requires_grad=True)
    y = torch.tensor([[3.0, 8.0]], requires_grad=True)

    class iqTestLayer(torch.nn.Module):
        def __init__(self):
            super(iqTestLayer, self).__init__()

        def forward(self, x, y):
            return iqmul(self, x, y, 'test')

    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)

    net = iqTestLayer()
    m = net(a, b)
    # print(m)
    assert abs(m.data[0][0].item()-18.1417) < 0.2
    assert abs(m.data[0][1].item()-32) < 0.3
    m.sum().backward()

    p = torch.tensor([[6.0, 4.0]], requires_grad=True)
    q = torch.tensor([[3.0, 8.0]], requires_grad=True)
    m = p*q
    m = m.sum()
    m.backward()
    err = p.grad.data - x.grad.data
    assert a.grad is None
    assert err.abs().sum() < 0.01
    err = q.grad.data - y.grad.data
    assert b.grad is None
    assert err.abs().sum() < 0.01


def test_iqmul_module():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.aa = torch.nn.Parameter(torch.zeros((1)))

        def forward(self, x, y):
            return x * y
    net = TestModel().cuda()
    x = torch.tensor([[6.0, 4.0]], requires_grad=True)
    y = torch.tensor([[3.0, 8.0]], requires_grad=True)
    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)
    net = linger.init(net)
    z = net(a, b)
    assert isinstance(z, IQTensor)


def test_iqimul_module():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.aa = torch.nn.Parameter(torch.zeros((1)))

        def forward(self, x, y):
            x *= y
            return x
    net = TestModel().cuda()
    x = torch.tensor([[6.0, 4.0]], requires_grad=True)
    y = torch.tensor([[3.0, 8.0]], requires_grad=True)
    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)
    net = linger.init(net)
    z = net(a, b)
    assert isinstance(z, IQTensor)


def test_iqAdd():
    x = torch.tensor([[6.0, 4.0]], device=torch.device(
        'cuda:0'), requires_grad=True)

    y = torch.tensor([[3.0, 8.0]], device=torch.device(
        'cuda:0'), requires_grad=True)
    iq_layer = iqAddLayer().cuda()
    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)

    m = iq_layer(a, b, 127.0/12)  # iqAdd.apply(x,y).sum().backward()
    assert m.data[0][0]-16.535 < 0.1
    assert m.data[0][1]-20.031 < 0.1
    m.sum().backward()

    p = torch.tensor([[6.0, 4.0]], device=torch.device(
        'cuda:0'), requires_grad=True).cuda()
    q = torch.tensor([[3.0, 8.0]], device=torch.device(
        'cuda:0'), requires_grad=True).cuda()
    m = p+q
    m = m.sum()
    m.backward()
    err = p.grad.data - x.grad.data
    assert err.abs().sum() < 0.01
    err = q.grad.data - y.grad.data
    assert err.abs().sum() < 0.01

def test_iqAdd_layer():
    x = torch.tensor([[6.0, 4.0]], device=torch.device(
        'cuda:0'), requires_grad=True)
    y = torch.tensor([[3.0, 8.0]], device=torch.device(
        'cuda:0'), requires_grad=True)

    class iqTestLayer(torch.nn.Module):
        def __init__(self):
            super(iqTestLayer, self).__init__()

        def forward(self, x, y):
            return iqadd(self, x, y, 'test')

    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)

    net = iqTestLayer().cuda()  # iqAdd.apply(x,y).sum().backward()
    m = net(a, b)
    assert m.data[0][0]-16.535 < 0.1
    assert m.data[0][1]-20.031 < 0.1
    m.sum().backward()

    p = torch.tensor([[6.0, 4.0]], device=torch.device(
        'cuda:0'), requires_grad=True)
    q = torch.tensor([[3.0, 8.0]], device=torch.device(
        'cuda:0'), requires_grad=True)
    m = p+q
    m = m.sum()
    m.backward()
    err = p.grad.data - x.grad.data
    assert a.grad is None
    assert err.abs().sum() < 0.01
    err = q.grad.data - y.grad.data
    assert b.grad is None
    assert err.abs().sum() < 0.01


def test_iqAdd_layer_cpu():
    x = torch.tensor([[6.0, 4.0]], requires_grad=True)
    y = torch.tensor([[3.0, 8.0]], requires_grad=True)

    class iqTestLayer(torch.nn.Module):
        def __init__(self):
            super(iqTestLayer, self).__init__()

        def forward(self, x, y):
            return iqadd(self, x, y, 'test')

    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)

    net = iqTestLayer()
    m = net(a, b)
    assert m.data[0][0]-16.535 < 0.1
    assert m.data[0][1]-20.031 < 0.1
    m.sum().backward()

    p = torch.tensor([[6.0, 4.0]], requires_grad=True)
    q = torch.tensor([[3.0, 8.0]], requires_grad=True)
    m = p+q
    m = m.sum()
    m.backward()
    err = p.grad.data - x.grad.data
    assert a.grad is None
    assert err.abs().sum() < 0.01
    err = q.grad.data - y.grad.data
    assert b.grad is None
    assert err.abs().sum() < 0.01


def test_iqadd_module():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.aa = torch.nn.Parameter(torch.zeros((1)))

        def forward(self, x, y):
            return x + y
    net = TestModel().cuda()
    x = torch.tensor([[6.0, 4.0]], requires_grad=True)
    y = torch.tensor([[3.0, 8.0]], requires_grad=True)
    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)
    net = linger.init(net)
    z = net(a, b)
    assert isinstance(z, IQTensor)


def test_iqiadd_module():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.aa = torch.nn.Parameter(torch.zeros((1)))

        def forward(self, x, y):
            x += y
            return x
    net = TestModel().cuda()
    x = torch.tensor([[6.0, 4.0]], requires_grad=True)
    y = torch.tensor([[3.0, 8.0]], requires_grad=True)
    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)
    net = linger.init(net)
    z = net(a, b)
    assert isinstance(z, IQTensor)


def test_iqcat_function():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.aa = torch.nn.Parameter(torch.zeros((1)))

        def forward(self, x, y, z):
            return torch.cat((x, y, z), dim=0)

    x = torch.tensor([[6.0, 4.0]], requires_grad=True)
    y = torch.tensor([[3.0, 8.0]], requires_grad=True)
    z = torch.tensor([[5.0, 7.0]], requires_grad=True)

    a = torch_cat((x, y, z), dim=0)

    a.sum().backward()

    x_ = torch.tensor([[6.0, 4.0]], requires_grad=True)
    y_ = torch.tensor([[3.0, 8.0]], requires_grad=True)
    z_ = torch.tensor([[5.0, 7.0]], requires_grad=True)
    x_iq = from_torch_tensor(x_, 127.0/6.0, 8)
    y_iq = from_torch_tensor(y_, 127.0/8.0, 8)
    z_iq = from_torch_tensor(z_, 127.0/7.0, 8)

    net = TestModel()
    net = linger.init(net)
    b = net(x_iq, y_iq, z_iq)
    b.sum().backward()
    assert (b-a).sum() < 0.2
    err = x_.grad.data - x.grad.data
    assert err.sum() < 0.1
    err = y_.grad.data - y.grad.data
    assert err.sum() < 0.1
    err = z_.grad.data - z.grad.data
    assert err.sum() < 0.1


def test_iqcat_running_o():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.aa = torch.nn.Parameter(torch.zeros((1)))

        def forward(self, x, y, z):
            return torch.cat((x, y, z), dim=0)

    x_ = torch.tensor([[6.0, 4.0]], requires_grad=True)
    y_ = torch.tensor([[3.0, 8.0]], requires_grad=True)
    z_ = torch.tensor([[5.0, 7.0]], requires_grad=True)
    x_iq = from_torch_tensor(x_, 127.0/6.0, 8)
    y_iq = from_torch_tensor(y_, 127.0/8.0, 8)
    z_iq = from_torch_tensor(z_, 127.0/7.0, 8)

    net = TestModel()
    net = linger.init(net)
    b = None
    for _ in range(100):
        b = net(x_iq, y_iq, z_iq)
    net.eval()
    c = net(x_iq, y_iq, z_iq)
    assert (c - b).abs().sum() < 0.1


def test_iqcat_state_dict():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()
            self.aa = torch.nn.Parameter(torch.zeros((1)))

        def forward(self, x, y, z):
            return torch.cat((x, y, z), dim=0)

    x_ = torch.tensor([[6.0, 4.0]], requires_grad=True)
    y_ = torch.tensor([[3.0, 8.0]], requires_grad=True)
    z_ = torch.tensor([[5.0, 7.0]], requires_grad=True)
    x_iq = from_torch_tensor(x_, 127.0/6.0, 8)
    y_iq = from_torch_tensor(y_, 127.0/8.0, 8)
    z_iq = from_torch_tensor(z_, 127.0/7.0, 8)

    net = TestModel()
    net = linger.init(net)
    b = None
    for _ in range(100):
        b = net(x_iq, y_iq, z_iq)
    torch.save(net.state_dict(), 'data.ignore/param.dict')

    net2 = TestModel()
    net2 = linger.init(net2)
    net2.load_state_dict(torch.load(
        'data.ignore/param.dict', map_location='cpu'))

    net.eval()
    c = net(x_iq, y_iq, z_iq)
    assert (c - b).abs().sum() < 0.1
