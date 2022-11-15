import torch
from linger.ops import from_torch_tensor


def test_relu():
    a = torch.randn((16, 16), requires_grad=True)
    x = a.detach().clone()
    x.requires_grad_()
    b = torch.relu(a)
    b.sum().backward()

    y = from_torch_tensor(x, 5, 8)
    c = torch.relu(y)
    assert c.scale_data == 5
    assert c.bits == 8
    c.sum().backward()

    assert (a.grad - x.grad).sum() == 0


def test_relu_():
    a = torch.randn((16, 16), requires_grad=True)
    x = a.detach().clone()
    x.requires_grad_()
    b = torch.relu_(a)
    b.sum().backward()

    y = from_torch_tensor(x, 5, 8)
    c = torch.relu_(y)
    assert c.scale_data == 5
    assert c.bits == 8
    c.sum().backward()

    assert (a.grad - x.grad).sum() == 0


def test_max_pool2d():
    a = torch.randn((1, 16, 16), requires_grad=True)
    x = a.detach().clone()
    x.requires_grad_()
    b = torch.max_pool2d(a, kernel_size=(2, 2), stride=2, padding=0)
    b.sum().backward()

    y = from_torch_tensor(x, 5, 8)
    c = torch.max_pool2d(y, kernel_size=(2, 2), stride=2, padding=0)
    assert c.scale_data == 5
    assert c.bits == 8
    c.sum().backward()

    assert (a.grad - x.grad).sum() == 0
