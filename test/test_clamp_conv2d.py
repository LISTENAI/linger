import torch
from linger import NormalizeConv2d


def test_normalize_conv2d_foward():
    module = NormalizeConv2d(2, 4, (3, 3), normalize_data=4,
                             normalize_weight=4, normalize_bias=4)
    module.weight.data.fill_(8)
    module.bias.data.fill_(8)
    input = 8 * torch.randn(1, 2, 4, 4)
    assert (input < 8).any()
    assert (input > 4).any()
    m = module(input)
    assert (m < 4.0001).all()