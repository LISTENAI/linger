#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
from linger import NormalizeConv1d


def test_normalize_conv2d_foward():
    module = NormalizeConv1d(64, 128, kernel_size=3,
                             normalize_data=4, normalize_weight=4, normalize_bias=4)
    module.weight.data.fill_(8)
    module.bias.data.fill_(8)
    input = 8 * torch.randn(1, 64, 512)
    assert (input < 8).any()
    assert (input > 4).any()
    m = module(input)
    assert (m < 4.0001).all()
