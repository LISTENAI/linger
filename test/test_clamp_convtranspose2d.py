#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from linger import NormalizeConvTranspose2d


def test_normalize_linear_foward():
    module = NormalizeConvTranspose2d(
        128, 512, 3, bias=True, normalize_data=4, normalize_weight=4, normalize_bias=4)
    module.weight.data.fill_(8)
    module.bias.data.fill_(8)
    input = 8 * torch.randn(2, 128, 10, 10)
    assert (input < 8).any()
    assert (input > 4).any()
    m = module(input)
    assert (m < 4.0001).all()
