#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
from linger import NormalizeLinear


def test_normalize_linear_foward():
    module = NormalizeLinear(512, 128, True, normalize_data=4,
                             normalize_weight=4, normalize_bias=4)
    module.weight.data.fill_(8)
    module.bias.data.fill_(8)
    input = 8 * torch.randn(2, 512)
    assert (input < 8).any()
    assert (input > 4).any()
    m = module(input)
    assert (m < 4.0001).all()
