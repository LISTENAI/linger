#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn

from ..quant import NormalizeFunction


class NormalizeLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True,
                 normalize_data=None, normalize_weight=None, normalize_bias=None, ahead_relu=False) -> None:
        assert normalize_data is None or normalize_data > 0, 'normalize value is None or must >0'
        assert normalize_weight is None or normalize_weight > 0, 'normalize value is None or must >0'
        assert normalize_bias is None or normalize_bias > 0, 'normalize value is None or must >0'
        nn.LayerNorm.__init__(self, normalized_shape, eps, elementwise_affine)
        self.normalize_data = normalize_data
        self.normalize_weight = normalize_weight
        self.normalize_bias = normalize_bias
        self.ahead_relu = ahead_relu

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        H, W = input.shape[-2], input.shape[-1]
        size = H * W
        if len(self.weight.shape) == 1:
            w_shape = (-1)
        elif len(self.weight.shape) == 2:
            w_shape = (-2, -1)
        elif len(self.weight.shape) == 3:
            w_shape = (-3, -2, -1)
        elif len(self.weight.shape) == 4:
            w_shape = (-4, -3, -2, -1)
        else:
            assert False, f"weight.shape=={self.weight.shape}, please check your LayerNorm definition."
        mean = input.clone().sum(w_shape, keepdim=True) / size
        var = input.clone().pow(2).sum(w_shape, keepdim=True) / size - \
            (input.clone().sum(w_shape, keepdim=True) / size).pow(2)
        var = 1/torch.sqrt(var + self.eps)
        var = torch.clamp(var, min=0.0)
        x_normal = (input - mean) * var

        normalized_weight = self.weight
        if self.normalize_weight is not None:
            normalized_weight = NormalizeFunction.apply(
                normalized_weight, self.normalize_weight, self.training)
        normalized_bias = self.bias
        if (normalized_bias is not None) and (self.normalize_bias is not None):
            normalized_bias = NormalizeFunction.apply(
                normalized_bias, self.normalize_bias, self.training)
        out = normalized_weight * x_normal + normalized_bias
        if self.normalize_data is not None:
            out = NormalizeFunction.apply(
                out, self.normalize_data, self.training, False)
        return out

    def extra_repr(self) -> str:
        s = nn.LayerNorm.extra_repr(self)
        extra_s = ',normalize_data:{normalize_data},normalize_weight:{normalize_weight},normalize_bias:{normalize_bias},ahead_relu:{ahead_relu}'.format(
            **self.__dict__)
        return s+extra_s


__all__ = ['NormalizeLayerNorm']
