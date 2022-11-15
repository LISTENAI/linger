#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn

from ..quant import NormalizeFunction


class NormalizeBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 normalize_data=None, normalize_weight=None, normalize_bias=None, ahead_relu=False) -> None:
        assert normalize_data is None or normalize_data > 0, 'normalize value is None or must >0'
        assert normalize_weight is None or normalize_weight > 0, 'normalize value is None or must >0'
        assert normalize_bias is None or normalize_bias > 0, 'normalize value is None or must >0'
        nn.BatchNorm2d.__init__(self, num_features, eps,
                                momentum, affine, track_running_stats)
        self.normalize_data = normalize_data
        self.normalize_weight = normalize_weight
        self.normalize_bias = normalize_bias
        self.ahead_relu = ahead_relu

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batchsize, channels, height, width = input.shape
        size = batchsize * height * width
        if self.training:
            mean = input.sum((0, 2, 3), keepdim=True) / size
            var = input.pow(2).sum((0, 2, 3), keepdim=True) / size - \
                (input.sum((0, 2, 3), keepdim=True) / size).pow(2)
            var = torch.clamp(var, min=0.0)
            self.running_mean = (
                1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze().detach()
            self.running_var = (1 - self.momentum) * self.running_var + \
                self.momentum * var.squeeze().detach()
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
        sigma = 1 / torch.sqrt(var + self.eps)
        alpha = self.weight.view(1, -1, 1, 1) * sigma
        beta = self.bias.view(1, -1, 1, 1) - mean * alpha
        if self.normalize_weight is not None:
            alpha = NormalizeFunction.apply(
                alpha, self.normalize_weight, self.training)
        if self.normalize_bias is not None:
            beta = NormalizeFunction.apply(
                beta, self.normalize_bias, self.training)
        out = alpha * input + beta
        if self.normalize_data is not None:
            out = NormalizeFunction.apply(
                out, self.normalize_data, self.training, False)
        return out

    def extra_repr(self) -> str:
        return 'normalize_data:{normalize_data},normalize_weight:{normalize_weight},normalize_bias:{normalize_bias},ahead_relu:{ahead_relu}'.format(**self.__dict__)


__all__ = ['NormalizeBatchNorm2d']
