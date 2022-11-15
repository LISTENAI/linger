#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant import NormalizeFunction

class NormalizeConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 normalize_data=None, normalize_weight=None, normalize_bias=None, ahead_relu=False) -> None:
        assert normalize_data is None or normalize_data > 0, 'normalize value is None or must >0'
        assert normalize_weight is None or normalize_weight > 0, 'normalize value is None or must >0'
        assert normalize_bias is None or normalize_bias > 0, 'normalize value is None or must >0'
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size,
                           stride, padding, dilation, groups, bias, padding_mode)
        self.normalize_data = normalize_data
        self.normalize_weight = normalize_weight
        self.normalize_bias = normalize_bias
        self.ahead_relu = ahead_relu

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_weight = self.weight
        if self.normalize_weight is not None:
            normalized_weight = NormalizeFunction.apply(
                normalized_weight, self.normalize_weight, self.training)
        normalized_bias = self.bias
        if (normalized_bias is not None) and (self.normalize_bias is not None):
            normalized_bias = NormalizeFunction.apply(
                normalized_bias, self.normalize_bias, self.training)
        out = None
        if self.padding_mode != 'zeros':
            out = F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                         normalized_weight, normalized_bias, self.stride,
                         tuple(0, 0), self.dilation, self.groups)
        else:
            out = F.conv1d(input, normalized_weight, normalized_bias,
                         self.stride, self.padding, self.dilation, self.groups)
        if self.normalize_data is not None:
            out = NormalizeFunction.apply(out, self.normalize_data, self.training, False)
        return out

    def extra_repr(self):
        s = nn.Conv1d.extra_repr(self)
        extra_s = ',normalize_data:{normalize_data},normalize_weight:{normalize_weight},normalize_bias:{normalize_bias},ahead_relu:{ahead_relu}'.format(
            **self.__dict__)
        return s+extra_s


__all__ = ['NormalizeConv1d']
