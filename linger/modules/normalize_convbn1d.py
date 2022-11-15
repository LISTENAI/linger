#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant import NormalizeFunction


class NormalizeConvBN1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 normalize_data=None, normalize_weight=None, normalize_bias=None, ahead_relu=False) -> None:
        assert normalize_data is None or normalize_data > 0, 'normalize value is None or must >0'
        assert normalize_weight is None or normalize_weight > 0, 'normalize value is None or must >0'
        assert normalize_bias is None or normalize_bias > 0, 'normalize value is None or must >0'
        super(NormalizeConvBN1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        self.bn = nn.BatchNorm1d(
            out_channels, eps, momentum, affine, track_running_stats)

        self.normalize_data = normalize_data
        self.normalize_weight = normalize_weight
        self.normalize_bias = normalize_bias
        self.ahead_relu = ahead_relu

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            conv_rlt = self.conv(input)
            batchsize, channels, lenth = conv_rlt.size()
            numel = batchsize * lenth
            conv_rlt = conv_rlt.permute(
                1, 0, 2).contiguous().view(channels, numel)
            sum_ = conv_rlt.sum(1)
            sum_of_square = conv_rlt.pow(2).sum(1)
            mean = sum_ / numel
            sumvar = sum_of_square - sum_ * mean
            unbias_var = sumvar / (numel - 1)
            unbias_var = torch.clamp(unbias_var, min=0.0)
            self.bn.running_mean = (
                (1 - self.bn.momentum) * self.bn.running_mean + self.bn.momentum * mean.detach())
            self.bn.running_var = (
                (1 - self.bn.momentum) * self.bn.running_var + self.bn.momentum * unbias_var.detach())

            bias_var = sumvar / numel
            bias_var = torch.clamp(bias_var, min=0.0)
            inv_std = 1 / (bias_var + self.bn.eps).pow(0.5)
            bn_rlt = ((conv_rlt - mean.unsqueeze(1)) * inv_std.unsqueeze(1)
                      * self.bn.weight.unsqueeze(1) + self.bn.bias.unsqueeze(1))
            bn_rlt = bn_rlt.view(channels, batchsize, lenth).permute(
                1, 0, 2).contiguous()

            w_bn = self.bn.weight.div(torch.sqrt(self.bn.eps + unbias_var))
            new_weight = self.conv.weight.mul(w_bn.view(-1, 1, 1))
            if self.conv.bias is not None:
                b_conv = self.conv.bias
            else:
                b_conv = torch.zeros(
                    self.conv.weight.size(0), device=input.device)
            b_bn = self.bn.bias - \
                self.bn.weight.mul(mean).div(
                    torch.sqrt(unbias_var + self.bn.eps))
            new_bias = b_conv.mul(w_bn) + b_bn
            alpha = 1.0
            if self.normalize_weight is not None:
                new_weight = NormalizeFunction.apply(
                    new_weight, self.normalize_weight, self.training)
            if self.normalize_bias is not None:
                new_bias = NormalizeFunction.apply(
                    new_bias, self.normalize_bias, self.training)
            new_conv_rlt = F.conv1d(input, new_weight, new_bias, self.conv.stride,
                                    self.conv.padding, self.conv.dilation, self.conv.groups)
            out = alpha * bn_rlt + (1 - alpha) * new_conv_rlt
        else:
            w_bn = self.bn.weight.div(torch.sqrt(
                self.bn.eps + self.bn.running_var))
            new_weight = self.conv.weight.mul(w_bn.view(-1, 1, 1))
            if self.conv.bias is not None:
                b_conv = self.conv.bias
            else:
                b_conv = torch.zeros(
                    self.conv.weight.size(0), device=input.device)
            b_bn = self.bn.bias - self.bn.weight.mul(self.bn.running_mean).div(
                torch.sqrt(self.bn.running_var + self.bn.eps))
            new_bias = b_conv.mul(w_bn) + b_bn
            if self.normalize_weight is not None:
                new_weight = NormalizeFunction.apply(
                    new_weight, self.normalize_weight, self.training)
            if self.normalize_bias is not None:
                new_bias = NormalizeFunction.apply(
                    new_bias, self.normalize_bias, self.training)
            out = F.conv1d(input, new_weight, new_bias, self.conv.stride,
                              self.conv.padding, self.conv.dilation, self.conv.groups)
        if self.normalize_data is not None:
            out = NormalizeFunction.apply(out, self.normalize_data, self.training, False)
        return out

    def extra_repr(self):
        return 'normalize_data:{normalize_data},normalize_weight:{normalize_weight},normalize_bias:{normalize_bias},ahead_relu:{ahead_relu}'.format(**self.__dict__)


__all__ = ['NormalizeConvBN1d']
