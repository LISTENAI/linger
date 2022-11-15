import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant import NormalizeFunction


class NormalizeConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 normalize_data=None, normalize_weight=None, normalize_bias=None, ahead_relu=False, ahead_sigmoid=False) -> None:

        assert normalize_data is None or normalize_data > 0, 'normalize value is None or must >0'
        assert normalize_weight is None or normalize_weight > 0, 'normalize value is None or must >0'
        assert normalize_bias is None or normalize_bias > 0, 'normalize value is None or must >0'

        # venus limits
        if type(kernel_size) == int:
            assert kernel_size in (
                1, 2, 3, 4, 5), f"in NormalizeConv2d op, kernel size must be 1/2/3/4/5, but you have kernel size {kernel_size}"
        elif type(kernel_size) == tuple:
            assert kernel_size[0] in (1, 2, 3, 4, 5) and kernel_size[1] in (
                1, 2, 3, 4, 5), "in NormalizeConv2d op, kernel size must be 1/2/3/4/5, , but you have kernel size {kernel_size}"

        if type(stride) == int:
            assert stride in (
                1, 2, 4), "in NormalizeConv2d op, kernel size must be 1/2/4, but you have stride {stride}"
        elif type(stride) == tuple:
            assert stride[0] in (1, 2, 4) and stride[1] in (
                1, 2, 4), "in NormalizeConv2d op, kernel size must be 1/2/4, but you have stride {stride}"

        if type(padding) == int:
            assert padding in (
                0, 1, 2, 3, 4), "in NormalizeConv2d op, padding size must be 1/2/4, but you have padding {padding}"
        elif type(padding) == tuple:
            assert padding[0] in (0, 1, 2, 3, 4) and padding[1] in (
                0, 1, 2, 3, 4), "in NormalizeConv2d op, padding size must be 1/2/3/4/5, but you have padding {padding}"

        if type(kernel_size) == int:
            kernel_size_h = kernel_size
            kernel_size_w = kernel_size
        elif type(kernel_size) == tuple:
            kernel_size_h = kernel_size[0]
            kernel_size_w = kernel_size[1]
        else:
            assert False, "kernel size type error."
        # if (groups != in_channels):
        #     assert math.ceil(in_channels/8) * 8 * kernel_size_h * kernel_size_w * math.ceil(out_channels/2) * 2 <= 32 * \
        #         1024, f"in NormalizeConv2d op, kernel must meet the requirements of non-depthwise convolution, but you have math.ceil({in_channels}/8) * 8 * {kernel_size_h} * {kernel_size_w} * math.ceil({out_channels}/2) * 2 <= 32 * 1024"
        # if (groups == in_channels):
        #     assert math.ceil(in_channels/16) * 16 * kernel_size_h * kernel_size_w <= 32 * \
        #         1024, f"in NormalizeConv2d op, kernel must meet the requirements of depthwise convolution, but you have math.ceil({in_channels}/16) * 16 * {kernel_size_h} * {kernel_size_w} <= 32 * 1024"

        if type(stride) == int:
            stride_h = stride
            stride_w = stride
        elif type(stride) == tuple:
            stride_h = stride[0]
            stride_w = stride[1]
        else:
            assert False, "kernel size type error."

        if type(padding) == int:
            padding_h = padding
            padding_w = padding
        elif type(padding) == tuple:
            padding_h = padding[0]
            padding_w = padding[1]
        else:
            assert False, "kernel size type error."

        assert kernel_size_h >= stride_h and kernel_size_w >= stride_w, f"kernel_size_h >= stride_h and kernel_size_w >= stride_w, but you have {kernel_size_h} < {stride_h} or {kernel_size_w} < {stride_w}"
        assert padding_h < kernel_size_h and padding_w < kernel_size_w, f"pad_h < weight_h && pad_w < weight_w, but you have {padding_h} >= {kernel_size_h} or {padding_w} >= {kernel_size_w}"

        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size,
                           stride, padding, dilation, groups, bias, padding_mode)
        self.normalize_data = normalize_data
        self.normalize_weight = normalize_weight
        self.normalize_bias = normalize_bias
        self.ahead_relu = ahead_relu
        self.ahead_sigmoid = ahead_sigmoid

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # venus limits
        # assert input.shape[2] >= self.weight.shape[2] and input.shape[3] >= self.weight.shape[
            # 3], f"in NormalizeConv2d op, input's width >= weight's width && input'height >= weight'height, but you have {input.shape[2]} < {self.weight.shape[2]} and {input.shape[3]} < {self.weight.shape[3]}"

        # channel_in = self.weight.shape[1]/self.groups
        # assert not (math.ceil(channel_in/8/self.stride[1]) * (8*self.stride[1]) * math.ceil(input.shape[3]/8)*8*1 > 64 * 1024 and channel_in > 512) or not (math.ceil(channel_in/8/self.stride[1]) * (8*self.stride[1]) * math.ceil(8/8)*8*input.shape[2] > 64 * 1024 and channel_in >
                                                                                                                                                            # 512), f"in NormalizeConv2d op, the size of the input data after alignment exceed 64KB and channal_in > 512 at the same time is not allowed, but you have (math.ceil({channel_in}/8/{self.stride[1]}) * (8*{self.stride[1]}) * math.ceil({input.shape[3]}/8)*8*{1} > 64 * 1024 and {channel_in} > 512) or (math.ceil({channel_in}/8/{self.stride[1]}) * (8*{self.stride[1]}) * math.ceil({8}/8)*8*{input.shape[2]} > 64 * 1024 and {channel_in} > 512"

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
            out = F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                           normalized_weight, normalized_bias, self.stride,
                           tuple(0, 0), self.dilation, self.groups)
        else:
            out = F.conv2d(input, normalized_weight, normalized_bias,
                           self.stride, self.padding, self.dilation, self.groups)
        if self.normalize_data is not None:
            out = NormalizeFunction.apply(
                out, self.normalize_data, self.training, False)
        return out

    def extra_repr(self):
        s = nn.Conv2d.extra_repr(self)
        extra_s = ',normalize_data:{normalize_data},normalize_weight:{normalize_weight},normalize_bias:{normalize_bias},ahead_relu:{ahead_relu}'.format(
            **self.__dict__)
        return s+extra_s


__all__ = ['NormalizeConv2d']
