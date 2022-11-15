from typing import Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant import NormalizeFunction

T = TypeVar('T')
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_size_2_t = _scalar_or_tuple_2_t[int]


class NormalizeConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 output_padding: _size_2_t = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: int = 1,
                 padding_mode: str = 'zeros',
                 normalize_data=None, normalize_weight=None, normalize_bias=None,
                 ) -> None:
        assert normalize_data is None or normalize_data > 0, 'normalize value is None or must >0'
        assert normalize_weight is None or normalize_weight > 0, 'normalize value is None or must >0'
        assert normalize_bias is None or normalize_bias > 0, 'normalize value is None or must >0'

        # venus limits
        if type(stride) == int:
            if stride == 2:
                assert kernel_size in (
                    2, 3, 4, 5), f"in NormalizeConvTranspose2d op, when stride_h == 2, kernel_h must be 2, 3, 4 or 5, but you have kernel_size: {kernel_size}"
            elif stride == 4:
                assert kernel_size in (
                    4, 5), f"in NormalizeConvTranspose2d op, when stride_h == 4, kernel_h must be 4 or 5, but you have kernel_size: {kernel_size}"
        else:
            if stride[1] == 2:
                assert kernel_size[1] in (
                    2, 3, 4, 5), f"in NormalizeConvTranspose2d op, when stride_h == 2, kernel_h must be 2, 3, 4 or 5, but you have kernel_size[1]: {kernel_size[1]}"
            if stride[1] == 4:
                assert kernel_size[1] in (
                    4, 5), f"in NormalizeConvTranspose2d op, when stride_h == 4, kernel_h must be 4 or 5, but you have kernel_size[1]: {kernel_size[1]}"
            if stride[0] == 2:
                assert kernel_size[0] in (
                    2, 3, 4, 5), f"in NormalizeConvTranspose2d op, when stride_2 == 2, kernel_2 must be 2, 3, 4 or 5, but you have kernel_size[0]: {kernel_size[0]}"
            if stride[0] == 4:
                assert kernel_size[0] in (
                    4, 5), f"in NormalizeConvTranspose2d op, when stride_w == 4, kernel_w must be 4 or 5, but you have kernel_size[0]: {kernel_size[0]}"

        nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size,
                                    stride, padding, output_padding, groups, bias, dilation, padding_mode)
        self.normalize_data = normalize_data
        self.normalize_weight = normalize_weight
        self.normalize_bias = normalize_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_weight = self.weight
        if self.normalize_weight:
            normalized_weight = NormalizeFunction.apply(
                normalized_weight, self.normalize_weight, self.training)
        normalized_bias = self.bias
        if (self.bias is not None) and (self.normalize_bias is not None):
            normalized_bias = NormalizeFunction.apply(
                normalized_bias, self.normalize_bias, self.training)
        out = F.conv_transpose2d(
            input, normalized_weight, normalized_bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation)
        if self.normalize_data is not None:
            out = NormalizeFunction.apply(
                out, self.normalize_data, self.training, False)
        return out

    def extra_repr(self):
        s = nn.ConvTranspose2d.extra_repr(self)
        extra_s = ',normalize_data:{normalize_data},normalize_weight:{normalize_weight},normalize_bias:{normalize_bias}'.format(
            **self.__dict__)
        return s+extra_s


__all__ = ['NormalizeConvTranspose2d']
