import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant import NormalizeFunction


class NormalizeLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 normalize_data=None, normalize_weight=None, normalize_bias=None, ahead_relu=False, ahead_sigmoid=False) -> None:
        assert normalize_data is None or normalize_data > 0, 'normalize value is None or must >0'
        assert normalize_weight is None or normalize_weight > 0, 'normalize value is None or must >0'
        assert normalize_bias is None or normalize_bias > 0, 'normalize value is None or must >0'
        nn.Linear.__init__(self, in_features, out_features, bias)
        self.normalize_data = normalize_data
        self.normalize_weight = normalize_weight
        self.normalize_bias = normalize_bias
        self.ahead_relu = ahead_relu
        self.ahead_sigmoid = ahead_sigmoid

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_weight = self.weight
        if self.normalize_weight is not None:
            normalized_weight = NormalizeFunction.apply(
                normalized_weight, self.normalize_weight, self.training)
        normalized_bias = self.bias
        if (self.bias is not None) and (self.normalize_bias is not None):
            normalized_bias = NormalizeFunction.apply(
                normalized_bias, self.normalize_bias, self.training)
        out = F.linear(input, normalized_weight, normalized_bias)
        if self.normalize_data:
            out = NormalizeFunction.apply(out, self.normalize_data, self.training, False)
        return out

    def extra_repr(self):
        s = nn.Linear.extra_repr(self)
        extra_s = ',normalize_data:{normalize_data},normalize_weight:{normalize_weight},normalize_bias:{normalize_bias},ahead_relu:{ahead_relu}'.format(
            **self.__dict__)
        return s+extra_s


__all__ = ['NormalizeLinear']
