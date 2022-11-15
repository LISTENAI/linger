import math

import torch

from ..utils import QuantMode
from .iqtensor import IQTensor, from_torch_tensor


class Requant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bits_src, scale_src, bits_dst, mode=QuantMode.QValue):
        s_rescale = (math.pow(2, bits_dst-1) - 1.0) / \
            (math.pow(2, bits_src-1) - 1.0)
        if mode == QuantMode.QValue:
            s_rescale = math.pow(2, round(math.log(s_rescale, 2)))
        scale_bst = s_rescale*scale_src
        zero_point = 0
        if isinstance(input, IQTensor):
            zero_point = input.zero_point
        if zero_point != 0:
            zero_point = math.pow(2, bits_dst-1)
        s = from_torch_tensor(input, scale_bst, bits_dst, zero_point=zero_point)
        s.requant_()
        return s

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None, None, None

    @staticmethod
    def symbolic(g, input, bits_src, scale_src, bits_dst, mode=QuantMode.QValue):
        s_rescale = (math.pow(2, bits_dst-1) - 1.0) / \
            (math.pow(2, bits_src-1) - 1.0)
        if mode == QuantMode.QValue:
            s_rescale = math.pow(2, round(math.log(s_rescale, 2)))
        scale_bst = s_rescale*scale_src
        return g.op("thinker::Requant", input, data_bits_i=bits_src, scale_x_f=scale_src, scale_o_f=scale_bst, o_bits_i=bits_dst)


__all__ = ['Requant']
