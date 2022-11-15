import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import config
from ..quant import (Quant, normalize_bias_with_config,
                     normalize_data_with_config, normalize_weight_with_config)
from ..utils import Dump, PlatFormQuant, QuantMode, ScalerBuffer
from .iqtensor import (IQTensor, from_torch_tensor, platform_to_string,
                       quantlinear)
from .ops import ModuleIntConfig
from .requant import Requant

from ..modules.normalize_shuffleChannel import NormalizeShuffleChannel


class ShuffleChannelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, groups, 
                data_bits, o_bits,
                running_x, running_o, scale_x, scale_o,
                training, clamp_data):
        if training:
            ctx.clamp_data = clamp_data
            zero_point = input.zero_point if isinstance(input, IQTensor) else 0
            is_iq_tensor = True if isinstance(input, IQTensor) else False
            ctx.value = zero_point, is_iq_tensor
            ctx.bits = data_bits, o_bits
            saved_tensors = [input]


    @staticmethod
    def backward(ctx, gradOutput):
        pass

    @staticmethod
    def symbolic(g):
        pass

class ShuffleChannelInt(NormalizeShuffleChannel):
    def __init__(self, groups: int, data_bits=8, o_bits=None, clamp_data=None):
        NormalizeShuffleChannel.__init__(self, groups)
        self.groups = groups
        self.clamp_data = clamp_data
        self.register_buffer('running_x', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))
        self.register_buffer('scale_x', torch.zeros(1))
        self.register_buffer('scale_o', torch.zeros(1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale_x = ScalerBuffer(self.scale_x)
        running_x = ScalerBuffer(self.running_x)
        if isinstance(input, IQTensor):
            self.is_not_from_iqtensor = False
            if input.bits != self.data_bits:
                input = Requant.apply(
                    input, input.bits, input.scale_data, self.data_bits, self.mode)
            scale_x = ScalerBuffer(input.scale_data)
            running_x = ScalerBuffer(input.running_data)
        running_o = ScalerBuffer(self.running_o)
        scale_o = ScalerBuffer(self.scale_o)

        ret = ShuffleChannelFunction.apply(input, self.groups, 
                                        self.data_bits, self.o_bits,
                                        running_x, running_o, scale_x, scale_o,
                                        self.training, self.clamp_data)

        self.running_x.fill_(running_x())
        self.running_o.fill_(running_o())
        self.scale_x.fill_(scale_x())
        self.scale_o.fill_(scale_o())
        return ret

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return ModuleIntConfig.state_dict_global(self, destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def extra_repr(self):
        s = nn.Linear.extra_repr(self)
        extra_s = ',clamp_data:{clamp_data},clamp_bias:{clamp_bias},ahead_relu:{ahead_relu},ahead_sigmoid:{ahead_sigmoid}'.format(
            **self.__dict__)
        extra_s += ',data_bits:{data_bits},o_bits:{o_bits}'.format(
            **self.__dict__)
        return s+extra_s
