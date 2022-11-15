#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import config
from ..quant import Quant
from ..utils import Dump, PlatFormQuant, QuantMode, ScalerBuffer
from .iqtensor import (IQTensor, from_torch_tensor, platform_to_string,
                       quantlinear)
from .ops import ModuleIntConfig
from .requant import Requant


class AvgPool2dIntFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override,
                data_bits, training, momentum, running_x, running_o, eval_scale_x, eval_scale_o, prefix, dump, path, mode, o_bits, quant,
                is_not_from_iqtensor):
        # venus limits
        assert data_bits in (
            4, 8), f"in AvgPool2d op, AvgPool2d data_bits only support 4/8 bits, but you have data_bits {data_bits}"
        assert o_bits in (
            4, 8, 16), f"in AvgPool2d op, AvgPool2d o_bits only support 4/8/16 bits, but you have o_bits {o_bits}"

        scale_x = None
        scale_o = None
        if training:
            ctx.save_for_backward(input)
            ctx.params = [kernel_size, stride, padding,
                          ceil_mode, count_include_pad, divisor_override]
            zero_point = input.zero_point if isinstance(input, IQTensor) else 0
            is_iq_tensor = True if isinstance(input, IQTensor) else False
            # ctx.bits = data_bits
            if isinstance(input, IQTensor):
                q_input, _, max_value_x = quant.quant(
                    input.data, data_bits, eval_scale_x, mode=QuantMode.QValue, quant_data='input',
                    iq_zero_point=input.zero_point)
                scale_x = eval_scale_x
            else:
                q_input, scale_x, max_value_x = quant.quant(
                    input.data, data_bits, mode=mode, quant_data='input')
                scale_x = ScalerBuffer(scale_x)
                running_x.mul_(1-momentum).add_(momentum*max_value_x)
            q_input = q_input.float() if data_bits <= 8 else q_input.double()
            q_outputs_float = F.avg_pool2d(q_input.contiguous(), kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                                           count_include_pad=count_include_pad, divisor_override=divisor_override)
            if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                if isinstance(kernel_size, tuple):
                    kernel_area = kernel_size[0] * kernel_size[1]
                else:
                    kernel_area = kernel_size * kernel_size
                q_outputs = (
                    (q_outputs_float * kernel_area).round() / kernel_area + 0.5).floor()
                bound_value = math.pow(2, data_bits-1) - 1
                q_outputs.clamp_(-bound_value-1, bound_value)
            else:
                assert False, "linger only support luna quant."
            outputs = quant.dequant(q_outputs, scale_x)
            if o_bits is not None:
                running_o.fill_(running_x())
                scale_o = scale_x
            ctx.value = scale_x, data_bits, zero_point, is_iq_tensor
        else:
            assert running_x > 0, 'invalid running_x <= 0, please fintune first'
            scale_x = None
            scale_o = None
            scale_x = quant.running_to_scale(running_x, data_bits, mode=mode)
            scale_x = ScalerBuffer(scale_x)
            if isinstance(input, IQTensor):
                scale_x = eval_scale_x
                q_input, _, _ = quant.quant(
                    input.data, data_bits, scale_x, mode=QuantMode.QValue, quant_data='input', iq_zero_point=input.zero_point)
            else:
                q_input, _, _ = quant.quant(
                    input.data, data_bits, scale_x, mode=mode, quant_data='input')
            q_input = q_input.double()
            q_outputs = F.avg_pool2d(q_input.contiguous(), kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode,
                                     count_include_pad=count_include_pad, divisor_override=divisor_override)
            if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                if isinstance(kernel_size, tuple):
                    kernel_area = kernel_size[0] * kernel_size[1]
                else:
                    kernel_area = kernel_size * kernel_size
                q_outputs = ((q_outputs * kernel_area).round() /
                             kernel_area + 0.5).floor()
                bound_value = math.pow(2, data_bits-1) - 1
                q_outputs.clamp_(-bound_value-1, bound_value)
            else:
                assert False, "linger only support luna quant."
            outputs = quant.dequant(q_outputs, scale_x)
            scale_o = eval_scale_o
            if o_bits is not None:
                scale_o.fill_(scale_x())
            if dump:
                name_list = ["input", "outputs", "q_input",  "q_outputs",
                             "scale_x", "scale_o", "running_x", "running_o"]
                attr_list = [input, outputs, q_input, q_outputs,
                             scale_x.data, scale_o.data, running_x.data,  running_o.data]
                Dump.dump_file(prefix, ".AvgPool2dInt.",
                               zip(name_list, attr_list), path)

        if o_bits is None:
            return outputs
        elif isinstance(scale_o, float):
            return from_torch_tensor(outputs, scale_o, o_bits)
        elif isinstance(scale_o, torch.Tensor):
            return from_torch_tensor(outputs, scale_o.item(), o_bits)
        else:
            return from_torch_tensor(outputs, scale_o.data, o_bits)

    @staticmethod
    def backward(ctx, gradoutput):
        input, = ctx.saved_tensors
        scale_x, data_bits, zero_point, is_iq_tensor = ctx.value
        kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override = ctx.params
        if is_iq_tensor:
            backward_input = input.data
        else:
            q_input, _, _ = Quant.quant(
                input.data, data_bits, scale_x, mode=QuantMode.QValue, quant_data='input')
            backward_input = Quant.dequant(q_input, scale_x)
        backward_input = backward_input.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            output = F.avg_pool2d(backward_input, kernel_size, stride,
                                  padding, ceil_mode, count_include_pad, divisor_override)
            grad = torch.autograd.grad(output, backward_input, gradoutput)
        return grad[0], None, None, None, None, None, None, \
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override,
                 data_bits, training, momentum, running_x, running_o, scale_x, scale_o, prefix, dump, path, mode, o_bits, quant, is_not_from_iqtensor):
        op_inner = None
        if is_not_from_iqtensor:
            platform_quant = platform_to_string(
                config.PlatFormQuant.platform_quant)
            op_inner = quantlinear(g, input, scale_x(),
                                   platform_quant, data_bits)
        if isinstance(padding, int):
            padding = [padding]*2
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size]*2
        if isinstance(stride, int):
            stride = [stride]*2
        paddings = padding + padding
        param_dict = {'kernel_shape_i': kernel_size, 'strides_i': stride, 'pads_i': paddings, 'ceil_mode_i': ceil_mode, 'data_bits_i': data_bits,
                      'scale_x_f': scale_x()}
        if is_not_from_iqtensor:
            input_list = [op_inner, ]
        else:
            input_list = [input, ]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict['platform_quant_s'] = platform_quant
        if o_bits is not None:
            param_dict['scale_o_f'] = scale_o()
            param_dict['o_bits_i'] = o_bits
        return g.op("thinker::AvgPool2dInt", *input_list, **param_dict)


class AvgPool2dInt(nn.AvgPool2d, ModuleIntConfig):
    r"""实现AvgPool2dInt的量化训练与测试，继承自nn.AvgPool2d,
    Args: 
        kernel_size stride padding, ceil_mode count_include_pad divisor_override
        与nn.AvgPool2d一致
        data_bits(int, default=8): 输入的量化位数
        mode(Enum): 量化方式，支持MaxValue与Qvalue
        o_bits(int, default=None):输出量化位数
    Notes:
        支持luna_quant
        luna_quant: x = (x + 0.5).floor()

    """

    def __init__(self, kernel_size, stride, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None,
                 data_bits=8, mode=QuantMode.QValue, o_bits=None):
        nn.AvgPool2d.__init__(self, kernel_size, stride, padding,
                              ceil_mode, count_include_pad, divisor_override)
        ModuleIntConfig.__init__(
            self, data_bits=data_bits, mode=mode, o_bits=o_bits)
        self.momentum = 0.1
        self.is_not_from_iqtensor = True
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.register_buffer('running_x', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))
        self.register_buffer('scale_x', torch.zeros(1))
        self.register_buffer('scale_o', torch.zeros(1))

    def forward(self, input):
        scale_x = ScalerBuffer(self.scale_x)
        running_x = ScalerBuffer(self.running_x)
        assert (self.o_bits is None or self.o_bits ==
                self.data_bits), 'AvgPool2dInt out_bits must equal data_bits'
        if isinstance(input, IQTensor):
            self.is_not_from_iqtensor = False
            if input.bits != self.data_bits:
                input = Requant.apply(
                    input, input.bits, input.scale_data, self.data_bits)
            scale_x = ScalerBuffer(input.scale_data)
            running_x = ScalerBuffer(input.running_data)
        running_o = ScalerBuffer(self.running_o)
        scale_o = ScalerBuffer(self.scale_o)
        output = AvgPool2dIntFunction.apply(input.contiguous(), self.kernel_size,
                                            self.stride, self.padding, self.ceil_mode,
                                            self.count_include_pad, self.divisor_override,
                                            self.data_bits, self.training, self.momentum,
                                            running_x, running_o, scale_x, scale_o, self.prefix, self.dump, self.path,
                                            self.quant_mode, self.o_bits, self.quant, self.is_not_from_iqtensor)
        self.running_x.fill_(running_x())
        self.running_o.fill_(running_o())
        self.scale_x.fill_(scale_x())
        self.scale_o.fill_(scale_o())
        return output

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return ModuleIntConfig.state_dict_global(self, destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
