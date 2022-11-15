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


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, data_bits, parameter_bits, running_x, running_w, running_o, eval_scale_x, eval_scale_w, eval_scale_o,
                momentum, training, prefix, dump, path, mode, o_bits, quant, is_not_from_iqtensor, ahead_relu, clamp_data, clamp_weight, clamp_bias, ahead_sigmoid):
        scale_o = None

        # venus limits
        n = input.shape[-1]
        if (data_bits == 8):
            assert math.ceil(4/4) * 4 * math.ceil(n/8) * 8 * data_bits/8 <= 64 * \
                1024, f"in LinearInt op, input shape must satisfy math.ceil(4/4) * 4 * math.ceil(n/8) * 8 * data_bits/8 <= 64*1024, but you have math.ceil({4}/4) * 4 * math.ceil({n}/8) * 8 * {data_bits/8} <= 64*1024"
        elif (data_bits == 16):
            assert math.ceil(4/4) * 4 * math.ceil(n/2) * 2 * data_bits/8 <= 64 * \
                1024, f"in LinearInt op, input shape must satisfy math.ceil(4/4) * 4 * math.ceil(n/2) * 2 * data_bits/8 <= 64*1024, but you have math.ceil({4}/4) * 4 * math.ceil({n}/2) * 2 * {data_bits/8} <= 64*1024"
        elif (data_bits == 32):
            assert math.ceil(2/2) * 2 * math.ceil(n/2) * 2 * data_bits/8 <= 64 * \
                1024, f"in LinearInt op, input shape must satisfy math.ceil(2/2) * 2 * math.ceil(n/2) * 2 * data_bits/8 <= 64*1024, but you have math.ceil({2}/2) * 2 * math.ceil({n}/2) * 2 * {data_bits/8} <= 64*1024"

        if training:
            ctx.clamp_data = clamp_data
            ctx.clamp_weight = clamp_weight
            ctx.clamp_bias = clamp_bias
            zero_point = input.zero_point if isinstance(input, IQTensor) else 0
            is_iq_tensor = True if isinstance(input, IQTensor) else False
            ctx.value = zero_point, is_iq_tensor
            ctx.bits = data_bits, parameter_bits, o_bits
            saved_tensors = [input, weight, bias]
            if isinstance(input, IQTensor):
                q_input, _, max_value_x = quant.quant(
                    input.data, data_bits, eval_scale_x, mode=QuantMode.QValue, quant_data='input', iq_zero_point=input.zero_point)
                scale_x = eval_scale_x
            else:
                q_input, scale_x, max_value_x = quant.quant(
                    input, data_bits, mode=mode, quant_data='input')
                running_x.mul_(1-momentum).add_(momentum*max_value_x)

            q_weight, scale_w, max_value_w = quant.quant(
                weight, parameter_bits, mode=mode, quant_data='weight')
            running_w.mul_(1-momentum).add_(momentum*max_value_w)
            q_input = q_input.float() if data_bits + parameter_bits <= 16 else q_input.double()
            q_weight = q_weight.float() if data_bits + \
                parameter_bits <= 16 else q_weight.double()
            q_outputs = F.linear(q_input, q_weight)
            outputs = None
            q_bias = None
            if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                assert mode == QuantMode.QValue, 'castor quant only support QValue and o_bits=None'
                if bias is not None:
                    q_bias = (bias * scale_w * scale_x + 0.5).floor()
                    if data_bits + parameter_bits <= 16:
                        q_bias = q_bias.float()
                    else:
                        q_bias = q_bias.double()
                    q_outputs += q_bias.reshape(-1)
                outputs = quant.dequant(q_outputs, scale_x*scale_w)
            else:
                assert False, "linger only support luna quant."

            # f_input = quant.dequant(q_input, scale_x)
            # f_weight = quant.dequant(q_weight, scale_w)
            # f_bias = None if bias is None else quant.dequant(q_bias, scale_x*scale_w)
            # saved_tensors = [f_input, f_weight, f_bias]

            out_tensor = outputs
            # ctx.save_for_backward(*saved_tensors)
            if o_bits is not None:
                outputs = normalize_data_with_config(outputs, clamp_data)
                out_tensor = outputs
                q_outputs, scale_o, max_value_o = quant.quant(
                    outputs, o_bits, mode=mode, quant_data='output', ahead_relu=ahead_relu)
                running_o.mul_(1-momentum).add_(momentum*max_value_o)
                outputs = quant.dequant(q_outputs, scale_o)
            ctx.scale = scale_x, scale_w, scale_o
            saved_tensors += [out_tensor]
            ctx.save_for_backward(*saved_tensors)

        else:
            assert running_x > 0, 'invalid running_x, please finetune training'
            q_weight = None
            scale_o = eval_scale_o
            if weight.dtype == torch.float32:
                scale_x = ScalerBuffer(quant.running_to_scale(
                    running_x, data_bits, mode=mode))
                q_weight, scale_w, _ = quant.quant(
                    weight, parameter_bits, mode=mode, quant_data='weight')
                scale_w = ScalerBuffer(scale_w)
                if o_bits is not None:
                    assert running_o > 0, 'invalid running_o <= 0, please finetune training'
                    scale_o = ScalerBuffer(quant.running_to_scale(
                        running_o, o_bits, mode=mode))
            else:
                scale_x = eval_scale_x
                scale_w = eval_scale_w
                q_weight = weight.double()
                if o_bits is not None:
                    scale_o = eval_scale_o

            if isinstance(input, IQTensor):
                scale_x = eval_scale_x
                q_input, _, _ = quant.quant(
                    input.data, data_bits, scale_x, mode=QuantMode.QValue, quant_data='input', iq_zero_point=input.zero_point)
            else:
                q_input, scale_x, _ = quant.quant(
                    input.data, data_bits, scale_x, mode=mode, quant_data='input')
            q_input = q_input.double()
            q_weight = q_weight.double()
            q_outputs = F.linear(q_input, q_weight)

            outputs = None
            q_bias = None
            if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                assert mode == QuantMode.QValue, 'castor quant only support QValue and o_bits=None'
                if bias is not None:
                    if bias.dtype == torch.float32:
                        q_bias = (bias * scale_w * scale_x + 0.5).floor()
                        if data_bits + parameter_bits <= 16:
                            q_bias = q_bias.float().double()
                        else:
                            q_bias = q_bias.double()
                    else:
                        q_bias = bias.double()
                    q_outputs += q_bias.reshape(-1)
                outputs = quant.dequant(q_outputs, scale_x*scale_w)
            else:
                assert False, "linger only support luna quant."
            if o_bits is not None:
                if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                    q_outputs, _, _ = quant.quant(
                        outputs, o_bits, scale_o, mode=mode, quant_data='output')
                outputs = quant.dequant(q_outputs, scale_o)
            if dump:
                if bias is not None:
                    name_list = ["input", "weight", "bias", "outputs", "q_input", "q_weight", "q_bias",
                                 "q_outputs", "running_x", "running_w", "running_o", "scale_x", "scale_w", "scale_o"]
                    attr_list = [input, weight, bias, outputs, q_input, q_weight, q_bias, q_outputs,
                                 running_x.data, running_w.data, running_o.data, scale_x.data, scale_w.data, scale_o.data]
                    Dump.dump_file(prefix, ".LinearInt.",
                                   zip(name_list, attr_list), path)
                else:
                    name_list = ["input", "weight", "outputs", "q_input", "q_weight", "q_outputs",
                                 "running_x", "running_w", "running_o", "scale_x", "scale_w", "scale_o"]
                    attr_list = [input, weight, outputs, q_input, q_weight, q_outputs, running_x.data,
                                 running_w.data, running_o.data, scale_x.data, scale_w.data, scale_o.data]
                    Dump.dump_file(prefix, ".LinearInt.",
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
    def backward(ctx, gradOutput):
        clamp_data = ctx.clamp_data
        data_bits, parameter_bits, o_bits = ctx.bits
        zero_point, is_iq_tensor = ctx.value
        scale_x, scale_w, scale_o = ctx.scale
        input, weights, bias, outputs = ctx.saved_tensors
        if is_iq_tensor:
            f_input = input.data
        else:
            q_input, _, _ = Quant.quant(
                input.data, data_bits, scale_x, mode=QuantMode.QValue, quant_data='input')
            f_input = Quant.dequant(q_input, scale_x)

        f_input = f_input.detach().clone().requires_grad_(True)
        q_weights, _, _ = Quant.quant(
            weights.data, parameter_bits, scale_w, mode=QuantMode.QValue, quant_data='weight')
        f_weights = Quant.dequant(q_weights, scale_w)

        f_weights = f_weights.detach().clone().requires_grad_(True)
        bias = None if bias is None else bias.detach().clone().requires_grad_(True)
        gradInput = gradWeight = gradBias = None

        with torch.enable_grad():
            z = F.linear(f_input, f_weights, bias)
            if o_bits is not None:
                z = normalize_data_with_config(z, clamp_data)
            if bias is not None:
                gradInput, gradWeight, gradBias = torch.autograd.grad(
                    z, (f_input, f_weights, bias), gradOutput)
            else:
                gradInput, gradWeight, = torch.autograd.grad(
                    z, (f_input, f_weights), gradOutput)

        return gradInput, gradWeight, gradBias, None, None, None, None, None, None, None, None, None, None, None, None,\
            None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, input, weight, bias, data_bits, parameter_bits, running_x, running_w, running_o, scale_x, scale_w, scale_o, momentum, training,
                 prefix, dump, path, mode, o_bits, quant, is_not_from_iqtensor, ahead_relu, clamp_data, clamp_weight, clamp_bias, ahead_sigmoid):
        op_inner = None
        if is_not_from_iqtensor:
            platform_quant = platform_to_string(
                config.PlatFormQuant.platform_quant)
            op_inner = quantlinear(g, input, scale_x(),
                                   platform_quant, data_bits)
            input_list = [op_inner, weight]
        else:
            input_list = [input, weight]
        param_dict = {'scale_x_f': scale_x(), 'scale_w_f': scale_w(),
                      'data_bits_i': data_bits, 'parameter_bits_i': parameter_bits}
        if bias is not None:
            input_list.append(bias)
        if o_bits is not None:
            param_dict['scale_o_f'] = scale_o()
            param_dict['o_bits_i'] = o_bits

        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict['platform_quant_s'] = platform_quant

        return g.op("thinker::LinearInt", *input_list, **param_dict)


class LinearInt(nn.Linear, ModuleIntConfig):
    r"""实现LinearInt的量化训练与测试，继承自nn.Linear,

    Args:
        in_features out_features bias
        与nn.Linear一致的参数
        data_bits(int): 输入量化位数
        parameter_bits(int): 参数量化位数
        mode(Enum): 量化方式，支持MaxValue与Qvalue
        o_bits(int, default=None):输出量化位数
        clamp_data(float or None): 针对输出的clamp数值
        clamp_weight(float or None): 针对转为weight的clamp数值
        clamp_bias(float or None): 与clamp_weight一致
        ahead_relu(bool): 是否做融合relu之后的数值统计scale

    """

    def __init__(self, in_features, out_features, bias=True, data_bits=8, parameter_bits=8, mode=QuantMode.QValue,
                 o_bits=None, clamp_data=None, clamp_weight=None, clamp_bias=None, ahead_relu=False, ahead_sigmoid=False):
        # assert data_bits == parameter_bits, "data_bits and parameter_bits must be equal"
        assert data_bits in (8, 16, 32), "data_bits only support 8, 16, 32"
        nn.Linear.__init__(self, in_features, out_features, bias)
        ModuleIntConfig.__init__(
            self, data_bits=data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=o_bits)

        self.prefix = ""
        self.dump = False
        self.path = ""
        self.momentum = 0.1
        self.is_not_from_iqtensor = True
        self.clamp_data = clamp_data
        self.clamp_weight = clamp_weight
        self.clamp_bias = clamp_bias
        self.ahead_relu = ahead_relu
        self.ahead_sigmoid = ahead_sigmoid
        self.mode = mode
        self.register_buffer('running_x', torch.zeros(1))
        self.register_buffer('running_w', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))
        self.register_buffer('scale_x', torch.zeros(1))
        self.register_buffer('scale_w', torch.zeros(1))
        self.register_buffer('scale_o', torch.zeros(1))

    def forward(self, input):
        scale_x = ScalerBuffer(self.scale_x)
        running_x = ScalerBuffer(self.running_x)
        if isinstance(input, IQTensor):
            self.is_not_from_iqtensor = False
            if input.bits != self.data_bits:
                input = Requant.apply(
                    input, input.bits, input.scale_data, self.data_bits, self.mode)
            scale_x = ScalerBuffer(input.scale_data)
            running_x = ScalerBuffer(input.running_data)
        running_w = ScalerBuffer(self.running_w)
        running_o = ScalerBuffer(self.running_o)
        scale_w = ScalerBuffer(self.scale_w)
        scale_o = ScalerBuffer(self.scale_o)
        weight = self.weight
        bias = self.bias
        if weight.dtype == torch.float32:
            weight = normalize_weight_with_config(
                self.weight, self.clamp_weight, self.training)
            if self.bias is not None:
                bias = normalize_bias_with_config(
                    self.bias, self.clamp_bias, self.training)
        ret = LinearFunction.apply(input, weight, bias,
                                   self.data_bits, self.parameter_bits,
                                   running_x, running_w, running_o, scale_x, scale_w, scale_o,
                                   self.momentum, self.training, self.prefix, self.dump, self.path,
                                   self.quant_mode, self.o_bits, self.quant, self.is_not_from_iqtensor, self.ahead_relu,
                                   self.clamp_data, self.clamp_weight, self.clamp_bias, self.ahead_sigmoid)
        self.running_x.fill_(running_x())
        self.running_w.fill_(running_w())
        self.running_o.fill_(running_o())
        self.scale_x.fill_(scale_x())
        self.scale_w.fill_(scale_w())
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
        extra_s = ',clamp_data:{clamp_data},clamp_weight:{clamp_weight},clamp_bias:{clamp_bias},ahead_relu:{ahead_relu},ahead_sigmoid:{ahead_sigmoid}'.format(
            **self.__dict__)
        extra_s += ',data_bits:{data_bits},parameter_bits:{parameter_bits},o_bits:{o_bits}'.format(
            **self.__dict__)
        return s+extra_s
