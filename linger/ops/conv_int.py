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


class ConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, kernel_size, padding, stride, dilation, groups, params,
                data_bits, parameter_bits, training, momentum, running_x, running_w, running_o, eval_scale_x, eval_scale_w, eval_scale_o,
                prefix, dump, path, mode, o_bits, quant, ahead_relu, is_not_from_iqtensor, clamp_data, clamp_weight, clamp_bias, ahead_sigmoid):
        scale_o = None

        # venus limits
        # assert input.shape[2] >= weights.shape[2] and input.shape[3] >= weights.shape[
        #     3], f"in Conv2dInt op, input's width >= weight's width && input'height >= weight'height, but you have {input.shape[2]} >= {weights.shape[2]} and {input.shape[3]} >= {weights.shape[3]}"
        assert data_bits in (
            4, 8), f"in Conv2dInt op, data_bits only support 4/8, but you have data_bits {data_bits}"
        assert parameter_bits in (
            4, 8), f"in Conv2dInt op, parameter_bits only support 4/8, but you have parameter_bits {parameter_bits}"
        assert o_bits in (
            4, 8, 16, 32), f"in Conv2dInt op, o_bits only support 4/8/16/32, but you have o_bits {o_bits}"

        channel_in = weights.shape[1]/groups
        # assert not (math.ceil(channel_in/8/stride[1]) * (8*stride[1]) * math.ceil(input.shape[3]/8)*8*1 > 64 * 1024 and channel_in > 512) or not (math.ceil(channel_in/8/stride[1]) * (8*stride[1]) * math.ceil(8/8)*8*input.shape[2] > 64 * 1024 and channel_in >
        #                                                                                                                                           512), f"in Conv2dInt op, the size of the input data after alignment exceed 64KB and channal_in > 512 at the same time is not allowed, but you have (math.ceil({channel_in}/8/{stride[1]}) * (8*{stride[1]}) * math.ceil({input.shape[3]}/8)*8*{1} > 64 * 1024 and {channel_in} > 512) or (math.ceil({channel_in}/8/{stride[1]}) * (8*{stride[1]}) * math.ceil({8}/8)*8*{input.shape[2]} > 64 * 1024 and {channel_in} > 512"

        if training:
            ctx.clamp_data = clamp_data
            ctx.clamp_weight = clamp_weight
            ctx.clamp_bias = clamp_bias
            zero_point = input.zero_point if isinstance(input, IQTensor) else 0
            is_iq_tensor = True if isinstance(input, IQTensor) else False
            ctx.bits = data_bits, parameter_bits, o_bits
            ctx.value = zero_point, is_iq_tensor

            saved_tensors = [input, weights, bias, params]
            if isinstance(input, IQTensor):
                q_input, _, max_value_x = quant.quant(
                    input.data, data_bits, eval_scale_x, mode=QuantMode.QValue, quant_data='input',
                    iq_zero_point=input.zero_point)
                scale_x = eval_scale_x
            else:
                q_input, scale_x, max_value_x = quant.quant(
                    input.data, data_bits, mode=mode, quant_data='input')
                running_x.mul_(1-momentum).add_(momentum*max_value_x)

            q_weights, scale_w, max_value_w = quant.quant(
                weights, parameter_bits, mode=mode, quant_data='weight')
            running_w.mul_(1-momentum).add_(momentum*max_value_w)
            q_input = q_input.float() if data_bits + parameter_bits <= 16 else q_input.double()
            q_weights = q_weights.float() if data_bits + \
                parameter_bits <= 16 else q_weights.double()
            q_outputs = F.conv2d(q_input, q_weights, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups)
            outputs = None
            q_bias = None
            if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                assert mode == QuantMode.QValue, 'castor quant only support QValue and o_bits=None'
                if bias is not None:
                    q_bias = (bias * scale_w * scale_x + 0.5).floor()
                    if data_bits + parameter_bits <= 16:
                        q_bias = q_bias.float().round()
                    else:
                        q_bias = q_bias.double()
                    # 小心计算溢出问题，未做保护，bias不超过2**23-1数值
                    q_outputs += q_bias.reshape(1, -1, 1, 1)
                outputs = quant.dequant(q_outputs, scale_x*scale_w)
            else:
                assert False, "linger only support luna quant."

            # f_input = quant.dequant(q_input, scale_x)
            # f_weights = quant.dequant(q_weights, scale_w)
            # f_bias = None if bias is None else quant.dequant(q_bias, scale_x*scale_w)
            # saved_tensors = [f_input, f_weights, f_bias, params]

            out_tensor = outputs
            scale_o = None
            if o_bits is not None:
                outputs = normalize_data_with_config(outputs, clamp_data)
                out_tensor = outputs
                q_outputs, scale_o, max_value_o = quant.quant(
                    outputs, o_bits, mode=mode, quant_data='conv_output', ahead_relu=ahead_relu)
                outputs = quant.dequant(q_outputs, scale_o)
                running_o.mul_(1-momentum).add_(momentum*max_value_o)
            ctx.scale = scale_x, scale_w, scale_o
            ctx.save_for_backward(*saved_tensors)

        else:
            assert running_x > 0, 'invalid running_x = 0, please finetune training before eval'
            scale_x = None
            scale_w = None
            scale_o = eval_scale_o
            q_weights = None
            if weights.dtype == torch.float32:
                scale_x = ScalerBuffer(quant.running_to_scale(
                    running_x, data_bits, mode=mode))
                q_weights, scale_w, _ = quant.quant(
                    weights, parameter_bits, mode=mode, quant_data='weight')
                scale_w = ScalerBuffer(scale_w)
                if o_bits is not None:
                    assert running_o > 0, 'invalid running_o <= 0, please finetune training'
                    scale_o = ScalerBuffer(quant.running_to_scale(
                        running_o, o_bits, mode=mode))
            else:
                scale_x = eval_scale_x
                scale_w = eval_scale_w
                q_weights = weights.double()
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
            q_weights = q_weights.double()
            q_outputs = F.conv2d(q_input, q_weights, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups)

            outputs = None
            q_bias = None

            if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                assert mode == QuantMode.QValue, 'castor quant only support QValue and o_bits=None'
                if bias is not None:
                    if bias.dtype == torch.float32:
                        q_bias = (bias * scale_w * scale_x + 0.5).floor()
                        if data_bits + parameter_bits <= 16:
                            # 降低bias精度，保持训练一致
                            q_bias = q_bias.float().round().double()
                        else:
                            q_bias = q_bias.double()
                    else:
                        q_bias = bias.double()
                    q_outputs += q_bias.reshape(1, -1, 1, 1)
                outputs = quant.dequant(q_outputs, scale_x*scale_w)
            else:
                assert False, "linger only support luna quant."

            if o_bits is not None:
                if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                    q_outputs, _, _ = quant.quant(
                        outputs, o_bits, scale_o, mode=mode, quant_data='output')
                else:
                    assert False, "linger only support luna quant."
                outputs = quant.dequant(q_outputs, scale_o)
            if dump:
                if q_bias is not None:
                    name_list = ["input", "weights", "bias", "outputs", "q_input",  "q_weights", "q_bias",
                                 "q_outputs", "scale_x", "scale_w", "scale_o", "running_x", "running_w", "running_o"]
                    attr_list = [input, weights, bias, outputs, q_input, q_weights, q_bias, q_outputs,
                                 scale_x.data, scale_w.data, scale_o.data, running_x.data, running_w.data, running_o.data]
                    Dump.dump_file(prefix, ".Conv2dInt.",
                                   zip(name_list, attr_list), path)
                else:
                    name_list = ["input", "weights", "outputs", "q_input",  "q_weights", "q_outputs",
                                 "scale_x", "scale_w", "scale_o", "running_x", "running_w", "running_o"]
                    attr_list = [input, weights, outputs, q_input, q_weights, q_outputs, scale_x.data,
                                 scale_w.data, scale_o.data, running_x.data, running_w.data, running_o.data]
                    Dump.dump_file(prefix, ".Conv2dInt.",
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
        input, weights, bias, params = ctx.saved_tensors

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

        stride = int(params[0]), int(params[1])
        padding = int(params[2]), int(params[3])
        dilation = int(params[4]), int(params[5])
        groups = int(params[6])
        gradBias = None
        # input_clamp = input
        with torch.enable_grad():
            z = F.conv2d(f_input, f_weights, bias,
                         stride, padding, dilation, groups)
            if o_bits is not None:
                z = normalize_data_with_config(z, clamp_data)
            if bias is not None:
                gradInput, gradWeight, gradBias = torch.autograd.grad(
                    z, (f_input, f_weights, bias), gradOutput)
            else:
                gradInput, gradWeight, = torch.autograd.grad(
                    z, (f_input, f_weights), gradOutput)

        return gradInput, gradWeight, gradBias, None, None, None, None, None, None, None, None, None, None, None, None, None,\
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, input, weights, bias, kernel_size, padding, stride, dilation, groups, params,
                 data_bits, parameter_bits, training, momentum, running_x, running_w, running_o, scale_x, scale_w, scale_o,
                 prefix, dump, path, mode, o_bits, quant, ahead_relu, is_not_from_iqtensor, clamp_data, clamp_weight, clamp_bias, ahead_sigmoid):
        op_inner = None
        if is_not_from_iqtensor:
            platform_quant = platform_to_string(
                config.PlatFormQuant.platform_quant)
            op_inner = quantlinear(g, input, scale_x(),
                                   platform_quant, data_bits)
        paddings = padding + padding
        param_dict = {'scale_x_f': scale_x(), 'scale_w_f': scale_w(),
                      'dilations_i': dilation, 'group_i': groups, 'kernel_shape_i': kernel_size, 'pads_i': paddings, 'strides_i': stride,
                      'data_bits_i': data_bits, 'parameter_bits_i': parameter_bits}
        if is_not_from_iqtensor:
            input_list = [op_inner, weights]
        else:
            input_list = [input, weights]
        if bias is not None:
            input_list.append(bias)
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        if o_bits is not None:
            param_dict['scale_o_f'] = scale_o()
            param_dict['o_bits_i'] = o_bits
        param_dict['platform_quant_s'] = platform_quant
        if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
            param_dict['castor_mode_s'] = "luna"
        return g.op("thinker::Conv2dInt", *input_list, **param_dict)


class Conv2dInt(nn.Conv2d, ModuleIntConfig):
    r"""实现Conv2dInt的量化训练与测试，继承自nn.Conv2d,

    Args:
        in_channels out_channels kernel_size stride padding dilation groups bias padding_mode
        与nn.Conv2d一致的参数
        data_bits(int): 输入量化位数
        parameter_bits(int): 参数量化位数
        mode(Enum): 量化方式，支持MaxValue与Qvalue
        o_bits(int, default=None):输出量化位数
        clamp_data(float or None): 针对输出的clamp数值
        clamp_weight(float or None): 针对转为weight的clamp数值
        clamp_bias(float or None): 与clamp_weight一致
        ahead_relu(bool): 是否做融合relu之后的数值统计scale

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 data_bits=8, parameter_bits=8, mode=QuantMode.QValue, o_bits=8,
                 clamp_data=None, clamp_weight=None, clamp_bias=None, ahead_relu=False, ahead_sigmoid=False):
        # venus limits
        if type(kernel_size) == int:
            assert kernel_size in (
                1, 2, 3, 4, 5), f"in Conv2dInt op, kernel size must be 1/2/3/4/5, but you have kernel size {kernel_size}"
        elif type(kernel_size) == tuple:
            assert kernel_size[0] in (1, 2, 3, 4, 5) and kernel_size[1] in (
                1, 2, 3, 4, 5), "in Conv2dInt op, kernel size must be 1/2/3/4/5, but you have kernel size {kernel_size}"

        if type(stride) == int:
            assert stride in (
                1, 2, 4), "in Conv2dInt op, stride size must be 1/2/4, but you have stride size {stride}"
        elif type(stride) == tuple:
            assert stride[0] in (1, 2, 4) and stride[1] in (
                1, 2, 4), "in Conv2dInt op, stride size must be 1/2/4, but you have stride size {stride}"

        if type(padding) == int:
            assert padding in (
                0, 1, 2, 3, 4), "in Conv2dInt op, padding size must be 1/2/4, but you have padding size {padding}"
        elif type(padding) == tuple:
            assert padding[0] in (0, 1, 2, 3, 4) and padding[1] in (
                0, 1, 2, 3, 4), "in Conv2dInt op, padding size must be 1/2/3/4/5, but you have padding size {padding}"

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
        #         1024, f"in Conv2dInt op, kernel must meet the requirements of non-depthwise convolution, but you have math.ceil({in_channels}/8) * 8 * {kernel_size_h} * {kernel_size_w} * math.ceil({out_channels}/2) * 2 <= 32 * 1024"
        # if (groups == in_channels):
        #     assert math.ceil(in_channels/16) * 16 * kernel_size_h * kernel_size_w <= 32 * \
        #         1024, f"in Conv2dInt op, kernel must meet the requirements of depthwise convolution, but you have math.ceil({in_channels}/16) * 16 * {kernel_size_h} * {kernel_size_w} <= 32 * 1024"

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

        nn.Conv2d.__init__(self, in_channels, out_channels,
                           kernel_size, stride, padding, dilation, groups, bias)
        ModuleIntConfig.__init__(
            self, data_bits=data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=o_bits)
        self.params = torch.Tensor([self.stride[0], self.stride[1], self.padding[0],
                                   self.padding[1], self.dilation[0], self.dilation[1], groups])
        self.momentum = 0.1
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.ahead_relu = ahead_relu
        self.ahead_sigmoid = ahead_sigmoid
        self.is_not_from_iqtensor = True
        self.clamp_data = clamp_data
        self.clamp_weight = clamp_weight
        self.clamp_bias = clamp_bias
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
        ret = ConvFunction.apply(input, weight, bias,
                                 self.kernel_size, self.padding, self.stride, self.dilation, self.groups,
                                 self.params, self.data_bits, self.parameter_bits, self.training, self.momentum,
                                 running_x, running_w, running_o, scale_x, scale_w, scale_o,
                                 self.prefix, self.dump, self.path, self.quant_mode, self.o_bits, self.quant, self.ahead_relu, self.is_not_from_iqtensor,
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
        s = nn.Conv2d.extra_repr(self)
        extra_s = ',clamp_data:{clamp_data},clamp_weight:{clamp_weight},clamp_bias:{clamp_bias},ahead_relu:{ahead_relu},ahead_sigmoid:{ahead_sigmoid}'.format(
            **self.__dict__)
        extra_s += ',data_bits:{data_bits},parameter_bits:{parameter_bits},o_bits:{o_bits}'.format(
            **self.__dict__)
        return s+extra_s
