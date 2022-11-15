import math
import itertools
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.onnx import is_in_onnx_export

from ..config import config
from ..quant import (Quant, normalize_bias_with_config,
                     normalize_data_with_config, normalize_weight_with_config)
from ..utils import Dump, PlatFormQuant, QuantMode, ScalerBuffer
from .iqtensor import (IQTensor, from_torch_tensor, platform_to_string,
                       quantlinear)
from .ops import ModuleIntConfig
from .requant import Requant

import lingerext

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, normalized_shape,
                training, momentum, eps,
                running_x, running_normal, running_w, running_o,
                eval_scale_x, eval_scale_normal, eval_scale_w, eval_scale_o,
                data_bits, parameter_bits, prefix, dump,
                path, mode, o_bits, quant, is_not_from_iqtensor,
                ahead_relu, clamp_data, clamp_weight, clamp_bias):
        N = 1
        for dim_size in normalized_shape:
            N *= dim_size
        if len(weights.shape) == 1:
            w_shape = (-1)
        elif len(weights.shape) == 2:
            w_shape = (-2, -1)
        elif len(weights.shape) == 3:
            w_shape = (-3, -2, -1)
        elif len(weights.shape) == 4:
            w_shape = (-4, -3, -2, -1)
        else:
            assert False, f"weight.shape=={weights.shape}, please check your LayerNorm definition."
        H, W = input.shape[-2], input.shape[-1]
        scale_x = None
        scale_w = None
        scale_o = None
        if training:
            ctx.clamp_data = clamp_data
            ctx.clamp_weight = clamp_weight
            ctx.clamp_bias = clamp_bias
            ctx.eps = eps
            ctx.bits = data_bits, parameter_bits, o_bits
            zero_point = input.zero_point if isinstance(input, IQTensor) else 0
            is_iq_tensor = True if isinstance(input, IQTensor) else False
            ctx.value = zero_point, is_iq_tensor
            ctx.N = N
            ctx.w_shape = w_shape
            saved_tensors = [input, weights, bias]

            # x
            if isinstance(input, IQTensor):
                q_input, _, max_value_x = quant.quant(
                    input.data, data_bits, eval_scale_x, mode=QuantMode.QValue, quant_data='input', iq_zero_point=input.zero_point)
                scale_x = eval_scale_x
            else:
                q_input, scale_x, max_value_x = quant.quant(
                    input.data, data_bits, mode=mode, quant_data='input')
                running_x.mul_(1-momentum).add_(momentum*max_value_x)
            # q_input = q_input.float() if data_bits + parameter_bits <= 16 else q_input.double()
            q_input = q_input.contiguous().int()
            sum_x = q_input.clone().sum(w_shape, keepdim=True)
            sum_x2 = q_input.clone().pow(2).sum(w_shape, keepdim=True)
            denominator = N * sum_x2 - sum_x * sum_x
            denominator = denominator.contiguous().int()
            denominator, deno_scale= lingerext.luna_layernormint(denominator.contiguous(), math.log2(scale_x.data))
            deno_scale_f = torch.pow(2, deno_scale.float())
            q_x_normal = q_input * N
            q_x_normal = q_x_normal - sum_x
            q_x_normal = q_x_normal * denominator
            q_x_normal = (q_x_normal * deno_scale_f + 0.5).floor().int()
            q_x_normal.clamp_(-2**15, 2**15-1)
            eval_scale_normal.fill_(2**12)

            q_weights, scale_w, max_value_w = quant.quant(
                weights, parameter_bits, mode=mode, quant_data='weight')
            running_w.mul_(1-momentum).add_(momentum*max_value_w)

            q_outputs = q_x_normal * q_weights

            if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                q_bias = (bias * scale_w * eval_scale_normal + 0.5).floor()
                q_bias.clamp_(-2**31, 2**31-1)
                q_outputs = q_outputs + q_bias
                q_outputs.clamp_(-2**31, 2**31-1)
                outputs = quant.dequant(q_outputs, eval_scale_normal*scale_w)
            else:
                assert False, "linger only support luna quant."

            out_tensor = outputs
            scale_o = None
            if o_bits is not None:
                outputs = normalize_data_with_config(outputs, clamp_data)
                out_tensor = outputs
                q_outputs, scale_o, max_value_o = quant.quant(
                    outputs, o_bits, mode=mode, quant_data='output', ahead_relu=ahead_relu)
                running_o.mul_(1-momentum).add_(momentum*max_value_o)
                outputs = quant.dequant(q_outputs, scale_o)
            saved_tensors += [out_tensor]
            ctx.scale = scale_x, eval_scale_normal, scale_w, scale_o
            ctx.save_for_backward(*saved_tensors)

        else:
            assert running_x > 0, 'invalid running_x <= 0, please finetune training before eval'
            eval_scale_normal.fill_(2**12)
            if weights.dtype == torch.float32:
                scale_x = ScalerBuffer(quant.running_to_scale(
                    running_x, data_bits, mode=mode))
                if o_bits is not None:
                    scale_o = ScalerBuffer(quant.running_to_scale(
                        running_o, o_bits, mode=mode))
                weigths = normalize_weight_with_config(
                    weights, clamp_weight, False)
                bias = normalize_bias_with_config(bias, clamp_bias, False)
                q_weights, scale_w, _ = quant.quant(
                    weigths, parameter_bits, mode=mode, quant_data='weight')
                q_bias = None
                if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                    if bias.dtype == torch.float32:
                        q_bias = (bias * scale_w * eval_scale_normal + 0.5).floor()
                    else:
                        q_bias = bias.contiguous()                    
                else:
                    assert False, 'linger only support luna quant.'
            else:
                scale_x = eval_scale_x
                scale_normal = eval_scale_normal
                scale_w = eval_scale_w
                scale_o = eval_scale_o
                q_weights = weights.contiguous()
                q_bias = bias.contiguous()
            q_bias.clamp_(-2**31, 2**31-1)
            if isinstance(input, IQTensor):
                scale_x = eval_scale_x
                q_input, _, _ = quant.quant(
                    input.data, data_bits, scale_x, mode=QuantMode.QValue, quant_data='input', iq_zero_point=input.zero_point)
            else:
                q_input, _, _ = quant.quant(
                    input.data, data_bits, scale_x, mode=mode, quant_data='input')
            q_input = q_input.contiguous().int()
            q_weights = q_weights.contiguous().int()
            sum_x = q_input.clone().sum(w_shape, keepdim=True)
            sum_x2 = q_input.clone().pow(2).sum(w_shape, keepdim=True)
            denominator = N * sum_x2 - sum_x * sum_x
            denominator = denominator.contiguous().int()
            denominator, deno_scale= lingerext.luna_layernormint(denominator.contiguous(), math.log2(scale_x.data))
            deno_scale_f = torch.pow(2, deno_scale.float())
            q_x_normal = q_input * N
            q_x_normal = q_x_normal - sum_x
            q_x_normal = q_x_normal * denominator
            q_x_normal = (q_x_normal * deno_scale_f + 0.5).floor().int()
            q_x_normal.clamp_(-2**15, 2**15-1)
            q_outputs = q_x_normal * q_weights + q_bias
            q_outputs.clamp_(-2**31, 2**31-1)
            outputs = quant.dequant(q_outputs, eval_scale_normal * scale_w)

            if o_bits is not None:
                if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                    q_outputs, _, _ = quant.quant(
                        outputs, o_bits, scale_o, mode=mode, quant_data='output')
                else:
                    assert False, "linger only support luna quant."
                outputs = quant.dequant(q_outputs, scale_o)

            if dump:
                name_list = ["input", "weights", "bias", "q_weights",  "q_input", "q_input_normal", "denominator", "q_bias", "q_outputs",
                             "scale_normal", "scale_x", "scle_w", "scale_o"]
                attr_list = [input, weights, bias, q_weights, q_input, q_x_normal, denominator, q_bias, q_outputs,
                             eval_scale_normal.data, scale_x.data, scale_w.data, scale_o.data]
                Dump.dump_file(prefix, ".LayerNormInt.",
                               zip(name_list, attr_list), path)

        if isinstance(scale_o, float):
            return from_torch_tensor(outputs, scale_o, o_bits)
        elif isinstance(scale_o, torch.Tensor):
            return from_torch_tensor(outputs, scale_o.item(), o_bits)
        else:
            return from_torch_tensor(outputs, scale_o.data, o_bits)

    @staticmethod
    def backward(ctx, gradOutput):
        input, weights, bias, outputs = ctx.saved_tensors
        clamp_data = ctx.clamp_data
        data_bits, parameter_bits, o_bits = ctx.bits
        scale_x, scale_normal, scale_w, scale_o = ctx.scale
        zero_point, is_iq_tensor = ctx.value
        N = ctx.N
        w_shape = ctx.w_shape
        eps = ctx.eps

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

        with torch.enable_grad():
            mean = f_input.clone().sum(w_shape, keepdim=True) / N
            var = input.clone().pow(2).sum(w_shape, keepdim=True) / N - \
                (input.clone().sum(w_shape, keepdim=True) / N).pow(2)
            var = 1/torch.sqrt(var + eps)
            var = torch.clamp(var, min=0.0)
            f_input_normal = (input - mean) * var
            z = f_weights * f_input_normal + bias
            z = normalize_data_with_config(z, clamp_data)
            gradInput, gradWeight, gradBias = torch.autograd.grad(
                z, (f_input, f_weights, bias), gradOutput)

        return gradInput, gradWeight, gradBias, None, \
            None, None, None, None, None, \
            None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, \
            None, None, None, None, None, \
            None, None, None, None

    @staticmethod
    def symbolic(g, input, weights, bias, normalized_shape,
                 training, momentum, eps,
                 running_x, running_normal, running_w, running_o,
                 scale_x, scale_normal, scale_w, scale_o,
                 data_bits, parameter_bits, prefix, dump,
                 path, mode, o_bits, quant, is_not_from_iqtensor,
                 ahead_relu, clamp_data, clamp_weight, clamp_bias):
        op_inner = None
        if is_not_from_iqtensor:
            platform_quant = platform_to_string(
                config.PlatFormQuant.platform_quant)
            op_inner = quantlinear(g, input, scale_x(),
                                   platform_quant, data_bits)
        param_dict = {'data_bits_i': data_bits, 'parameter_bits_i': parameter_bits, 'o_bits_i': o_bits,
                      'scale_x_f': scale_x(), 'scale_w_f': scale_w(), 'scale_o_f': scale_o()}

        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict['platform_quant_s'] = platform_quant
        if is_not_from_iqtensor:
            input_list = [op_inner, weights, bias]
        else:
            input_list = [input, weights, bias]
        return g.op("thinker::LayerNormInt", *input_list, **param_dict)


class LayerNormInt(nn.LayerNorm, ModuleIntConfig):
    r"""实现LayerNormInt的量化训练与测试，继承自nn.LayerNorm,

    Args:
        num_features eps momentum affine track_running_stats
        标准nn.LayerNorm的参数
        data_bits(int): 输入量化位数
        parameter_bits(int): 参数量化位数
        mode(Enum): 量化方式，支持MaxValue与Qvalue
        o_bits(int, default=None):输出量化位数
        clamp_data(float or None): 针对输出的clamp数值
        clamp_weight(float or None): 针对转为乘加操作之后的weight与bias的clamp数值，此处不使用
        clamp_bias(float or None): 与clamp_weight一致
        ahead_relu(bool): 是否做融合relu之后的数值统计scale

    Examples:
        test/test_layernorm_int.py

    """

    def __init__(self, normalized_shape, eps=1e-05, momentum=0.1, elementwise_affine=True, data_bits=8, parameter_bits=16, mode=QuantMode.QValue, o_bits=8,
                 clamp_data=None, clamp_weight=None, clamp_bias=None, ahead_relu=False):
        parameter_bits = 16
        nn.LayerNorm.__init__(self, normalized_shape, eps, elementwise_affine)
        ModuleIntConfig.__init__(
            self, data_bits=data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=o_bits)
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.is_not_from_iqtensor = True
        self.ahead_relu = ahead_relu
        self.clamp_data = clamp_data
        self.clamp_weight = clamp_weight
        self.clamp_bias = clamp_bias
        self.mode = mode
        self.momentum = momentum
        self.register_buffer('running_x', torch.zeros(1))
        self.register_buffer('running_normal', torch.zeros(1))
        self.register_buffer('running_w', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))
        self.register_buffer('scale_x', torch.zeros(1))
        self.register_buffer('scale_normal', torch.zeros(1))
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
        running_normal = ScalerBuffer(self.running_normal)
        running_w = ScalerBuffer(self.running_w)
        running_o = ScalerBuffer(self.running_o)
        scale_normal = ScalerBuffer(self.scale_normal)
        scale_w = ScalerBuffer(self.scale_w)
        scale_o = ScalerBuffer(self.scale_o)
        momentum = 0.1
        weight = self.weight
        bias = self.bias
        if self.weight.dtype == torch.float32:
            weight = normalize_weight_with_config(
                self.weight, self.clamp_weight, self.training)
            if self.bias is not None:
                bias = normalize_bias_with_config(
                    self.bias, self.clamp_bias, self.training)
        
        ret = LayerNormFunction.apply(input, weight, bias, self.normalized_shape,
                                      self.training and self.elementwise_affine, momentum, self.eps,
                                      running_x, running_normal, running_w, running_o,
                                      scale_x, scale_normal, scale_w, scale_o,
                                      self.data_bits, self.parameter_bits, self.prefix, self.dump,
                                      self.path, self.quant_mode, self.o_bits, self.quant, self.is_not_from_iqtensor,
                                      self.ahead_relu, self.clamp_data, self.clamp_weight, self.clamp_bias)
        self.running_x.fill_(running_x())
        self.running_normal.fill_(running_normal())
        self.running_w.fill_(running_w())
        self.running_o.fill_(running_o())
        self.scale_x.fill_(scale_x())
        self.scale_normal.fill_(scale_normal())
        self.scale_w.fill_(scale_w())
        self.scale_o.fill_(scale_o())
        return ret

    def state_dict(module, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]
                              ] = local_metadata = dict(version=module._version)
        scale_x = ScalerBuffer(module._buffers['scale_x'])
        scale_normal = ScalerBuffer(module._buffers['scale_normal'])
        if is_in_onnx_export():
            assert module._buffers['running_x'] > 0, 'invalid running_x <= 0, cannot access param before training, layer prefix is: {}'.format(
                prefix)
            if module.is_not_from_iqtensor:
                scale_x = ScalerBuffer(module.quant.running_to_scale(ScalerBuffer(
                    module._buffers['running_x']), module.data_bits, mode=module.quant_mode))
                module._buffers['scale_x'].data.fill_(scale_x())
            # scale_normal = ScalerBuffer(module.quant.running_to_scale(ScalerBuffer(
            #     module._buffers['running_normal']), module.data_bits, mode=module.quant_mode))
            scale_normal.fill_(2**12)
            module._buffers['scale_normal'].data.fill_(scale_normal())
            if module.o_bits is not None:
                scale_o = ScalerBuffer(module.quant.running_to_scale(ScalerBuffer(
                    module._buffers['running_o']), module.o_bits, mode=module.quant_mode))
                module._buffers['scale_o'].data.fill_(scale_o())
        if 'scale_w' in module._buffers and module._parameters['weight'].dtype == torch.float:
            weight_tensor = module._parameters['weight']
            weight_tensor_clamp = None
            bias_tensor_clamp = None
            if hasattr(module, 'clamp_weight'):
                weight_tensor_clamp = normalize_weight_with_config(
                    weight_tensor, module.clamp_weight, False)
            else:
                weight_tensor_clamp = weight_tensor
            weight_tensor.data = weight_tensor_clamp
            if module.bias is not None:
                bias_tensor = module._parameters['bias']
                # bias_temp = None
                if hasattr(module, 'clamp_bias'):
                    bias_tensor_clamp = normalize_bias_with_config(
                        bias_tensor, module.clamp_bias, False)
                else:
                    bias_tensor_clamp = bias_tensor
                bias_tensor.data = bias_tensor_clamp
            if is_in_onnx_export():
                weight_temp, scale_w, _ = module.quant.quant(
                    weight_tensor_clamp, module.parameter_bits, mode=module.quant_mode)
                scale_w = ScalerBuffer(scale_w)
                module._buffers['scale_w'].data.fill_(scale_w())

                if module.parameter_bits <= 8:
                    weight_tensor.data = weight_temp.char()
                    weight_tensor.char()
                elif module.parameter_bits <= 16:
                    weight_tensor.data = weight_temp.short()
                    weight_tensor.short()
                else:
                    weight_tensor.data = weight_temp.int()
                    weight_tensor.int()
                if module.bias is not None:
                    bias_tensor_clamp = module._parameters['bias']
                    if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                        assert module.quant_mode == QuantMode.QValue, 'luna_quant only support Qvalue and o_bits=None'
                        if module.data_bits + module.parameter_bits <= 16:
                            module._parameters['bias'].data = (
                                bias_tensor_clamp * scale_w * scale_normal + 0.5).floor().float().int()
                        else:
                            module._parameters['bias'].data = (
                                bias_tensor_clamp * scale_w * scale_normal + 0.5).floor().int()
                        module._parameters['bias'].int()
                    else:
                        assert False, "linger only support luna quant."
        module._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in module._modules.items():
            if module is not None:
                module.state_dict(destination, prefix +
                                  name + '.', keep_vars=keep_vars)
        for hook in module._state_dict_hooks.values():
            hook_result = hook(module, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def _load_from_state_dict(module, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        allow_missing_keys = ['running_w', 'running_x', 'running_normal', 'running_o',
                              'scale_x', 'running_normal', 'scale_w', 'scale_o']
        local_missing_keys = []
        module._load_from_state_dict_global_(state_dict, prefix, local_metadata, strict,
                                             local_missing_keys, unexpected_keys, error_msgs)
        matched = True
        fake_missing_keys = []
        for k_local in local_missing_keys:
            if k_local.replace(prefix, '', 1) not in allow_missing_keys:
                matched = False
                fake_missing_keys.append(k_local)
        if matched:
            local_missing_keys = []
        else:
            local_missing_keys = fake_missing_keys
        missing_keys += local_missing_keys

    def _load_from_state_dict_global_(module, state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs):
        for hook in module._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs)
        local_name_params = itertools.chain(
            module._parameters.items(), module._buffers.items())
        local_state = {k: v.data for k,
                       v in local_name_params if v is not None}
        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue

                if isinstance(input_param, torch.nn.Parameter):
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                    if input_param.dtype == torch.int32 or input_param.dtype == torch.int8 or input_param.dtype == torch.int16:
                        module._parameters[name] = param.int()

                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)
        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]
                    if input_name not in module._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def extra_repr(self):
        s = nn.LayerNorm.extra_repr(self)
        extra_s = ' ,clamp_data:{clamp_data},clamp_weight:{clamp_weight},clamp_bias:{clamp_bias},ahead_relu:{ahead_relu}'.format(
            **self.__dict__)
        extra_s += ',data_bits:{data_bits}, parameter_bits:{parameter_bits}, o_bits:{o_bits}'.format(
            **self.__dict__)
        return s+extra_s
