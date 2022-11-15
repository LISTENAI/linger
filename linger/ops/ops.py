import itertools
from collections import OrderedDict

import torch
import torch.onnx
from torch.onnx import is_in_onnx_export

from ..config import config
from ..quant import (Quant, normalize_bias_with_config,
                     normalize_weight_with_config)
from ..utils import PlatFormQuant, QuantMode, ScalerBuffer


class ModuleIntConfig():
    def __init__(self, data_bits=8, parameter_bits=8, mode=QuantMode.QValue, o_bits=None):
        self.data_bits = data_bits
        self.parameter_bits = parameter_bits
        self.quant_mode = mode
        self.o_bits = o_bits
        self.quant = Quant()

    @staticmethod
    def state_dict_global(module, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]
                              ] = local_metadata = dict(version=module._version)
        if is_in_onnx_export():
            assert module._buffers['running_x'] > 0, 'invalid running_x <= 0, cannot access param before training, layer prefix is: {}'.format(
                prefix)
            scale_x = ScalerBuffer(module._buffers['scale_x'])
            if module.is_not_from_iqtensor:
                scale_x = ScalerBuffer(module.quant.running_to_scale(ScalerBuffer(
                    module._buffers['running_x']), module.data_bits, mode=module.quant_mode))
                module._buffers['scale_x'].data.fill_(scale_x())
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
                                bias_tensor_clamp * scale_w * scale_x + 0.5).floor().float().int()
                        else:
                            module._parameters['bias'].data = (
                                bias_tensor_clamp * scale_w * scale_x + 0.5).floor().int()
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

    @staticmethod
    def _load_from_state_dict_global(module, state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs):
        allow_missing_keys = ['running_w', 'running_x', 'running_y', 'running_o', 'running_i', 'running_iw', 'running_h', 'running_hw', 'running_io', 'running_ho',
                              'running_i_reverse', 'running_iw_reverse', 'running_h_reverse', 'running_hw_reverse', 'running_io_reverse', 'running_ho_reverse',
                              'scale_w', 'scale_x', 'scale_y', 'scale_o', 'scale_i', 'scale_h', 'scale_iw', 'scale_hw', 'sale_io', 'scale_ho', 'scale_i_reverse', 'scale_h_reverse',
                              'scale_iw_reverse', 'scale_hw_reverse', 'sale_io_reverse', 'scale_ho_reverse', 'running_c', 'scale_c', 'scale_cw', 'min_thresh', 'max_thresh',
                              "running_co", "scale_io", "scale_co", "sigmoid_table", "tanh_table", 'scale_o_reverse', 'scale_io_reverse', 'running_o_reverse',
                              "running_q", "running_k", "running_v", "running_attn", "scale_q", "scale_k", "scale_v", "scale_attn", "running_pos", "scale_pos"]
        local_missing_keys = []
        ModuleIntConfig._load_from_state_dict_global_(module, state_dict, prefix, local_metadata, strict,
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

    @staticmethod
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

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue

                if isinstance(input_param, torch.nn.Parameter):
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                    if input_param.dtype == torch.int32:
                        module._parameters[name] = param.int()
                    elif input_param.dtype == torch.int16:
                        module._parameters[name] = param.short()
                    elif input_param.dtype == torch.int8:
                        module._parameters[name] = param.char()

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


__all__ = ['ModuleIntConfig']
