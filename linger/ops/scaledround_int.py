from collections import OrderedDict

import torch
import torch.onnx
from torch.onnx import is_in_onnx_export

from ..config import config
from ..quant import Quant
from ..utils import *
from ..utils import Dump
from .iqtensor import from_torch_tensor, platform_to_string, quantlinear
from .ops import ModuleIntConfig


class ScaledRoundLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, momentum, running_x, eval_scale_x, data_bits, training, prefix, dump, path, mode, quant, zero_point):
        if training:
            q_input, scale_x, max_value_x = quant.quant(
                input, data_bits, mode=mode, quant_data='input', iq_zero_point=zero_point)
            running_x.mul_(1-momentum).add_(momentum*max_value_x)
            scale_x = ScalerBuffer(scale_x)
            outputs = quant.dequant(q_input, scale_x)
        else:
            assert running_x != 0, 'invalid running_x=0, please finetune training before eval'
            scale_x = ScalerBuffer(quant.running_to_scale(
                running_x, data_bits, mode=mode, zero_point=zero_point))
            q_input, _, _ = quant.quant(
                input, data_bits, scale_x, mode=mode, quant_data='input', iq_zero_point=zero_point)
            outputs = quant.dequant(q_input, scale_x)

            if dump:
                name_list = ['input', 'q_input',
                             'outputs', 'scale_x', 'running_x']
                attr_list = [input, q_input, outputs, scale_x(), running_x()]
                Dump.dump_file(prefix, "ScaledRoundLayer.",
                               zip(name_list, attr_list), path)
            eval_scale_x.fill_(scale_x())
        if isinstance(scale_x, float):
            return from_torch_tensor(outputs, scale_x, data_bits, zero_point)
        elif isinstance(scale_x, torch.Tensor):
            return from_torch_tensor(outputs, scale_x.item(), data_bits, zero_point)
        else:
            return from_torch_tensor(outputs, scale_x.data, data_bits, zero_point)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output, None, None, None, None, None, None, None, None, None, None, None)

    @staticmethod
    def symbolic(g, input, momentum, running_x, scale_x, data_bits, training, prefix, dump, path, mode, quant, zero_point):
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        return quantlinear(g, input, scale_x(), platform_quant, data_bits)


class ScaledRoundLayer(torch.nn.Module):
    r"""
        将浮点的tensor依据统计的scale转为定点数值

    Args:
        bits: 输出量化的位数
        mode：量化方式，支持MaxValue和Qvalue
    """

    def __init__(self, bits=8, mode=QuantMode.QValue, zero_point=0):
        super(ScaledRoundLayer, self).__init__()

        self.prefix = ""
        self.dump = False
        self.path = ""
        self.data_bits = bits
        self.quant_mode = mode
        self.momentum = 0.1
        self.register_buffer('scale_x', torch.zeros(1))
        self.register_buffer('running_x', torch.zeros(1))
        self.quant = Quant()
        self.zero_point = zero_point

    def forward(self, x, *others):
        assert self.data_bits == 8, 'quant_tensor only support 8bit'
        scale_x = ScalerBuffer(self.scale_x)
        x = ScaledRoundLayerFunction.apply(x, self.momentum, self.running_x, scale_x, self.data_bits, self.training,
                                           self.prefix, self.dump, self.path, self.quant_mode, self.quant, self.zero_point)
        self.scale_x.fill_(scale_x.data)
        if len(others) == 0:
            return x
        else:
            return tuple([x])+others

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]
                              ] = local_metadata = dict(version=self._version)
        if is_in_onnx_export():
            assert self._buffers['running_x'] > 0, 'invalid running_x, please finetune first'
            scale_x = ScalerBuffer(self.quant.running_to_scale(ScalerBuffer(
                self._buffers['running_x']), self.data_bits, mode=self.quant_mode, zero_point=self.zero_point))
            self._buffers['scale_x'].data.fill_(scale_x())

        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix +
                                  name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


__all__ = ['ScaledRoundLayer']
