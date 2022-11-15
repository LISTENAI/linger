import logging
import os
from enum import Enum

import numpy as np
import torch

logger = logging.getLogger("linger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s  - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)


class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance


class PlatFormQuant(Enum):
    luna_quant = 1


class QuantMode(Enum):
    QValue = 1


class QuantInfo():
    def __init__(self):
        self.data_bits = 8
        self.parameter_bits = 8
        self.output_bits = None
        self.mode = QuantMode.QValue

    def set_data_bits(self, bits):
        self.data_bits = bits

    def set_parameter_bits(self, bits):
        self.parameter_bits = bits

    def set_output_bits(self, bits):
        self.output_bits = bits

    def set_mode(self, mode):
        self.mode = mode


class ClampInfo():
    def __init__(self):
        self.clamp_weight_value = 8
        self.clamp_bias_value = 8
        self.clamp_output_value = None

    def set_clamp_weight_value(self, value):
        self.clamp_weight_value = value

    def set_clamp_bias_value(self, value):
        self.clamp_bias_value = value

    def set_clamp_output_value(self, value):
        self.clamp_output_value = value


def get_device(model):
    device = None
    for parameter in model.parameters():
        device = parameter.device
        break
    assert device is not None, 'paramers of model:%s is empty' % (
        model._get_name())
    return device


def get_max_value(input):
    max_value = -1
    if isinstance(input, list):
        input_tmp = [data.detach() for data in input]
        for data in input_tmp:
            tmp = torch.max(torch.abs(data))
            max_value = max_value if max_value > tmp else tmp
    else:
        input_tmp = input.detach()
        tmp = torch.max(torch.abs(input_tmp))
        max_value = max_value if max_value > tmp else tmp
    return max_value


class Dump():
    @staticmethod
    def dump_file(header, c_name, print_list, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for name, item in print_list:
            if(isinstance(item, torch.Tensor)):
                np.savetxt(os.path.join(file_path, header+c_name+name),
                           item.detach().reshape(-1).cpu().numpy(), fmt="%.6f")


def _unbind(src_tensor):
    dim_0 = src_tensor.size(0)
    nums = dim_0.item() if isinstance(dim_0, torch.Tensor) else dim_0
    sub_tensor_list = [each.squeeze(0)
                       for each in src_tensor.split([1]*nums, dim=0)]
    return sub_tensor_list


def _unbind_packed(packed_tensor, batch_sizes):
    offset = 0
    tensor_list = []
    batch_size_list = [each.item() for each in batch_sizes]
    for batch_len in batch_size_list:
        t = packed_tensor.narrow(0, offset, batch_len)
        offset += batch_len
        tensor_list.append(t)
    return tensor_list, batch_size_list


def _slice(input, start, end):
    if isinstance(input, (tuple)):
        return tuple([each.narrow(0, start, end-start) for each in input])
    else:
        return input.narrow(0, start, end-start)


def hx_slice(input_hidden, cur_hidden, last_batch_size, cur_batch_size):
    if input_hidden is None:  # forward: slice cur_hidden
        assert cur_batch_size < last_batch_size, 'error: forward batch_sizes is not desc order'
        return _slice(cur_hidden, 0, cur_batch_size)
    else:  # reverse: add cur_hidden and sliced input_hidden
        assert cur_batch_size > last_batch_size, 'error: reverse batch_sizes is not asc order'
        slice_hidden = _slice(input_hidden, last_batch_size, cur_batch_size)
        if isinstance(slice_hidden, tuple):
            new_hidden = [torch.cat((cur_e, slice_e), 0)
                          for cur_e, slice_e in zip(cur_hidden, slice_hidden)]
            return tuple(new_hidden)
        return torch.cat((cur_hidden, slice_hidden), 0)


def qshift_round_away_from_zero(x, ):
    x_mask = torch.ones_like(x)
    x_mask[x < 0] = -1
    x_abs = (x.abs() + 0.5).floor()
    x = x_abs * x_mask
    return x


class ScalerBuffer():
    def __init__(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, ScalerBuffer):
            value = value.data
        self.value = np.float32(value)

    def __repr__(self):
        return self.value

    def __str__(self):
        return str(self.value)

    def fill_(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        elif isinstance(x, ScalerBuffer):
            x = x.data
        self.value = np.float32(x)

    def __add__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        elif isinstance(x, ScalerBuffer):
            x = x.data
        return ScalerBuffer(self.value + np.float32(x))

    def __radd__(self, x):
        return x + self.value

    def add_(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        elif isinstance(x, ScalerBuffer):
            x = x.data
        self.value = self.value + np.float32(x)
        return self

    def __mul__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        elif isinstance(x, ScalerBuffer):
            x = x.data
        return ScalerBuffer(self.value * np.float32(x))

    def __rmul__(self, x):
        return x * self.value

    def mul_(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        elif isinstance(x, ScalerBuffer):
            x = x.data
        self.value = self.value * np.float32(x)
        return self

    def __truediv__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        elif isinstance(x, ScalerBuffer):
            x = x.data
        return ScalerBuffer(self.value / np.float32(x))

    def __rtruediv__(self, x):
        return x/self.value

    def __gt__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        elif isinstance(x, ScalerBuffer):
            x = x.data
        return True if self.value > np.float32(x) else False

    def __eq__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        elif isinstance(x, ScalerBuffer):
            x = x.data
        return True if self.value == np.float32(x) else False

    def __ne__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.item()
        elif isinstance(x, ScalerBuffer):
            x = x.data
        return True if self.value != np.float32(x) else False

    def __call__(self):
        return self.value

    @property
    def data(self):
        return self.value


__all__ = ['Singleton', 'PlatFormQuant', 'QuantMode', 'QuantInfo', 'ClampInfo', 'get_max_value', 'ClampInfo', 'get_device',
           '_unbind', '_unbind_packed', '_slice', 'hx_slice', 'qshift_round_away_from_zero', 'ScalerBuffer', 'logger']
