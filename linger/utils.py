import math
import torch
import collections
import numpy as np
from itertools import repeat
from typing import List, Dict, Any
from enum import Enum

LINGER_IGNORE_PAMAMTER = "_linger_ignore_parameter"
LINGER_ACTIVATION_TYPE = "_linger_activation_type"

# 单例模式
class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance

class PlatForm(Enum):
    venus   = 1
    mars    = 2
    arcs    = 3
    venusA  = 4
    jupiter = 5    

class ActivationType(Enum):
    none    = 1
    Relu    = 2
    LeakRelu= 3
    ReluX   = 4
    Sigmoid = 5
    Tanh    = 6

class QuantMode(Enum):
    floor       = 0
    floor_add   = 1
    round       = 2
    ceil        = 3

class QuantStrategy(Enum):
    MSE         = 1
    RANGE_MEAN  = 2
    NSTD        = 3
    HIST        = 4
    KLD         = 5
    TQT         = 6


class FakeQuantMethod(Enum):
    NATIVE   = 1
    CUDA     = 2 
    COMPILE  = 3
    TRITON   = 4
    CUDA_GS  = 5

class QatMethod(Enum):
    TQT     = 1
    MOM     = 2 

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

def get_device(model):
    device = None
    for parameter in model.parameters():
        device = parameter.device
        break
    assert device is not None, 'paramers of model:%s is empty' % (
        model._get_name())
    return device

def quant(x, bits=8, scale=-1, zero_point=0, mode=QuantMode.floor_add):
    bound_value = None
    scale_local = None
    max_abs = None

    zero_point_ = zero_point
    if hasattr(x, 'zero_point'):
        zero_point_ = x.zero_point
    y = x.detach().clone()

    bound_value = math.pow(2, bits - 1) - 1
    if scale > 0:
        scale_local = scale
        max_abs = (bound_value + zero_point_) / scale_local
    else:
        min_x = torch.min(x)
        max_x = torch.max(x)
        if min_x == max_x == 0:
            scale_local = math.pow(2, bits)
        else:
            max_abs = torch.max(-min_x, max_x)
            max_value = round(math.log((bound_value + zero_point_) / max_abs, 2))
            scale_local = math.pow(2, max_value)
            max_abs = (bound_value + zero_point_) / scale_local

    x = y * scale_local
    
    if mode == QuantMode.floor_add:
        x_quant = (x + 0.5).floor()
    elif mode == QuantMode.floor:
        x_quant = x.floor()
    else:
        x_quant = x.round()

    x_quant = x_quant.clamp(-bound_value - 1 + zero_point_, bound_value + zero_point_)    
    x = x_quant.float()
    return x, scale_local

def dequant(x, scale):
    scale_tensor = None
    if isinstance(scale, (float, np.float32)):
        scale_tensor = torch.tensor(
            scale, dtype=torch.float32, device=x.device)
    else:
        scale_tensor = torch.tensor(
            scale.data, dtype=torch.float32, device=x.device)
    return (x / scale_tensor).float()


## for RNN pack
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


__all__ = ['Singleton', 'PlatForm', 'ActivationType', 'QuantMode', 'QuantStrategy', 'FakeQuantMethod', 'QatMethod', 'get_device', 'quant', 'dequant',
           '_unbind', '_unbind_packed', '_slice', 'hx_slice', 'LINGER_IGNORE_PAMAMTER', 'LINGER_ACTIVATION_TYPE']
