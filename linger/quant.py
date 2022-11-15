import math

import lingerext
import numpy as np
import torch
import torch.nn.functional as F
import torch.onnx.symbolic_helper as sym_help

from linger.config import config
from linger.utils import PlatFormQuant, QuantMode, ScalerBuffer
epsilon = 1e-10


class Quant():
    @staticmethod
    def quant(x, bits=8, scale=-1, mode=QuantMode.QValue, *, quant_data='weight', ahead_relu=False, iq_zero_point=0):
        bound_value = None
        scale_local = None
        max_abs = None
        assert quant_data == 'input' or quant_data == 'output' or quant_data == 'weight' or quant_data == 'hidden' or quant_data == 'conv_output', 'invalid quant_data, please confirm'
        # assert drop_rate >= 0.0, 'invalid drop_rate param'
        zero_point = iq_zero_point
        if hasattr(x, 'zero_point'):
            zero_point = x.zero_point
        y = x.detach().clone()
        if (ahead_relu and quant_data == 'output') or (ahead_relu and quant_data == 'conv_output'):
            x = F.relu(x)
        if mode == QuantMode.QValue:
            max_abs = 0
            bound_value = math.pow(2, bits-1)-1
            if scale > 0:
                scale_local = scale
                max_abs = (bound_value+zero_point) / scale_local
            else:
                min_x = torch.min(x)
                max_x = torch.max(x)
                if min_x == max_x == 0:
                    scale_local = math.pow(2, bits)
                else:
                    max_abs = torch.max(-min_x, max_x)
                    max_value = round(
                        math.log((bound_value+zero_point) / max_abs, 2))
                    scale_local = math.pow(2, max_value)
                    max_abs = (bound_value+zero_point) / scale_local
        else:
            print('Error Quant Mode!!!')

        scale_local = ScalerBuffer(scale_local)
        x = y * scale_local
        
        #not iqtensor or test
        if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
            x_quant = (x + 0.5).floor()
        else:
            assert False, "linger only support luna quant."
        x_quant.clamp_(-bound_value-1+zero_point, bound_value+zero_point)
        # drop_rate means quant percent, if drop_rate=1.0, all quant
        
        x = x_quant.float()
        return x, scale_local, max_abs

    @staticmethod
    def dequant(x, scale):
        scale_tensor = None
        if isinstance(scale, (float, np.float32)):
            scale_tensor = torch.tensor(
                scale, dtype=torch.float32, device=x.device)
        else:
            scale_tensor = torch.tensor(
                scale.data, dtype=torch.float32, device=x.device)
        return (x/scale_tensor).float()

    @staticmethod
    def running_to_scale(running_data, bits, mode=QuantMode.QValue, zero_point=0):
        running_data = ScalerBuffer(running_data)
        scale_data = None
        bound_value = math.pow(2, bits-1)-1
        if mode == QuantMode.QValue:
            max_value = round(math.log(
                (bound_value+zero_point)/running_data(), 2)) if running_data() != 0 else 0.0
            scale_data = math.pow(2, max_value)
        scale_data = ScalerBuffer(scale_data)
        return scale_data

class ClipGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, clip_threshold):
        ctx.clip_threshold = clip_threshold
        return data

    @staticmethod
    def backward(ctx, grad_output):
        clip_threshold = ctx.clip_threshold
        grad_output = grad_output.clamp(-clip_threshold, clip_threshold)
        return grad_output, None


class NormalizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, bound, training=True, is_weight=True):
        data, temp_loc1, temp_loc2 = lingerext.normalize_function_forward(
            data, float(bound), training, is_weight)
        ctx.save_for_backward(temp_loc1, temp_loc2)
        return data

    @staticmethod
    def backward(ctx, grad_output):
        temp_loc1, temp_loc2 = ctx.saved_tensors
        grad_output = lingerext.normalize_function_backward(
            grad_output, temp_loc1, temp_loc2)

        return grad_output, None, None, None

    @staticmethod
    def symbolic(g, input, bound, training, is_weight=True):
        dtype = input.type().scalarType()
        if dtype is None:
            dtype = 6  # float
        else:
            dtype = sym_help.scalar_type_to_onnx.index(
                sym_help.cast_pytorch_to_onnx[dtype])
        min_val = g.op("Constant", value_t=torch.tensor(-bound,
                       dtype=sym_help.scalar_type_to_pytorch_type[dtype]))
        max_val = g.op("Constant", value_t=torch.tensor(
            bound, dtype=sym_help.scalar_type_to_pytorch_type[dtype]))
        return g.op("Clip", *[input, min_val, max_val])


def normalize_data_with_config(input, normalize_data):
    if normalize_data is not None:
        input_clamp_data = normalize_data
        input = input.clamp(-input_clamp_data, input_clamp_data)
    return input


def normalize_weight_with_config(weights, normalize_weight, training=False):
    if normalize_weight is not None:
        weights = NormalizeFunction.apply(weights, normalize_weight, training)
    return weights


def normalize_bias_with_config(bias, normalize_bias, training=False):
    if normalize_bias is not None:
        bias = NormalizeFunction.apply(bias, normalize_bias, training)
    return bias
