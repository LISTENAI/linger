import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import config
from ..ops.ops import ModuleIntConfig
from ..utils import Dump, QuantMode, ScalerBuffer
from .iqtensor import (IQTensor, from_torch_tensor, platform_to_string,
                       quantlinear)
from .ops import ModuleIntConfig
from .requant import Requant
from ..quant import Quant


class ReLU6IntFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min_thresh, max_thresh, data_bits, training, momentum, running_x, eval_scale_x, mode, quant, is_not_from_iqtensor, prefix, dump, path):

        scale_x = None
        if training:
            ctx.save_for_backward(input)
            zero_point = input.zero_point if isinstance(input, IQTensor) else 0
            is_iq_tensor = True if isinstance(input, IQTensor) else False
            ctx.value = zero_point, is_iq_tensor
            if isinstance(input, IQTensor):
                q_input, _, max_value_x = quant.quant(input, data_bits, eval_scale_x, mode=QuantMode.QValue, quant_data='input',
                                                      iq_zero_point=input.zero_point)
                scale_x = eval_scale_x
            else:
                q_input, scale_x, max_value_x = quant.quant(
                    input, data_bits, mode=mode, quant_data='input')
                scale_x = scale_x
                running_x.mul_(1-momentum).add_(momentum*max_value_x)
            bound_value = math.pow(2, data_bits-1)-1
            max_thresh_int = round(6 * scale_x)
            max_thresh_int = bound_value if max_thresh_int > bound_value else max_thresh_int
            q_outputs = q_input.clamp(0, max_thresh_int)
            outputs = quant.dequant(q_outputs, scale_x)
            max_thresh.fill_(max_thresh_int)
            ctx.scale = scale_x, data_bits
        else:
            assert running_x > 0, 'invalid running_x <= 0, please fintune first'
            scale_x = None
            scale_x = ScalerBuffer(quant.running_to_scale(
                running_x, data_bits, mode=mode))
            if isinstance(input, IQTensor):
                scale_x = eval_scale_x
                q_input, _, _ = quant.quant(
                    input, data_bits, scale_x, mode=QuantMode.QValue, quant_data='input', iq_zero_point=input.zero_point)
            else:
                q_input, _, _ = quant.quant(
                    input, data_bits, scale_x, mode=mode, quant_data='input')
            bound_value = math.pow(2, data_bits-1)-1
            max_thresh_int = round(6 * scale_x)
            max_thresh_int = bound_value if max_thresh_int > bound_value else max_thresh_int
            q_outputs = q_input.clamp(0, max_thresh_int)
            outputs = quant.dequant(q_outputs, scale_x)
            max_thresh.fill_(max_thresh_int)
            if dump:
                name_list = ["input", "outputs", "q_input",
                             "q_outputs", "scale_x", "running_x"]
                attr_list = [input, outputs, q_input,
                             q_outputs, scale_x.data, running_x.data]
                Dump.dump_file(prefix, ".ReLU6Int.", zip(
                    name_list, attr_list), path)

        if isinstance(scale_x, float):
            return from_torch_tensor(outputs, scale_x, data_bits)
        elif isinstance(scale_x, torch.Tensor):
            return from_torch_tensor(outputs, scale_x.item(), data_bits)
        else:
            return from_torch_tensor(outputs, scale_x.data, data_bits)

    @staticmethod
    def backward(ctx, gradoutput):
        input, = ctx.saved_tensors
        scale_x, data_bits = ctx.scale
        zero_point, is_iq_tensor = ctx.value
        if is_iq_tensor:
            f_input = input.data
        else:
            q_input, _, _ = Quant.quant(
                input.data, data_bits, scale_x, mode=QuantMode.QValue, quant_data='input')
            f_input = Quant.dequant(q_input, scale_x)
        f_input = f_input.detach().clone().requires_grad_(True)
        gradInput = None
        with torch.enable_grad():
            y = F.hardtanh(f_input, 0, 6, False)
            gradInput, = torch.autograd.grad(y, f_input, gradoutput)
        return gradInput, None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, input, min_thresh, max_thresh, data_bits, training, momentum, running_x, scale_x, mode, quant, is_not_from_iqtensor, prefix, dump, path):
        op_inner = None
        if is_not_from_iqtensor:
            platform_quant = platform_to_string(
                config.PlatFormQuant.platform_quant)
            op_inner = quantlinear(g, input, scale_x(),
                                   platform_quant, data_bits)
        if is_not_from_iqtensor:
            input_list = [op_inner, min_thresh, max_thresh]
        else:
            input_list = [input, min_thresh, max_thresh]
        return g.op("Clip", *input_list)


class ReLU6Int(nn.ReLU6, ModuleIntConfig):
    r"""
        实现relu6的clamp按照定点形式计算，而非浮点6

    Args:
        data_bits(int): 输入量化位数
        mode(Enum):量化计算方式，支持MaxValue, QValue
        o_bits(int):输出量化位数 
        scale_x(np.float32): 输出量化scale，与输入scale_x一致
    """

    def __init__(self, data_bits=8, mode=QuantMode.QValue):
        nn.ReLU6.__init__(self)
        ModuleIntConfig.__init__(
            self, data_bits=data_bits, mode=mode, o_bits=None)
        self.momentum = 0.1
        self.is_not_from_iqtensor = True
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.register_buffer('running_x', torch.zeros(1))
        self.register_buffer('scale_x', torch.zeros(1))
        self.register_buffer('min_thresh', torch.tensor(0, dtype=torch.int8))
        self.register_buffer('max_thresh', torch.tensor(0, dtype=torch.int8))

    def forward(self, input):
        scale_x = ScalerBuffer(self.scale_x)
        running_x = ScalerBuffer(self.running_x)
        assert self.data_bits == 8, 'relu6int only support 8bit'
        if isinstance(input, IQTensor):
            self.is_not_from_iqtensor = False
            if input.bits != self.data_bits:
                input = Requant.apply(
                    input, input.bits, input.scale_data, self.data_bits)
            scale_x = ScalerBuffer(input.scale_data)
            running_x = ScalerBuffer(input.running_data)
        output = ReLU6IntFunction.apply(input.contiguous(), self.min_thresh, self.max_thresh, self.data_bits, self.training, self.momentum,
                                        running_x, scale_x, self.quant_mode, self.quant, self.is_not_from_iqtensor, self.prefix, self.dump, self.path)
        self.running_x.fill_(running_x())
        self.scale_x.fill_(scale_x())

        return output

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return ModuleIntConfig.state_dict_global(self, destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
