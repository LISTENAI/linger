import math
from collections import OrderedDict

import numpy as np
import torch
import torch._C as _C
import torch.onnx.symbolic_helper as sym_help
from torch.onnx import is_in_onnx_export

from torch.onnx.symbolic_opset9 import flatten as onnx_syms_flatten
from torch.onnx.symbolic_opset9 import permute as onnx_syms_permute
from torch.onnx.symbolic_opset9 import prim_ConstantChunk as onnx_sym_chunk
from torch.onnx.symbolic_opset9 import reshape_as as onnx_syms_reshape_as
from torch.onnx.symbolic_opset9 import squeeze as onnx_syms_squeeze
from torch.onnx.symbolic_opset9 import transpose as onnx_syms_transpose
from torch.onnx.symbolic_opset9 import unsqueeze as onnx_syms_unsqueeze
from torch.onnx.symbolic_opset9 import view as onnx_syms_view
from torch.onnx.symbolic_opset10 import flip as onnx_syms_flip

from ..config import config
from ..ops.ops import ModuleIntConfig
from ..ops.ops_names import (LINGER_IQTENSOR_LAYER_COUNTER,
                             LINGER_MIX_INT8_MANUAL_ROUND_LAYERS, LINGER_MODE)
from ..quant import Quant
from ..utils import Dump, PlatFormQuant, QuantMode, ScalerBuffer
from .module_self import get_current_module


def platform_to_string(platform_quant):
    if platform_quant == PlatFormQuant.luna_quant:
        return "luna_quant"


def quantlinear(g, input,  scale_x, platform_quant, data_bits=8):
    return g.op("thinker::Quant", input, data_bits_i=data_bits, scale_x_f=scale_x, platform_quant_s=platform_quant)


def dequantlinear(g, input, scale_x):
    return g.op("thinker::Dequant", input, scale_x_f=scale_x)


class Quant2IQTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, data_bits, mode, quant_data):
        q_outputs, scale, _ = Quant.quant(
            data, data_bits, mode=mode, quant_data=quant_data)
        outputs = Quant.dequant(q_outputs, scale)
        outputs = from_torch_tensor(outputs, scale(), data_bits)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class Convert2IQTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t):
        s = IQTensor()
        s.data = t.data
        return s

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput

    @staticmethod
    def symbolic(g, input):
        return g.op("Identity", input)


def from_torch_tensor(t, scale, bits, zero_point=0, running=None):
    r"""把torch tensor 转换成IQTensor

    Args:
        t(torch.Tensor):需要转换的torch tensor
        scale(float):IQTensor 的scale_data
        bits(int):IQTensor的精度
        zero_point(int): 控制uint8->int8偏移量
    returns:
        转换后的IQTensor

    Notes:
        这次转换是有grad信息，会被记录到图中
    """
    if scale is None:
        return t
    s = Convert2IQTensor.apply(t)
    s.scale_data = scale
    s.bits = bits
    s.zero_point = zero_point
    s.running_data = running
    if s.running_data is None:
        bound_value = math.pow(2, bits-1)-1
        s.running_data = (bound_value+zero_point) / s.scale_data
    return s


class iqView(torch.autograd.Function):
    @staticmethod
    def forward(self, input, *size):
        y = super(input.__class__, input,).view(*size)
        self.num_in = len(size)
        self.size = input.size()
        return from_torch_tensor(y, input.scale_data, input.bits, input.zero_point)

    @staticmethod
    def backward(self, s):
        size = self.size
        l = [None for i in range(self.num_in)]
        ret = [s.contiguous().view(size)] + l
        return tuple(ret)

    @staticmethod
    def symbolic(g, input, *size):
        if isinstance(size[0], tuple):
            size = size[0]
        CValue_list = [t for t in size if sym_help._is_value(t)]
        if len(CValue_list) != 0:
            unsqueezed = []
            for t in size:
                if sym_help._is_value(t):
                    unsqueezed.append(g.op("Unsqueeze", t, axes_i=[0]))
                else:
                    const_op = g.op("Constant", value_t=torch.tensor(t))
                    unsqueezed.append(g.op("Unsqueeze", const_op, axes_i=[0]))
            size = g.op("Concat", *unsqueezed, axis_i=0)
        return onnx_syms_view(g, input, size)


class iqReshape(torch.autograd.Function):
    @staticmethod
    def forward(self, input, *size):
        y = super(input.__class__, input,).reshape(*size)
        self.num_in = len(size)
        self.size = input.size()
        return from_torch_tensor(y, input.scale_data, input.bits, input.zero_point)

    @staticmethod
    def backward(self, s):
        size = self.size
        l = [None for i in range(self.num_in)]
        ret = [s.reshape(size)] + l
        return tuple(ret)

    @staticmethod
    def symbolic(g, input, *size):
        if isinstance(size[0], tuple):
            size = size[0]
        CValue_list = [t for t in size if sym_help._is_value(t)]
        if len(CValue_list) != 0:
            unsqueezed = []
            for t in size:
                if sym_help._is_value(t):
                    unsqueezed.append(g.op("Unsqueeze", t, axes_i=[0]))
                else:
                    const_op = g.op("Constant", value_t=torch.tensor(t))
                    unsqueezed.append(g.op("Unsqueeze", const_op, axes_i=[0]))
            size = g.op("Concat", *unsqueezed, axis_i=0)
        return onnx_syms_view(g, input, size)


class iqReshape_as(torch.autograd.Function):
    @staticmethod
    def forward(self, input, other):
        y = super(input.__class__, input,).reshape_as(other)
        self.size = input.size()
        return from_torch_tensor(y, input.scale_data, input.bits, input.zero_point)

    @staticmethod
    def backward(self, s):
        size = self.size
        ret = s.reshape(size)
        return ret, None

    @staticmethod
    def symbolic(g, input, other):
        return onnx_syms_reshape_as(g, input, other)


class iqFlip(torch.autograd.Function):
    @staticmethod
    def forward(self, input, *args):
        y = super(input.__class__, input,).flip(*args)
        self.num_in = len(args)
        self.dim = args
        return from_torch_tensor(y, input.scale_data, input.bits, input.zero_point)

    @staticmethod
    def backward(self, s):
        num_in = self.num_in
        dim = self.dim
        return s.flip(*dim), None

    @staticmethod
    def symbolic(g, input, *dims):
        if isinstance(dims[0], int):
            return onnx_syms_flip(g, input, dims)
        return onnx_syms_flip(g, input, *dims)


class iqSplit(torch.autograd.Function):
    @staticmethod
    def forward(self, input, split_size_or_sections, dim=0):
        y = super(input.__class__, input,).split(split_size_or_sections, dim)
        self.dim = dim
        y = tuple([from_torch_tensor(t, input.scale_data,
                  input.bits, input.zero_point) for t in y])
        return y

    @staticmethod
    def backward(self, *s):
        dim = self.dim
        return torch.cat(s, dim), None, None

    @staticmethod
    def symbolic(g, input, split_size_or_sections, dim=0):
        sizes = sym_help._get_tensor_dim_size(input, dim)
        if (isinstance(split_size_or_sections, int)):
            splits = [split_size_or_sections] * \
                (sizes // split_size_or_sections)
            leftover = sizes % split_size_or_sections
            if leftover:
                splits.append(leftover)
        else:
            splits = list(split_size_or_sections)
        return g.op("Split", input, split_i=splits, axis_i=dim, outputs=len(splits))


class iqSqueeze(torch.autograd.Function):
    @staticmethod
    def forward(self, input, *args):
        y = super(input.__class__, input,).squeeze(*args)
        self.num_in = len(args)
        self.size = input.size()
        return from_torch_tensor(y, input.scale_data, input.bits, input.zero_point)

    @staticmethod
    def backward(self, s):
        size = self.size
        l = [None for i in range(self.num_in)]
        ret = [s.reshape(size)] + l
        return tuple(ret)

    @staticmethod
    def symbolic(g, input, *dim):
        return onnx_syms_squeeze(g, input, *dim)


class iqUnsqueeze(torch.autograd.Function):
    @staticmethod
    def forward(self, input, dim):
        y = super(input.__class__, input,).unsqueeze(dim)
        self.size = input.size()
        return from_torch_tensor(y, input.scale_data, input.bits, input.zero_point)

    @staticmethod
    def backward(self, s):
        size = self.size
        return s.reshape(size), None

    @staticmethod
    def symbolic(g, input, dim):
        return onnx_syms_unsqueeze(g, input, dim)


class iqTranspose(torch.autograd.Function):
    @staticmethod
    def forward(self, input, dim0, dim1):
        y = super(input.__class__, input,).transpose(dim0, dim1)
        self.dim0 = dim0
        self.dim1 = dim1
        return from_torch_tensor(y, input.scale_data, input.bits, input.zero_point)

    @staticmethod
    def backward(self, s):
        dim0 = self.dim0
        dim1 = self.dim1
        return s.transpose(dim0, dim1), None, None

    @staticmethod
    def symbolic(g, input, dim0, dim1):
        return onnx_syms_transpose(g, input, dim0, dim1)


class iqPermute(torch.autograd.Function):
    @staticmethod
    def forward(self, input, *dims):
        y = super(input.__class__, input,).permute(*dims)
        self.dims = dims
        self.save_for_backward(input, )
        return from_torch_tensor(y, input.scale_data, input.bits, input.zero_point)

    @staticmethod
    def backward(self, s):
        input, = self.saved_tensors
        dims = self.dims
        input = input.detach().clone().requires_grad_(True)
        gradInput = None
        with torch.enable_grad():
            z = input.permute(*dims)
            gradInput, = torch.autograd.grad(z, (input, ), s)
        grad_tuple = [gradInput]
        for i in range(len(dims)):
            grad_tuple.append(None)
        return tuple(grad_tuple)

    @staticmethod
    def symbolic(g, input, *dims):
        if isinstance(dims[0], tuple):
            dims = dims[0]
        return onnx_syms_permute(g, input, dims)


class iqFlatten(torch.autograd.Function):
    @staticmethod
    def forward(self, input, start_dim=0, end_dim=-1):
        y = super(input.__class__, input,).flatten(start_dim, end_dim)
        self.input_dims = input.size()
        return from_torch_tensor(y, input.scale_data, input.bits, input.zero_point)

    @staticmethod
    def backward(self, s):
        input_dims = self.input_dims
        return s.reshape(input_dims), None, None

    @staticmethod
    def symbolic(g, input, start_dim=0, end_dim=-1):
        return onnx_syms_flatten(g, input, start_dim, end_dim)


class iqContiguous(torch.autograd.Function):
    @staticmethod
    def forward(self, input, *args, **kwargs):
        y = super(input.__class__, input,).contiguous(*args, **kwargs)
        return from_torch_tensor(y, input.scale_data, input.bits, input.zero_point)

    @staticmethod
    def backward(self, s):
        return s, None

    @staticmethod
    def symbolic(g, input, *args, **kwargs):
        return g.op("Identity", input)


class iqChunk(torch.autograd.Function):
    @staticmethod
    def forward(self, input, chunks, dim=0):
        self.dim = dim
        y = super(input.__class__, input,).chunk(chunks, dim)
        return tuple(from_torch_tensor(ret, input.scale_data, input.bits, input.zero_point) for ret in y)

    @staticmethod
    def backward(self, *s):
        dim = self.dim
        return torch.cat(s, dim), None, None

    @staticmethod
    def symbolic(g, input, chunks, dim):
        return onnx_sym_chunk(g, input, chunks, dim)


class iqMul(torch.autograd.Function):
    @staticmethod
    def forward(self, x, y, c_y, scale_x, scale_y, zero_x, zero_y, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        self.save_for_backward(x, y)
        
        scale_z_iq = local_scale_o
        momentum = 0.1
        if training:
            running_o.mul_(1-momentum).add_(momentum*(127/local_scale_o()))
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(scale_z_iq(), 2))
                scale_z_iq = math.pow(2, scale_log)
            scale_z_iq = ScalerBuffer(scale_z_iq)
        else:
            assert running_o.data > 0, 'Must at least training one batch'
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(127/running_o.data, 2))
                scale_z_iq = math.pow(2, scale_log)
            else:
                scale_z_iq = np.float32((math.pow(2, 8-1)-1) / running_o.data)
            scale_z_iq = ScalerBuffer(scale_z_iq)
            scale_o.fill_(scale_z_iq())
        if math.log(scale_x(),2) + math.log(scale_y(),2) < math.log(scale_z_iq(),2):
            scale_y_add = math.log(scale_z_iq(),2) - math.log(scale_x(),2) - math.log(scale_y(),2)
            scale_y.fill_(scale_y()*2**scale_y_add)
        x_int = x.quant_to_int8(scale_x())
        y_int = y.quant_to_int8(scale_y())
        x_int = x_int.contiguous()
        y_int = y_int.contiguous()
        if (x_int.size() != y_int.size()):
            x_int, y_int = torch.broadcast_tensors(x_int, y_int)
        z_int = None

        if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
            r_scale = np.float32(scale_z_iq()/(scale_x()*scale_y()))
            z_int = (((x_int * y_int)*r_scale)+0.5).floor()
            z_int.clamp_(-128, 127)
        else:
            assert False, 'platform_quant mode donot support for iqmul'
        # z_float = z_int.float() /scale_z_iq()
        z_float = Quant.dequant(z_int, scale_z_iq)

        if dump:
            name_list = ["input1", "input2", "outputs",
                         "q_input1",  "q_input2", "q_outputs"]
            attr_list = [x, y, z_float, x_int, y_int, z_int]
            Dump.dump_file(prefix, ".iqMul.", zip(name_list, attr_list), path)

        return from_torch_tensor(z_float, scale_z_iq(), 8)

    @staticmethod
    def backward(self, s):
        x, y = self.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            z = x * y
            grad = torch.autograd.grad(z, (x, y), s)
        return grad[0], grad[1], None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, y, c_y, scale_x, scale_y, zero_x, zero_y, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        if c_y is not None:
            c_y = c_y * scale_y()
            y = g.op("Constant", value_t=torch.tensor(c_y, dtype=torch.int8))
        input_list = [x, y]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict = {'scale_x_f': scale_x(), 'scale_y_f': scale_y(
        ), 'scale_o_f': scale_o(), 'platform_quant_s': platform_quant}

        return g.op("thinker::iqMul", *input_list, **param_dict)


class iqMulLayer(torch.nn.Module):
    r"""对iqmul的layer封装

    """

    def __init__(self):
        super(iqMulLayer, self).__init__()
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.is_y_constant = False

        self.register_buffer('scale_o', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))
        self.register_buffer('y', torch.zeros(1))

    def forward(self, x, y, local_scale_o, quant_mode=QuantMode.QValue):
        r""" 前向操作

        Args:
            x,y(IQTensor):执行x*y的IQTensor
            local_scale_o(float):本batch的scale值
            quant_mode(QuantMode):输出z的量化方法，目前仅支持Q值量化
        returns:
            加法结果z 类型为IQTensor 

        """
        scale_x = ScalerBuffer(x.scale_data)
        c_y = None
        if not isinstance(y, torch.Tensor):
            c_y = y
            self.is_y_constant = True
            if 0 < y < 1:
                y_ = math.log(y, 2)
                if y_ % 1 != 0:
                    assert False, f"in iqmul, input y must equals to 2**a, where a is integer, but you have y = {y}"
                y = torch.tensor(y, dtype=torch.float32, device=x.device)
                y = from_torch_tensor(y, 2**(-y_), 8)
            elif y >= 1:
                if y & (y-1) != 0:
                    assert False, f"in iqmul, input y must equals to 2**a, where a is integer, but you have y = {y}"
                y = torch.tensor(y, dtype=torch.float32, device=x.device)
                y = from_torch_tensor(y, 1, 8)
            else:
                assert False, f"in iqmul, input y must lager than 0, and equals to 2**a, where a is integer, but you have y = {y}"

        scale_y = ScalerBuffer(y.scale_data)
        scale_o = ScalerBuffer(self.scale_o)
        running_o = ScalerBuffer(self.running_o)
        local_scale_o = ScalerBuffer(local_scale_o)
        z = iqMul.apply(x, y, c_y, scale_x, scale_y, x.zero_point, y.zero_point, local_scale_o,
                        scale_o, running_o, self.training, quant_mode, self.prefix, self.dump, self.path)
        self.scale_o.fill_(scale_o.data)
        self.running_o.fill_(running_o.data)
        return z

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def iqmul(module, x, y, name="_default"):
    r"""实现IQTensor加法,并对输出值进行定标,当前输出支持最值定标和Q值(2幂次)定标

    .. math::
        z = x * y

    Args:
        module(torch.nn.Module):乘法注册的module，如果iqmul在module的forward里面使用一般是self
        x(IOTensor):IQTensor变量
        y(IOTensor):IQTensor变量
        name(str):该加法产生的子module名字，是moudule的成员变量名字
    Notes:
        如果使用IQTensor，该加法会自动生效,通过linger.SetIQTensorMul(False)可以关闭iqmul
    Example:
        >>> class iqTestLayer(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super(iqTestLayer,self).__init__()
        >>>     def forward(self,x,y):
        >>>         return iqmul(self,x,y,'test') 
        >>> a = from_torch_tensor(x,127.0/6.0,8)
        >>> b = from_torch_tensor(y,127.0/8,8)
        >>> net = iqTestLayer().cuda()
        >>> m = net(a,b)

    """
    quant_mode = getattr(module, LINGER_MODE, QuantMode.QValue)
    # assert quant_mode is not None, 'invalid add quant mode'
    assert isinstance(x, IQTensor)
    # assert isinstance(y,IQTensor)
    assert x.bits == 8, 'iqmul only support 8bit'
    if isinstance(y, IQTensor):
        assert y.bits == 8, 'iqmul only support 8bit tensor'
    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + '_iqmul_' + name
    iq_layer = None
    if hasattr(module, var_name):
        iq_layer = getattr(module, var_name)
    else:
        iq_layer = iqMulLayer()
        iq_layer.training = module.training
        iq_layer = iq_layer.to(x.device)
        setattr(module, var_name, iq_layer)
    scale_z = None
    with torch.no_grad():
        z_f = torch.mul(x, y)
        max_z = torch.max(torch.abs(z_f))
        if max_z == 0:
            scale_z = 1.0
        else:
            scale_z = 127 / max_z.item()
    return iq_layer(x, y, scale_z, quant_mode)


class iqDiv(torch.autograd.Function):
    @staticmethod
    def forward(self, x, y, scale_x, scale_y, zero_x, zero_y, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        self.save_for_backward(x, y)
        scale_z_iq = local_scale_o
        momentum = 0.1
        if training:
            running_o.mul_(1-momentum).add_(momentum*(127/local_scale_o()))
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(scale_z_iq(), 2))
                scale_z_iq = math.pow(2, scale_log)
            scale_z_iq = ScalerBuffer(scale_z_iq)
        else:
            assert running_o.data > 0, 'Must at least training one batch'
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(127/running_o.data, 2))
                scale_z_iq = math.pow(2, scale_log)
            else:
                scale_z_iq = np.float32((math.pow(2, 8-1)-1) / running_o.data)
            scale_z_iq = ScalerBuffer(scale_z_iq)
            scale_o.fill_(scale_z_iq())
        z_int = None
        z_float = x.data / y.data
        if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
            z_int = (z_float * scale_z_iq() + 0.5).floor().int()
            z_int.clamp_(-128, 127)
        else:
            assert False, 'platform_quant mode donot support for iqdiv'
        # z_float = z_int.float() /scale_z_iq()
        z_float = Quant.dequant(z_int, scale_z_iq)

        if dump:
            name_list = ["input1", "input2", "outputs", "q_outputs"]
            attr_list = [x, y, z_float, z_int]
            Dump.dump_file(prefix, ".iqDiv.", zip(name_list, attr_list), path)

        return from_torch_tensor(z_float, scale_z_iq(), 8)

    @staticmethod
    def backward(self, s):
        x, y = self.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            z = x / y
            grad = torch.autograd.grad(z, (x, y), s)
        return grad[0], grad[1], None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, y, scale_x, scale_y, zero_x, zero_y, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        input_list = [x, y]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict = {'scale_x_f': scale_x(), 'scale_y_f': scale_y(
        ), 'scale_o_f': scale_o(), 'platform_quant_s': platform_quant}

        return g.op("thinker::iqDiv", *input_list, **param_dict)


class iqDivScalar(torch.autograd.Function):
    @staticmethod
    def forward(self, x, y, scale_x, scale_y, zero_x, zero_y, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        assert isinstance(y, (int, float)), 'only support div scalar here'
        self.save_for_backward(x)
        self.y = y
        scale_z_iq = local_scale_o
        momentum = 0.1
        if training:
            running_o.mul_(1-momentum).add_(momentum*(127/local_scale_o()))
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(scale_z_iq(), 2))
                scale_z_iq = math.pow(2, scale_log)
            scale_z_iq = ScalerBuffer(scale_z_iq)
        else:
            assert running_o.data > 0, 'Must at least training one batch'
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(127/running_o.data, 2))
                scale_z_iq = math.pow(2, scale_log)
            else:
                scale_z_iq = np.float32((math.pow(2, 8-1)-1) / running_o.data)
            scale_z_iq = ScalerBuffer(scale_z_iq)
            scale_o.fill_(scale_z_iq())
        z_int = None
        x_float = x
        y_float = torch.tensor(y, dtype=torch.float32, device=x.device)
        z_float = x_float / y_float
        if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
            z_int = (z_float * scale_z_iq() + 0.5).floor().int()
            z_int.clamp_(-128, 127)
        else:
            assert False, 'platform_quant mode donot support for iqdiv'
        # z_float = z_int.float() /scale_z_iq()
        z_float = Quant.dequant(z_int, scale_z_iq)

        if dump:
            name_list = ["input1", "input2", "outputs", "q_outputs"]
            attr_list = [x, y, z_float, z_int]
            Dump.dump_file(prefix, ".iqDivScalar.",
                           zip(name_list, attr_list), path)

        return from_torch_tensor(z_float, scale_z_iq(), 8)

    @staticmethod
    def backward(self, s):
        x, = self.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        y = self.y
        grad = None
        with torch.enable_grad():
            z = x / y
            grad = torch.autograd.grad(z, (x,), s)
        return grad[0], None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, y, scale_x, scale_y, zero_x, zero_y, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        tensor_y = g.op("Constant", value_t=torch.tensor(
            y, dtype=torch.float32))
        input_list = [x, tensor_y]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict = {'scale_x_f': scale_x(), 'scale_y_f': scale_y(
        ), 'scale_o_f': scale_o(), 'platform_quant_s': platform_quant}

        return g.op("thinker::iqDiv", *input_list, **param_dict)


class iqDivLayer(torch.nn.Module):
    r"""对iqdiv的layer封装

    """

    def __init__(self):
        super(iqDivLayer, self).__init__()
        self.prefix = ""
        self.dump = False
        self.path = ""

        self.register_buffer('scale_o', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))

    def forward(self, x, y, local_scale_o, quant_mode=QuantMode.QValue):
        r""" 前向操作

        Args:
            x,y(IQTensor):执行x/y的IQTensor
            local_scale_o(float):本batch的scale值
            quant_mode(QuantMode):输出z的量化方法，目前仅支持Q值量化
        returns:
            除法结果z 类型为IQTensor 

        """
        scale_x = ScalerBuffer(x.scale_data)
        scale_y = ScalerBuffer(y.scale_data) if isinstance(
            y, IQTensor) else ScalerBuffer(1.0)
        scale_o = ScalerBuffer(self.scale_o)
        running_o = ScalerBuffer(self.running_o)
        local_scale_o = ScalerBuffer(local_scale_o)
        if isinstance(y, IQTensor):
            z = iqDiv.apply(x, y, scale_x, scale_y, x.zero_point, y.zero_point, local_scale_o,
                            scale_o, running_o, self.training, quant_mode, self.prefix, self.dump, self.path)
        else:
            z = iqDivScalar.apply(x, y, scale_x, scale_y, x.zero_point, 0, local_scale_o,
                                  scale_o, running_o, self.training, quant_mode, self.prefix, self.dump, self.path)
        # print('running_o: ', running_o())
        self.scale_o.fill_(scale_o.data)
        self.running_o.fill_(running_o.data)
        return z

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def iqdiv(module, x, y, name="_default"):
    r"""实现IQTensor除法(python3实现逻辑),并对输出值进行定标,当前输出支持最值定标和Q值(2幂次)定标

    .. math::
        z = x / y

    Args:
        module(torch.nn.Module):乘法注册的module，如果iqmul在module的forward里面使用一般是self
        x(IOTensor):IQTensor变量
        y(float):Scalar变量
        name(str):该加法产生的子module名字，是moudule的成员变量名字
    Notes:
        如果使用IQTensor，该加法会自动生效,通过linger.SetIQTensorDiv(False)可以关闭iqdiv
    Example:
        >>> class iqTestLayer(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super(iqTestLayer,self).__init__()
        >>>     def forward(self,x,y):
        >>>         return iqdiv(self,x,y,'test') 
        >>> a = from_torch_tensor(x,127.0/6.0,8)
        >>> b = from_torch_tensor(y,127.0/8,8)
        >>> net = iqTestLayer().cuda()
        >>> m = net(a,b)

    """
    quant_mode = getattr(module, LINGER_MODE, QuantMode.QValue)
    # assert quant_mode is not None, 'invalid add quant mode'
    assert isinstance(x, IQTensor)
    assert x.bits == 8, 'iqdiv only support 8bit'
    if isinstance(y, IQTensor):
        assert y.bits == 8, 'iqdiv only support 8bit'
    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + '_iqdiv_' + name
    iq_layer = None
    if hasattr(module, var_name):
        iq_layer = getattr(module, var_name)
    else:
        iq_layer = iqDivLayer()
        iq_layer.training = module.training
        iq_layer = iq_layer.to(x.device)
        setattr(module, var_name, iq_layer)
    scale_z = None
    with torch.no_grad():
        z_f = torch.div(x, y)
        max_z = torch.max(torch.abs(z_f))
        if max_z == 0:
            scale_z = 1.0
        else:
            scale_z = 127 / max_z.item()
    return iq_layer(x, y, scale_z, quant_mode)


class iqAdd(torch.autograd.Function):
    @staticmethod
    def forward(self, x, y, scale_x, scale_y, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        self.save_for_backward(x, y)
        x_int = x.quant_to_int8(scale_x())
        y_int = y.quant_to_int8(scale_y())
        x_int = x_int.contiguous()
        y_int = y_int.contiguous()
        scale_z_iq = local_scale_o
        momentum = 0.1
        if training:
            running_o.mul_(1-momentum).add_(momentum*(127/local_scale_o()))
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(scale_z_iq(), 2))
                scale_z_iq = math.pow(2, scale_log)
            scale_z_iq = ScalerBuffer(scale_z_iq)
        else:
            assert running_o.data > 0, 'Must at least training one batch'
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(127/running_o.data, 2))
                scale_z_iq = math.pow(2, scale_log)
            else:
                scale_z_iq = np.float32((math.pow(2, 8-1)-1) / running_o.data)
            scale_z_iq = ScalerBuffer(scale_z_iq)
            scale_o.fill_(scale_z_iq())
        if (x_int.size() != y_int.size()):
            x_int, y_int = torch.broadcast_tensors(x_int, y_int)
        z_int = None
        if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
            z_int = (x_int * (scale_z_iq()/scale_x()) + 0.5).floor().int() + \
                (y_int*(scale_z_iq()/scale_y()) + 0.5).floor().int()
            # z_int = (x_int * (scale_z_iq()/scale_x()) + y_int * (scale_z_iq()/scale_y()) + 0.5).floor().int()
            z_int.clamp_(-128, 127)
        else:
            assert False, 'platform_quant mode donot support for iqadd'
        z_float = Quant.dequant(z_int, scale_z_iq)
        # z_float = z_int.float() /scale_z_iq()

        if dump:
            name_list = ["input1", "input2", "outputs",
                         "q_input1",  "q_input2", "q_outputs"]
            attr_list = [x, y, z_float, x_int, y_int, z_int]
            Dump.dump_file(prefix, ".iqAdd.", zip(name_list, attr_list), path)

        return from_torch_tensor(z_float, scale_z_iq(), 8)

    @staticmethod
    def backward(self, s):
        x, y = self.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            z = x + y
            grad = torch.autograd.grad(z, (x, y), s)
        return grad[0], grad[1], None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, y, scale_x, scale_y, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        input_list = [x, y]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict = {'scale_x_f': scale_x(), 'scale_y_f': scale_y(
        ), 'scale_o_f': scale_o(), 'platform_quant_s': platform_quant}
        if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
            param_dict['mode_s'] = 'Non_t710_mode'
        return g.op("thinker::iqAdd", *input_list, **param_dict)


class iqAddScalar(torch.autograd.Function):
    @staticmethod
    def forward(self, x, y, scale_x, scale_y, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        self.save_for_backward(x,)
        self.y = y
        scale_z_iq = local_scale_o
        momentum = 0.1
        if training:
            running_o.mul_(1-momentum).add_(momentum*(127/local_scale_o()))
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(scale_z_iq(), 2))
                scale_z_iq = math.pow(2, scale_log)
            scale_z_iq = ScalerBuffer(scale_z_iq)
        else:
            assert running_o.data > 0, 'Must at least training one batch'
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(127/running_o.data, 2))
                scale_z_iq = math.pow(2, scale_log)
            else:
                scale_z_iq = np.float32((math.pow(2, 8-1)-1) / running_o.data)
            scale_z_iq = ScalerBuffer(scale_z_iq)
            scale_o.fill_(scale_z_iq())
        z_int = None
        y = torch.tensor(y, dtype=torch.float32, device=x.device)
        z_float = x + y
        if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
            z_int = ((z_float * scale_z_iq()) + 0.5).floor().int()
            z_int.clamp_(-128, 127)
        else:
            assert False, 'platform_quant mode donot support for iqadd'
        z_float = Quant.dequant(z_int, scale_z_iq)

        if dump:
            name_list = ["input1", "input2", "outputs", "q_outputs"]
            attr_list = [x, y, z_float, z_int]
            Dump.dump_file(prefix, ".iqAddScalar.",
                           zip(name_list, attr_list), path)

        return from_torch_tensor(z_float, scale_z_iq(), 8)

    @staticmethod
    def backward(self, s):
        x,  = self.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        y = self.y
        grad = None
        with torch.enable_grad():
            z = x + y
            grad = torch.autograd.grad(z, (x,), s)
        return grad[0], None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, y, scale_x, scale_y, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        tensor_y = g.op("Constant", value_t=torch.tensor(
            y, dtype=torch.float32))
        input_list = [x, tensor_y]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict = {'scale_x_f': scale_x(), 'scale_y_f': scale_y(
        ), 'scale_o_f': scale_o(), 'platform_quant_s': platform_quant}
        return g.op("thinker::iqAdd", *input_list, **param_dict)


class iqAddLayer(torch.nn.Module):
    r"""对iqadd的layer封装

    """

    def __init__(self):
        super(iqAddLayer, self).__init__()
        self.prefix = ""
        self.dump = False
        self.path = ""

        self.register_buffer('scale_o', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))

    def forward(self, x, y, local_scale_o, quant_mode=QuantMode.QValue):
        r""" 前向操作

        Args:
            x,y(IQTensor):执行x+y的IQTensor
            local_scale_o(float):本batch的scale值
            quant_mode(QuantMode):输出z的量化方法，目前仅支持Q值量化
        returns:
            加法结果z 类型为IQTensor 

        """
        scale_x = ScalerBuffer(x.scale_data)
        scale_y = ScalerBuffer(y.scale_data) if isinstance(
            y, IQTensor) else ScalerBuffer(1.0)
        local_scale_o = ScalerBuffer(local_scale_o)
        scale_o = ScalerBuffer(self.scale_o)
        running_o = ScalerBuffer(self.running_o)
        if isinstance(y, IQTensor):
            z = iqAdd.apply(x, y, scale_x, scale_y, local_scale_o, scale_o, running_o,
                            self.training, quant_mode, self.prefix, self.dump, self.path)
        else:
            z = iqAddScalar.apply(x, y, scale_x, scale_y, local_scale_o, scale_o,
                                  running_o, self.training, quant_mode, self.prefix, self.dump, self.path)
        self.scale_o.fill_(scale_o.data)
        self.running_o.fill_(running_o.data)
        return z

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def iqadd(module, x, y, name="_default"):
    r"""实现IQTensor加法,并对输出值进行定标,当前输出支持最值定标和Q值(2幂次)定标

    .. math::
        z = x + y

    Args:
        module(torch.nn.Module):加法注册的module，如果iqadd在module的forward里面使用一般是self
        x(IOTensor):IQTensor变量
        y(IOTensor):IQTensor变量
        name(str):该加法产生的子module名字，是moudule的成员变量名字
    Notes:
        如果使用IQTensor，该加法会自动生效,通过linger.SetIQTensorAdd(False)可以关闭iqadd
    Example:
        >>> class iqTestLayer(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super(iqTestLayer,self).__init__()
        >>>     def forward(self,x,y):
        >>>         return iqadd(self,x,y,'test') 
        >>> a = from_torch_tensor(x,127.0/6.0,8)
        >>> b = from_torch_tensor(y,127.0/8,8)
        >>> net = iqTestLayer().cuda()
        >>> m = net(a,b)

    """
    quant_mode = getattr(module, LINGER_MODE, QuantMode.QValue)
    # assert quant_mode is not None, 'invalid add quant mode'
    assert isinstance(x, IQTensor)
    # assert isinstance(y,IQTensor)
    assert x.bits == 8, 'iqadd only support 8bit'
    if isinstance(y, IQTensor):
        assert y.bits == 8, 'iqadd only support 8bit'
    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + '_iqadd_' + name
    iq_layer = None
    if hasattr(module, var_name):
        iq_layer = getattr(module, var_name)
    else:
        iq_layer = iqAddLayer()
        iq_layer.training = module.training
        iq_layer = iq_layer.to(x.device)
        setattr(module, var_name, iq_layer)
    scale_z = None
    with torch.no_grad():
        z_f = torch.add(x, y)
        if torch.is_tensor(y):
            z_f = torch.cat((x, y, z_f))
        else:
            z_f = torch.cat((x, z_f))
        max_z = torch.max(torch.abs(z_f))
        if max_z == 0:
            scale_z = 1.0
        else:
            scale_z = 127 / max_z.item()
    return iq_layer(x, y, scale_z, quant_mode)


class iqSum(torch.autograd.Function):
    @staticmethod
    def forward(self, x, scale_x, args, kwargs, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        self.save_for_backward(x,)
        self.args = args
        self.kwargs = kwargs
        scale_z_iq = local_scale_o
        momentum = 0.1
        if training:
            running_o.mul_(1-momentum).add_(momentum*(127/local_scale_o()))
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(scale_z_iq(), 2))
                scale_z_iq = math.pow(2, scale_log)
            scale_z_iq = ScalerBuffer(scale_z_iq)
        else:
            assert running_o.data > 0, 'Must at least training one batch'
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(127/running_o.data, 2))
                scale_z_iq = math.pow(2, scale_log)
            else:
                scale_z_iq = np.float32((math.pow(2, 8-1)-1) / running_o.data)
            scale_z_iq = ScalerBuffer(scale_z_iq)
            scale_o.fill_(scale_z_iq())
        z_int = None
        z_float = torch.sum(x, *args, **kwargs)
        if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
            z_int = (z_float * scale_z_iq() + 0.5).floor().int()
            z_int.clamp_(-128, 127)
        else:
            assert False, 'platform_quant mode donot support for iqSum'
        z_float = Quant.dequant(z_int, scale_z_iq)

        if dump:
            name_list = ["input", "outputs", "q_outputs"]
            attr_list = [x, z_float, z_int]
            Dump.dump_file(prefix, ".iqSum.", zip(name_list, attr_list), path)

        return from_torch_tensor(z_float, scale_z_iq(), 8)

    @staticmethod
    def backward(self, s):
        x,  = self.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        args = self.args
        kwargs = self.kwargs
        grad = None
        with torch.enable_grad():
            z = torch.sum(x, *args, **kwargs)
            grad = torch.autograd.grad(z, (x,), s)
        return grad[0], None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, scale_x, args, kwargs, local_scale_o, scale_o, running_o, training, quant_mode, prefix, dump, path):
        input_list = [x, ]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict = {'scale_x_f': scale_x(), 'scale_o_f': scale_o(),
                      'platform_quant_s': platform_quant}

        return g.op("thinker::iqSum", *input_list, **param_dict)


class iqSumLayer(torch.nn.Module):
    r"""对iqsum的layer封装

    """

    def __init__(self):
        super(iqSumLayer, self).__init__()
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.register_buffer('scale_o', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))

    def forward(self, x, args, kwargs, local_scale_o, quant_mode=QuantMode.QValue):
        r""" 前向操作

        Args:
            x,y(IQTensor):执行x.sum()的IQTensor
            local_scale_o(float):本batch的scale值
            quant_mode(QuantMode):输出z的量化方法，目前仅支持Q值量化
        returns:
            加法结果z 类型为IQTensor 

        """
        scale_x = ScalerBuffer(x.scale_data)
        local_scale_o = ScalerBuffer(local_scale_o)
        scale_o = ScalerBuffer(self.scale_o)
        running_o = ScalerBuffer(self.running_o)
        z = iqSum.apply(x, scale_x, args, kwargs, local_scale_o, scale_o, running_o,
                        self.training, quant_mode, self.prefix, self.dump, self.path)
        self.scale_o.fill_(scale_o.data)
        self.running_o.fill_(running_o.data)
        return z

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def iqsum(module, x, *args, **kwargs):
    r"""实现IQTensor求和,并对输出值进行定标,当前输出支持最值定标和Q值(2幂次)定标

    .. math::
        z = x.sum(*args, **kwargs)
    """
    quant_mode = getattr(module, LINGER_MODE, QuantMode.QValue)
    assert isinstance(x, IQTensor)
    assert x.bits == 8, 'iqsum only support 8bit'
    name = kwargs.pop('name', '_default')
    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + '_iqsum_' + name
    iq_layer = None
    if hasattr(module, var_name):
        iq_layer = getattr(module, var_name)
    else:
        iq_layer = iqSumLayer()
        iq_layer.training = module.training
        iq_layer = iq_layer.to(x.device)
        setattr(module, var_name, iq_layer)
    scale_z = None
    with torch.no_grad():
        z_f = torch.sum(x, *args, **kwargs)
        max_z = torch.max(torch.abs(z_f))
        if max_z == 0:
            scale_z = 1.0
        else:
            scale_z = 127 / max_z.item()
    return iq_layer(x, args, kwargs, scale_z, quant_mode)


class IQTensor(torch.Tensor):
    r"""实现量化方式和导出onnx方式IQTensor,除了包含torch.Tensor相关属性和变量外.

    Supports:
        `第一类`: bypass类函数，对数据不做任何处理,仅仅传递IQTensor属性。包括IQTensor类函数view,view_as,reshape,reshape_as,squeeze,unsqueeze,
        transpose,flatten,__getitem__ (即y=x[2:]类似切片操作)，以及torch的操作torch\.max_pool2d、torch\.relu、torch\.relu_ 

        `第二类`: 重定标函数,这些操作会在输出tensor重新定标，包括IQTensor类函数__add__(+)、__iadd__(+=) 、__mul__(*)和__imul__(*),以及torch的操作torch.cat。

        `第三类`: 其他函数，将执行torch.Tensor的默认行为

    Attributes:
        bits(int):表示IQTensor数据的精度，一般为8或16
        scale_data(float):表示IQTensor数据的scale，表示意义为 :math:`\frac{2^{bits-1}-1}{max\_value}`

    """
    if torch.__version__ >= '1.7.0':
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):

            if kwargs is None:
                kwargs = {}

            if not all(issubclass(cls, t) for t in types):
                return NotImplemented

            with _C.DisableTorchFunction():
                ret = func(*args, **kwargs)
                return ret

    def __add__(self, *args, **kwargs):
        if not isinstance(args[0], (IQTensor, float, int)) or not config.IQTensor.iqadd \
                or self.dtype != torch.float:
            return super(IQTensor, self).__add__(*args, **kwargs)
        module_self = get_current_module()
        if module_self is None:
            return super(IQTensor, self).__add__(*args, **kwargs)
        iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
        setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)
        return iqadd(module_self, self, args[0], '_default_index_'+str(iname_index))

    def __iadd__(self, *args, **kwargs):
        if not isinstance(args[0], (IQTensor, float, int)) or not config.IQTensor.iqadd \
                or self.dtype != torch.float:
            return super(IQTensor, self).__iadd__(*args, **kwargs)
        module_self = get_current_module()
        if module_self is None:
            return super(IQTensor, self).__iadd__(*args, **kwargs)
        iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
        setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)
        self = iqadd(module_self, self,
                     args[0], '_default_index_'+str(iname_index))
        return self

    def __mul__(self, *args, **kwargs):
        if not isinstance(args[0], (IQTensor, float, int)) or not config.IQTensor.iqmul \
                or self.dtype != torch.float:
            return super(IQTensor, self).__mul__(*args, **kwargs)
        module_self = get_current_module()
        if module_self is None:
            return super(IQTensor, self).__mul__(*args, **kwargs)
        iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
        setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)
        return iqmul(module_self, self, args[0], '_default_index_'+str(iname_index))

    def __imul__(self, *args, **kwargs):
        if not isinstance(args[0], (IQTensor, float, int)) or not config.IQTensor.iqmul \
                or self.dtype != torch.float:
            return super(IQTensor, self).__imul__(*args, **kwargs)
        module_self = get_current_module()
        if module_self is None:
            return super(IQTensor, self).__imul__(*args, **kwargs)
        iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
        setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)
        self = iqmul(module_self, self,
                     args[0], '_default_index_'+str(iname_index))
        return self

    def __truediv__(self, *args, **kwargs):
        if not isinstance(args[0], (IQTensor, float, int)) or not config.IQTensor.iqdiv \
                or self.dtype != torch.float:
            return super(IQTensor, self).__truediv__(*args, **kwargs)
        module_self = get_current_module()
        if module_self is None:
            return super(IQTensor, self).__truediv__(*args, **kwargs)
        iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
        setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)
        self = iqdiv(module_self, self,
                     args[0], '_default_index_'+str(iname_index))
        return self

    def view(self, *args, **kwargs):
        return iqView.apply(self, *args, **kwargs)

    def flip(self, *args, **kwargs):
        return iqFlip.apply(self, *args, **kwargs)

    def split(self, *args, **kwargs):
        return iqSplit.apply(self, *args, **kwargs)

    def view_as(self, other):
        return iqReshape_as.apply(self, other)

    def reshape(self, *args, **kwargs):
        return iqReshape.apply(self, *args, **kwargs)

    def reshape_as(self, other):
        return iqReshape_as.apply(self, other)

    def squeeze(self, *args):
        return iqSqueeze.apply(self, *args)

    def unsqueeze(self, dim):
        return iqUnsqueeze.apply(self, dim)

    def transpose(self, dim0, dim1):
        return iqTranspose.apply(self, dim0, dim1)

    def permute(self, *dims):
        return iqPermute.apply(self, *dims)

    def flatten(self, start_dim=0, end_dim=-1):
        return iqFlatten.apply(self, start_dim, end_dim)

    def contiguous(self, *args, **kwargs):
        return iqContiguous.apply(self, *args, **kwargs)

    def chunk(self, chunks, dim=0):
        m = super(IQTensor, self).chunk(chunks, dim)
        return tuple(from_torch_tensor(ret, self.scale_data, self.bits, self.zero_point) for ret in m)

    def sum(self, *args, **kwargs):
        if not config.IQTensor.iqsum or self.dtype != torch.float:
            return super(IQTensor, self).sum(*args, **kwargs)
        module_self = get_current_module()
        if module_self is None:
            return super(IQTensor, self).sum(*args, **kwargs)
        iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
        setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)
        kwargs['name'] = '_default_index_'+str(iname_index)
        self = iqsum(module_self, self, *args, **kwargs)
        return self

    def __getitem__(self, indices):
        m = super(IQTensor, self).__getitem__(indices)
        return from_torch_tensor(m, self.scale_data, self.bits, self.zero_point)

    def requant_(self):
        r"""重新quant数据，按照成员的scale_data和bits进行重新量化和还原float,执行inplace操作

        .. math::
            data=\frac{clamp(round(data*scale\_data),2^{bits-1},2^{bits-1}-1)}{scale\_data}

        Notes:
            该方法不涉及到grad操作
        retruns:
            没有返回值

        """
        with torch.no_grad():
            assert self.scale_data is not None
            r = super(IQTensor, self).__mul__(self.scale_data)
            m2 = math.pow(2, self.bits-1)
            if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                r = (r + 0.5).floor()
            else:
                assert False, "linger only support luna quant."
            r.clamp_(-m2-0.01+self.zero_point, m2 - 1+0.01+self.zero_point)
            self.data.copy_(r/self.scale_data)

    def quant_to_int8(self, scale=None):
        r"""quant数据到int8

        .. math::
            result=int(clamp(round(data*scale\_data),-128,127))

        Args:
            scale(float or None):进行quant的scale，如果为None，表示使用self.scale_data,默认为None
        returns:
            返回量化后的int值，注意：返回数据类型torch\.int32,值为int8范围

        """
        with torch.no_grad():
            if scale == None:
                scale = self.scale_data
            r = super(IQTensor, self).__mul__(scale)
            # training with same drop
            if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                r = (r + 0.5).floor()
            else:
                assert False, "linger only support luna quant."
            r.clamp_(-128.0-0.01+self.zero_point, 127.0+0.01+self.zero_point)
            r = r.float()
            return r

    def scale_to(self, target_scale, training=False):
        r"""将自身数据调整到target_scale

        .. math::
            result=\frac{clamp(round(int(clamp(round(data*scale\_data),-128,127))*(\frac{target\_scale}{scale\_data})),-128,127)}{target\_scale}

        Args:
            target_scale(float):进行调整的目标scale
        returns:
            返回调整后的浮点数据

        """
        with torch.no_grad():
            scale = self.scale_data
            r = super(IQTensor, self).__mul__(scale)
            if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                r = (r + 0.5).floor()
            else:
                assert False, "linger only support luna quant."
            r.clamp_(-128.0-0.01+self.zero_point, 127.0+0.01+self.zero_point)
            r = r * (target_scale/scale)
            if config.PlatFormQuant.platform_quant in (PlatFormQuant.luna_quant,):
                r = (r + 0.5).floor()
            else:
                assert False, "linger only support luna quant."
            r.clamp_(-128.0-0.01+self.zero_point, 127.0+0.01+self.zero_point)
            r = r / target_scale
            return r


torch_bmm = torch.bmm

__all__ = ['IQTensor', 'from_torch_tensor', 'iqAddLayer', 'iqadd', 'iqMulLayer', 'iqmul', 'iqSumLayer',
           'iqDivLayer', 'platform_to_string', 'quantlinear', 'dequantlinear', 'Quant2IQTensor', 'torch_bmm']
