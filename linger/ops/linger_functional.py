import logging
import math
from collections import OrderedDict

import lingerext
import numpy as np
import torch
from torch.onnx import is_in_onnx_export
from torch.onnx.symbolic_opset9 import max_pool2d as onnx_syms_max_pool2d
from torch.onnx.symbolic_opset11 import (_prepare_onnx_paddings,
                                         constant_pad_nd, reflection_pad,
                                         replication_pad)

from ..config import config
from ..ops.bmm_int import BmmInt
from ..ops.ops import ModuleIntConfig
from ..ops.ops_names import (LINGER_FUNCINT_BMM_COUNTER,
                             LINGER_IQTENSOR_LAYER_COUNTER,
                             LINGER_MIX_INT8_MANUAL_ROUND_LAYERS, LINGER_MODE,
                             LINGER_OBIT)
from ..quant import Quant
from ..utils import Dump, PlatFormQuant, QuantMode, ScalerBuffer
from .iqtensor import (IQTensor, from_torch_tensor, iqTranspose,
                       platform_to_string, quantlinear)
from .module_self import get_current_module
from .requant import Requant

torch_max = torch.max
torch_transpose = torch.transpose
torch_relu = torch.relu
torch_pad = torch.nn.functional.pad
torch_relu_ = torch.relu_
torch_cat = torch.cat
torch_max_pool2d = torch.max_pool2d
torch_sigmoid = torch.sigmoid
torch_sigmoid_ = torch.sigmoid_

torch_tanh = torch.tanh
torch_tanh_ = torch.tanh_
torch_clamp = torch.clamp
torch_clamp_ = torch.clamp_
torch_dropout = torch.nn.functional.dropout
torch_onnx_export = torch.onnx.export
torch_pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence
torch_pad_packed_sequence = torch.nn.utils.rnn.pad_packed_sequence

torch_softmax = torch.softmax
torch_logsoftmax = torch.log_softmax
torch_var = torch.var


def forward_torch_tensor(decision_tensor):
    return type(decision_tensor) == torch.Tensor


def find_sigmoidtable(x_int, sigmoid_table):
    x_int_uint8 = x_int + 256
    y_int = torch.where(x_int >= 0, x_int, x_int_uint8)
    y_int = y_int.reshape(-1)
    for i, ele in enumerate(y_int):
        y_int[i] = sigmoid_table[ele]

    return y_int.reshape(x_int.shape)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class channelShuffle(torch.autograd.Function):
    @staticmethod
    def forward(self, input, groups):
        self.save_for_backward(input)
        self.groups = groups
        y = channel_shuffle(input, groups)
        return from_torch_tensor(y, input.scale_data, input.bits)

    @staticmethod
    def backward(self, s):
        x, = self.saved_tensors
        groups = self.groups
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = channel_shuffle(x, groups)
            grad = torch.autograd.grad(y, x, s)
        return grad[0], None

    @staticmethod
    def symbolic(g, x, groups):
        param_dict = dict()
        input_list = [x, ]
        param_dict['groups_i'] = groups

        return g.op("thinker::ShuffleChannel", *input_list, **param_dict)

def channel_shuffle_quant(*args, **kwargs):
    if forward_torch_tensor(args[0]):
        return channel_shuffle(*args, **kwargs)
    assert isinstance(args[0], IQTensor)
    assert hasattr(args[0], 'scale_data')
    assert hasattr(args[0], 'bits')
    return channelShuffle.apply(*args)

class iqRelu(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        y = torch_relu(input)
        return from_torch_tensor(y, input.scale_data, input.bits)

    @staticmethod
    def backward(self, s):
        x, = self.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = torch_relu(x)
            grad = torch.autograd.grad(y, x, s)
        return grad

    @staticmethod
    def symbolic(g, x):

        input_list = [x, ]

        return g.op("Relu", *input_list)


def _constant_pad_nd(g, input, padding, value=None):
    mode = "constant"
    pad = _prepare_onnx_paddings(g, input.type().dim(), padding)
    return g.op("Pad", input, pad, value, mode_s=mode)


class iqPad(torch.autograd.Function):
    @staticmethod
    def forward(self, scale_data, input, pad, mode='constant', value=0.):
        self.save_for_backward(input)
        self.pad = pad
        self.mode = mode
        self.value = value
        y = torch_pad(input, pad, mode, value)
        if value == 0:
            return from_torch_tensor(y, input.scale_data, input.bits)
        else:
            return y

    @staticmethod
    def backward(self, s):
        x, = self.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = torch_pad(x, self.pad, self.mode, self.value)
            grad = torch.autograd.grad(y, (x,), s)
        return None, grad[0], None, None, None

    @staticmethod
    def symbolic(g, scale_data, input, padding, mode='constant', value=0.):
        if mode == "constant":
            if value == 0:
                value = Quant.quant(torch.tensor(
                    value), bits=8, scale=scale_data)[0].char()
                padding = g.op("Constant", value_t=torch.tensor(
                    padding, dtype=torch.int64))
                return _constant_pad_nd(g, input, padding, value)
            else:
                value = torch.tensor(value)
                padding = g.op("Constant", value_t=torch.tensor(
                    padding, dtype=torch.int64))
                return constant_pad_nd(g, input, padding, value)
        elif mode == "reflect":
            padding = g.op("Constant", value_t=torch.tensor(
                padding, dtype=torch.int64))
            return reflection_pad(g, input, padding)
        elif mode == "replicate":
            padding = g.op("Constant", value_t=torch.tensor(
                padding, dtype=torch.int64))
            return replication_pad(g, input, padding)


class iqMaxPool2d(torch.autograd.Function):
    @staticmethod
    def forward(self, input, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
        # venus limits
        assert input.bits in (
            4, 8), f"in iqMaxPool2d op, input bits only support 4/8 bits, but you have input bits {input.bits}"

        self.save_for_backward(input)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        y = torch_max_pool2d(input, kernel_size, stride,
                             padding, dilation, ceil_mode)
        return from_torch_tensor(y, input.scale_data, input.bits)

    @staticmethod
    def backward(self, s):
        x, = self.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding
        dilation = self.dilation
        ceil_mode = self.ceil_mode
        grad = None
        with torch.enable_grad():
            y = torch_max_pool2d(x, kernel_size, stride,
                                 padding, dilation, ceil_mode)
            grad = torch.autograd.grad(y, x, s)
        return grad[0], None, None, None, None, None


iqMaxPool2d.symbolic = onnx_syms_max_pool2d


class iqCat(torch.autograd.Function):
    @staticmethod
    def forward(self, local_scale_o, scale_o, running_o, dim, quant_mode, training, prefix, dump, path, *args):
        tensors = args[0:len(args)//2]
        scale_s = args[len(args)//2:]
        self.save_for_backward(*tensors)
        self.dim = dim
        scale_z_iq = local_scale_o
        momentum = 0.1
        if training:
            running_o.mul_(1-momentum).add_(momentum*(127/scale_z_iq()))
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
        if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
            list_tensor = []
            for m in tensors:
                list_tensor.append(m.scale_to(scale_z_iq(), training))
            y_float = torch_cat(list_tensor, dim)
        else:
            assert False, "linger only support luna quant."
        if dump:
            name_list = ["input", "outputs"]
            attr_list = [tensors, y_float]
            Dump.dump_file(prefix, ".iqCat.", zip(name_list, attr_list), path)
        return from_torch_tensor(y_float, scale_z_iq(), 8)

    @staticmethod
    def backward(self, s):
        tensors = self.saved_tensors
        tensors = [tensor.detach().clone().requires_grad_(True)
                   for tensor in tensors]
        dim = self.dim
        grad = None
        with torch.enable_grad():
            y = torch_cat(tensors, dim)
            grad = torch.autograd.grad(y, tensors, s)
        ret = [None, None, None, None, None, None, None, None, None]+list(grad)
        l = [None for _ in range(len(tensors))]
        return tuple(ret + l)

    @staticmethod
    def symbolic(g, local_scale_o, scale_o, running_o, dim, quant_mode, training, prefix, dump, path, *args):
        param_dict = {}
        input_list = args[0:len(args)//2]
        for i, value in enumerate(args[len(args)//2:]):
            param_dict['scale_x_'+str(i)+"_f"] = value()
        assert len(input_list) == len(param_dict)
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict['scale_o_f'] = scale_o()
        param_dict['dim_i'] = dim
        param_dict['platform_quant_s'] = platform_quant

        return g.op("thinker::iqCat", *input_list, **param_dict)


def relu(*args, **kwargs):
    if forward_torch_tensor(args[0]):
        return torch_relu(*args, **kwargs)
    assert isinstance(args[0], IQTensor)
    assert hasattr(args[0], 'scale_data')
    assert hasattr(args[0], 'bits')
    return iqRelu.apply(args[0])


def relu_(*args, **kwargs):
    return relu(*args, **kwargs)


def max_pool2d(input, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
    if forward_torch_tensor(input):
        return torch_max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    assert isinstance(input, IQTensor)
    assert hasattr(input, 'scale_data')
    assert hasattr(input, 'bits')
    return iqMaxPool2d.apply(input, kernel_size, stride, padding, dilation, ceil_mode)


def pad(*args, **kwargs):
    if forward_torch_tensor(args[0]):
        return torch_pad(*args, **kwargs)
    assert isinstance(args[0], IQTensor)
    assert hasattr(args[0], 'scale_data')
    assert hasattr(args[0], 'bits')
    return iqPad.apply(args[0].scale_data, *args)


class iqCatLayer(torch.nn.Module):
    def __init__(self):
        super(iqCatLayer, self).__init__()
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.register_buffer('scale_o', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))

    def forward(self, tensors, local_scale_o, dim, quant_mode=QuantMode.QValue):
        parmater_to_function = []
        for s in tensors:
            parmater_to_function.append(s)
        for s in tensors:
            parmater_to_function.append(ScalerBuffer(s.scale_data))

        scale_o = ScalerBuffer(self.scale_o)
        running_o = ScalerBuffer(self.running_o)
        local_scale_o = ScalerBuffer(local_scale_o)
        z = iqCat.apply(local_scale_o, scale_o, running_o, dim, quant_mode,
                        self.training, self.prefix, self.dump, self.path, *parmater_to_function)
        self.scale_o.fill_(scale_o.data)
        self.running_o.fill_(running_o.data)
        return z

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def cat(tensors, dim=0, out=None):
    is_iq_cat = True
    for t in tensors:
        if not isinstance(t, IQTensor):
            is_iq_cat = False
            break
    module_self = get_current_module()
    if module_self is None:
        is_iq_cat = False
    if not is_iq_cat:
        return torch_cat(tensors, dim, out=out)
    assert out == None, 'iqtensor not support out param cat for now please make sure the out is None'
    assert tensors[0].bits == 8, 'iqcat only support 8bit'
    quant_mode = getattr(module_self, LINGER_MODE, QuantMode.QValue)
    iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)
    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + '_iqcat_'+str(iname_index)
    if not module_self.training and not hasattr(module_self, var_name):
        logging.warning(
            'eval module has iqcat layer while do not match training module')
        return torch_cat(tensors, dim, out=out)
    iq_layer = None
    if hasattr(module_self, var_name):
        iq_layer = getattr(module_self, var_name)
    else:
        iq_layer = iqCatLayer()
        iq_layer.training = module_self.training
        iq_layer = iq_layer.to(tensors[0].device)
        setattr(module_self, var_name, iq_layer)
    scale_z = None
    with torch.no_grad():
        max_z = -1
        for m in tensors:
            max_z_t = torch.max(torch.abs(m)).item()
            if max_z < max_z_t:
                max_z = max_z_t
        if max_z == 0:
            scale_z = 1.0
        else:
            scale_z = 127 / max_z

    return iq_layer(tensors, scale_z, dim, quant_mode)


class softmaxInt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, data_bits, training, running_x, running_o, scale_x, scale_o, scale_local_o, prefix, dump, path, mode, o_bits, is_not_from_iqtensor):
        momentum = 0.1
        if training:
            ctx.save_for_backward(input)
            ctx.dim = dim
            ctx.o_bits = o_bits
            if isinstance(input, IQTensor):
                q_input, _, max_value_x = Quant().quant(input.data, data_bits, scale_x,
                                                           mode=QuantMode.QValue, quant_data='input', iq_zero_point=input.zero_point)
            else:
                q_input, _, max_value_x = Quant().quant(
                    input.data, data_bits, mode=mode, quant_data='input')
                running_x.mul_(1-momentum).add_(momentum*max_value_x)

            if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                n_dim = len(input.shape)
                # self.dim_ = n_dim - 1 if (self.dim_ == -1) else self.dim_
                dims = [i for i in range(n_dim)]
                dims[dim] = n_dim - 1
                dims[n_dim - 1] = dim
                x_ori = q_input.contiguous()
                if dim != -1 and dim != n_dim - 1:
                    x_ori = x_ori.permute(*dims)
                x_shaped = x_ori.reshape(-1, x_ori.shape[-1])

                l_scale = 25 - int(math.log2(scale_x.data))     # Q25 in
                if l_scale > 0:
                    x_int_shift = (x_shaped * pow(2, l_scale)).int()
                else:
                    x_int_shift = (
                        x_shaped * pow(2, l_scale) + 0.5).floor().int()

                q_output = lingerext.luna_softmax_int(
                    x_int_shift.contiguous(), float(scale_x()))   # Q25->Q15
                q_output.clamp_(0, 2**15-1)
                q_output = q_output.reshape(x_ori.shape)
                if dim != -1 and dim != n_dim - 1:
                    q_output = q_output.permute(*dims)
                scale_local_o.fill_(2**15)
                outputs = Quant().dequant(q_output, scale_local_o)  # Q15->float
            else:
                assert False, 'platform_quant mode donot support for softmaxInt'

            if o_bits is not None:
                q_output, scale_o, max_value_o = Quant().quant(
                    outputs, o_bits, mode=mode, quant_data='output')
                running_o.mul_(1-momentum).add_(momentum*max_value_o)
        else:
            assert running_x > 0, 'invalid running_x = 0, please finetune training before eval'
            if not isinstance(input, IQTensor):
                scale_x = ScalerBuffer(Quant().running_to_scale(running_x, data_bits, mode=mode))
            if o_bits is not None:
                assert running_o > 0, 'invalid running_o = 0 for softmaxInt'
                scale_o = ScalerBuffer(Quant().running_to_scale(running_o, o_bits, mode=mode))
                # scale_o.fill_(2**31)
     
            q_input, _, _ = Quant().quant(input.data, data_bits,
                                             scale_x, mode=mode, quant_data='input')
            if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                n_dim = len(input.shape)
                # self.dim_ = n_dim - 1 if (self.dim_ == -1) else self.dim_
                dims = [i for i in range(n_dim)]
                dims[dim] = n_dim - 1
                dims[n_dim - 1] = dim
                x_ori = q_input.contiguous()
                if dim != -1 and dim != n_dim - 1:
                    x_ori = x_ori.permute(*dims)
                x_shaped = x_ori.reshape(-1, x_ori.shape[-1])

                l_scale = 25 - int(math.log2(scale_x.data))     # Q25 in
                if l_scale > 0:
                    x_int_shift = (x_shaped * pow(2, l_scale)).int()
                else:
                    x_int_shift = (
                        x_shaped * pow(2, l_scale) + 0.5).floor().int()

                q_output = lingerext.luna_softmax_int(
                    x_int_shift.contiguous(), float(scale_x()))   # Q25->Q15
                q_output.clamp_(0, 2**15-1)
                q_output = q_output.reshape(x_ori.shape)
                if dim != -1 and dim != n_dim - 1:
                    q_output = q_output.permute(*dims)
                scale_local_o.fill_(2**15)
                outputs = Quant().dequant(q_output, scale_local_o)  # Q15->float
            else:
                assert False, 'platform_quant mode donot support for softmaxInt'

            if o_bits is not None:
                q_output, _, _ = Quant().quant(outputs, o_bits, scale_o,
                                                  mode=mode, quant_data='output')
                outputs = Quant().dequant(q_output, scale_o)

            if dump:
                name_list = ['input',  'outputs', 'q_input',  'q_outputs']
                attr_list = [input,  outputs, q_input,  q_output]
                Dump.dump_file(prefix, '.SoftmaxInt.',
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
    def backward(ctx, s):
        x, = ctx.saved_tensors
        dim = ctx.dim
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = torch_softmax(x, dim=dim)
            grad = torch.autograd.grad(y, x, s)
        return grad[0], None, None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, input, dim, data_bits, training,
                 running_x, running_o, scale_x, scale_o, scale_local_o,
                 prefix, dump, path, mode, o_bits, is_not_from_iqtensor):
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        if is_not_from_iqtensor:
            op_inner = quantlinear(g, input, scale_x(),
                                   platform_quant, data_bits)
        param_dict = {'scale_x_f': scale_x(
        ),  'data_bits_i': data_bits, 'dim_i': dim}
        input_list = []
        if is_not_from_iqtensor:
            input_list.append(op_inner)
        else:
            input_list.append(input)

        if o_bits is not None:
            param_dict['scale_o_f'] = scale_o()
            param_dict['o_bits_i'] = o_bits
        param_dict['platform_quant_s'] = platform_quant

        return g.op("thinker::SoftmaxInt", *input_list, **param_dict)


class softmaxIntLayer(torch.nn.Module):
    def __init__(self, data_bits=8, mode=QuantMode.QValue, o_bits=8):
        super(softmaxIntLayer, self).__init__()
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.data_bits = data_bits
        self.o_bits = o_bits
        self.mode = mode
        self.is_not_from_iqtensor = True
        self.register_buffer('running_x', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))
        self.register_buffer('scale_x', torch.zeros(1))
        self.register_buffer('scale_o', torch.zeros(1))
        self.register_buffer('scale_local_o', torch.zeros(1))

    def forward(self, input, dim=-1):
        running_x = ScalerBuffer(self.running_x)
        running_o = ScalerBuffer(self.running_o)
        scale_x = ScalerBuffer(self.scale_x)
        scale_o = ScalerBuffer(self.scale_o)
        scale_local_o = ScalerBuffer(self.scale_local_o)

        if isinstance(input, IQTensor):
            self.is_not_from_iqtensor = False
            if input.bits != self.data_bits:
                input = Requant.apply(
                    input, input.bits, input.scale_data, self.data_bits)
            scale_x = ScalerBuffer(input.scale_data)
            running_x = ScalerBuffer(input.running_data)

        z = softmaxInt.apply(input, dim, self.data_bits, self.training, running_x, running_o, scale_x, scale_o, scale_local_o,
                             self.prefix, self.dump, self.path, self.mode, self.o_bits, self.is_not_from_iqtensor)
        self.running_x.fill_(running_x())
        self.running_o.fill_(running_o())
        self.scale_x.fill_(scale_x())
        self.scale_o.fill_(scale_o())

        return z

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]
                              ] = local_metadata = dict(version=self._version)
        if is_in_onnx_export():
            assert self.running_x > 0, 'invalid running_x <=0'
            scale_x = ScalerBuffer(self.scale_x.data)
            if self.is_not_from_iqtensor:
                scale_x = ScalerBuffer(Quant().running_to_scale(
                    self.running_x, self.data_bits, mode=self.mode))
                self.scale_x.data.fill_(scale_x())

            if self.o_bits is not None:
                scale_o = ScalerBuffer(Quant().running_to_scale(
                    self.running_o, self.o_bits, mode=self.mode))
                # scale_o.fill_(2**31)
                self.scale_o.data.fill_(scale_o())
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


def softmax(tensor, dim, dtype=None):
    is_softmax_int = True
    if not isinstance(tensor, IQTensor):
        is_softmax_int = False
    module_self = get_current_module()
    if module_self is None:
        is_softmax_int = False
    if not is_softmax_int:
        scale_o = 1
        tensor.clamp_(-128, 127) 
        min_x = torch.min(tensor)
        max_x = torch.max(tensor)
        if min_x == max_x == 0:
            scale_o = math.pow(2, 8)
        else:
            max_abs = torch.max(-min_x, max_x)
            max_value = round(math.log((127) / max_abs, 2))
            scale_o = math.pow(2, max_value)
        tensor = from_torch_tensor(tensor, scale_o, 8)
        # return torch_softmax(tensor, dim=dim)
    # assert tensor.bits == 8, 'softmaxInt only support 8bit'
    quant_mode = getattr(module_self, LINGER_MODE, QuantMode.QValue)
    iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)

    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + \
        '_SoftmaxInt_'+str(iname_index)
    iq_layer = None
    if hasattr(module_self, var_name):
        iq_layer = getattr(module_self, var_name)
    else:
        iq_layer = softmaxIntLayer(mode=quant_mode)
        iq_layer.training = module_self.training
        iq_layer = iq_layer.to(tensor.device)
        setattr(module_self, var_name, iq_layer)
    # iq_layer.o_bits = 32
    return iq_layer(tensor, dim=dim)


class logsoftmaxInt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, data_bits, training, running_x, running_o, scale_x, scale_o, scale_local_o, prefix, dump, path, mode, o_bits, is_not_from_iqtensor):
        momentum = 0.1
        if training:
            ctx.save_for_backward(input)
            ctx.dim = dim
            ctx.o_bits = o_bits
            if isinstance(input, IQTensor):
                q_input, _, max_value_x = Quant().quant(input.data, data_bits, scale_x,
                                                           mode=QuantMode.QValue, quant_data='input', iq_zero_point=input.zero_point)
            else:
                q_input, _, max_value_x = Quant().quant(
                    input.data, data_bits, mode=mode, quant_data='input')
                running_x.mul_(1-momentum).add_(momentum*max_value_x)

            if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                n_dim = len(input.shape)
                # self.dim_ = n_dim - 1 if (self.dim_ == -1) else self.dim_
                dims = [i for i in range(n_dim)]
                dims[dim] = n_dim - 1
                dims[n_dim - 1] = dim
                x_ori = q_input.contiguous()
                if dim != -1 and dim != n_dim - 1:
                    x_ori = x_ori.permute(*dims)
                x_shaped = x_ori.reshape(-1, x_ori.shape[-1])

                l_scale = 25 - int(math.log2(scale_x.data))     # Q25 in
                if l_scale > 0:
                    x_int_shift = (x_shaped * pow(2, l_scale)).int()
                else:
                    x_int_shift = (
                        x_shaped * pow(2, l_scale) + 0.5).floor().int()
                q_output = lingerext.luna_logsoftmax_int(
                    x_int_shift.contiguous(), float(scale_x()))   # Q25->Q25
                q_output.clamp_(-2**31, 0)
                q_output = q_output.reshape(x_ori.shape)
                if dim != -1 and dim != n_dim - 1:
                    q_output = q_output.permute(*dims)
                scale_local_o.fill_(2**25)
                outputs = Quant().dequant(q_output, scale_local_o)  # Q25->float
            else:
                assert False, 'platform_quant mode donot support for logsoftmaxInt'

            if o_bits is not None:
                q_output, scale_o, max_value_o = Quant().quant(
                    outputs, o_bits, mode=mode, quant_data='output')
                running_o.mul_(1-momentum).add_(momentum*max_value_o)
        else:
            assert running_x > 0, 'invalid running_x = 0, please finetune training before eval'
            if not isinstance(input, IQTensor):
                scale_x = ScalerBuffer(Quant().running_to_scale(
                    running_x, data_bits, mode=mode))
            if o_bits is not None:
                assert running_o > 0, 'invalid running_o = 0 for logsoftmaxInt'
                scale_o = ScalerBuffer(Quant().running_to_scale(
                    running_o, o_bits, mode=mode))
                if scale_o.data > float(2**31):
                    scale_o.fill_(2**31)
            q_input, _, _ = Quant().quant(input.data, data_bits,
                                             scale_x, mode=mode, quant_data='input')
            if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                n_dim = len(input.shape)
                # self.dim_ = n_dim - 1 if (self.dim_ == -1) else self.dim_
                dims = [i for i in range(n_dim)]
                dims[dim] = n_dim - 1
                dims[n_dim - 1] = dim
                x_ori = q_input.contiguous()
                if dim != -1 and dim != n_dim - 1:
                    x_ori = x_ori.permute(*dims)
                x_shaped = x_ori.reshape(-1, x_ori.shape[-1])

                l_scale = 25 - int(math.log2(scale_x.data))     # Q25 in
                if l_scale > 0:
                    x_int_shift = (x_shaped * pow(2, l_scale)).int()
                else:
                    x_int_shift = (
                        x_shaped * pow(2, l_scale) + 0.5).floor().int()

                q_output = lingerext.luna_logsoftmax_int(
                    x_int_shift.contiguous(), float(scale_x()))   # Q25->Q25
                q_output.clamp_(-2**31, 0)
                q_output = q_output.reshape(x_ori.shape)
                if dim != -1 and dim != n_dim - 1:
                    q_output = q_output.permute(*dims)
                scale_local_o.fill_(2**25)
                outputs = Quant().dequant(q_output, scale_local_o)  # Q25->float
            else:
                assert False, 'platform_quant mode donot support for logsoftmaxInt'

            if o_bits is not None:
                q_output, _, _ = Quant().quant(outputs, o_bits, scale_o,
                                                  mode=mode, quant_data='output')
                outputs = Quant().dequant(q_output, scale_o)

            if dump:
                name_list = ['input',  'outputs', 'q_input',  'q_outputs']
                attr_list = [input,  outputs, q_input,  q_output]
                Dump.dump_file(prefix, '.SoftmaxInt.',
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
    def backward(ctx, s):
        x, = ctx.saved_tensors
        dim = ctx.dim
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = torch_logsoftmax(x, dim=dim)
            grad = torch.autograd.grad(y, x, s)
        return grad[0], None, None, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, input, dim, data_bits, training,
                 running_x, running_o, scale_x, scale_o, scale_local_o,
                 prefix, dump, path, mode, o_bits, is_not_from_iqtensor):
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        if is_not_from_iqtensor:
            op_inner = quantlinear(g, input, scale_x(),
                                   platform_quant, data_bits)
        param_dict = {'scale_x_f': scale_x(
        ),  'data_bits_i': data_bits, 'dim_i': dim}
        input_list = []
        if is_not_from_iqtensor:
            input_list.append(op_inner)
        else:
            input_list.append(input)
        if o_bits is not None:
            param_dict['scale_o_f'] = scale_o()
            param_dict['o_bits_i'] = o_bits
        param_dict['platform_quant_s'] = platform_quant
        return g.op("thinker::LogSoftmaxInt", *input_list, **param_dict)


class logsoftmaxIntLayer(torch.nn.Module):
    def __init__(self, data_bits=8, mode=QuantMode.QValue, o_bits=8):
        super(logsoftmaxIntLayer, self).__init__()
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.data_bits = data_bits
        self.o_bits = o_bits
        self.mode = mode
        self.is_not_from_iqtensor = True
        self.register_buffer('running_x', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))
        self.register_buffer('scale_x', torch.zeros(1))
        self.register_buffer('scale_o', torch.zeros(1))
        self.register_buffer('scale_local_o', torch.zeros(1))

    def forward(self, input, dim=-1):
        running_x = ScalerBuffer(self.running_x)
        running_o = ScalerBuffer(self.running_o)
        scale_x = ScalerBuffer(self.scale_x)
        scale_o = ScalerBuffer(self.scale_o)
        scale_local_o = ScalerBuffer(self.scale_local_o)

        if isinstance(input, IQTensor):
            self.is_not_from_iqtensor = False
            if input.bits != self.data_bits:
                input = Requant.apply(
                    input, input.bits, input.scale_data, self.data_bits)
            scale_x = ScalerBuffer(input.scale_data)
            running_x = ScalerBuffer(input.running_data)

        z = logsoftmaxInt.apply(input, dim, self.data_bits, self.training, running_x, running_o, scale_x, scale_o, scale_local_o,
                                self.prefix, self.dump, self.path, self.mode, self.o_bits, self.is_not_from_iqtensor)
        self.running_x.fill_(running_x())
        self.running_o.fill_(running_o())
        self.scale_x.fill_(scale_x())
        self.scale_o.fill_(scale_o())

        return z

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]
                              ] = local_metadata = dict(version=self._version)
        if is_in_onnx_export():
            assert self.running_x > 0, 'invalid running_x <=0'
            scale_x = ScalerBuffer(self.scale_x.data)
            if self.is_not_from_iqtensor:
                scale_x = ScalerBuffer(Quant().running_to_scale(
                    self.running_x, self.data_bits, mode=self.mode))
                self.scale_x.data.fill_(scale_x())
            if self.o_bits is not None:
                scale_o = ScalerBuffer(Quant().running_to_scale(
                    self.running_o, self.o_bits, mode=self.mode))
                if scale_o.data > float(2**31):
                    scale_o.fill_(2**31)
                self.scale_o.data.fill_(scale_o())
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


def logsoftmax(tensor, dim, dtype=None):
    is_logsoftmax_int = True
    if not isinstance(tensor, IQTensor):
        is_logsoftmax_int = False
    module_self = get_current_module()
    if module_self is None:
        is_logsoftmax_int = False
    if not is_logsoftmax_int:
        return torch_logsoftmax(tensor, dim=dim)
    # assert tensor.bits == 8, 'LogsoftmaxInt only support 8bit'
    quant_mode = getattr(module_self, LINGER_MODE, QuantMode.QValue)
    iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)

    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + \
        '_LogSoftmaxInt_'+str(iname_index)
    iq_layer = None
    if hasattr(module_self, var_name):
        iq_layer = getattr(module_self, var_name)
    else:
        iq_layer = logsoftmaxIntLayer(mode=quant_mode)
        iq_layer.training = module_self.training
        iq_layer = iq_layer.to(tensor.device)
        setattr(module_self, var_name, iq_layer)
    # iq_layer.o_bits = 32

    return iq_layer(tensor, dim=dim)


class iqSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale_x, local_scale_o, running_o, scale_o, training, quant_mode, prefix, dump, path, data_bits):
        ctx.save_for_backward(x)
        # x_int = x.quant_to_int8(scale_x())
        # x_int = x_int.contiguous().int()
        scale_z_iq = local_scale_o
        momentum = 0.1
        bound_value = 127

        if training:
            running_o.mul_(1-momentum).add_(momentum *
                                            (bound_value/local_scale_o()))
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(scale_z_iq(), 2))
                scale_z_iq = math.pow(2, scale_log)
            scale_z_iq = ScalerBuffer(scale_z_iq)
        else:
            assert running_o.data > 0, 'Must at least training one batch'
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(bound_value/running_o.data, 2))
                scale_z_iq = math.pow(2, scale_log)
            else:
                scale_z_iq = np.float32(bound_value / running_o.data)
            scale_z_iq = ScalerBuffer(scale_z_iq)
            scale_o.fill_(scale_z_iq())
        y_int = None

        if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
            q_input, _, _ = Quant().quant(
                x.data, data_bits, scale_x, mode=QuantMode.QValue, quant_data='input', iq_zero_point=x.zero_point)
            l_scale = 11 - int(math.log2(scale_x.data))

            if l_scale > 0:
                x_int_shift = (q_input * pow(2, l_scale)).int()
            else:
                x_int_shift = (q_input * pow(2, l_scale) + 0.5).floor().int()

            y_int = lingerext.luna_iqsigmoid(
                x_int_shift.contiguous(), float(scale_x()))
            y_int.clamp_(0, 2**7-1)
            scale_z_iq.fill_(2**7)
            scale_o.fill_(scale_z_iq())
            running_o.fill_(1.0)
            y_float = Quant.dequant(y_int, scale_z_iq)

            if dump:
                name_list = ['input',  'outputs', 'q_input',  'q_outputs']
                attr_list = [x,  y_float, q_input,  y_int]
                Dump.dump_file(prefix, '.iqSigmoid.',
                                zip(name_list, attr_list), path)

            return from_torch_tensor(y_float, scale_z_iq(), 8, zero_point=0)
        else:
            assert False, 'platform_quant mode donot support for iqSigmoid'

    @staticmethod
    def backward(ctx, s):
        x, = ctx.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = torch_sigmoid(x)
            grad = torch.autograd.grad(y, x, s)
        return grad[0], None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, scale_x, local_scale_o, running_o, scale_o, training, quant_mode, prefix, dump, path, data_bits):
        param_dict = {'scale_x_f': scale_x(), 'scale_o_f': scale_o()}
        input_list = [x, ]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict['platform_quant_s'] = platform_quant
        param_dict['castor_mode_s'] = "luna"
        op = None
        op = g.op("thinker::iqSigmoid", *input_list, **param_dict)
        return op


class iqSigmoidLayer(torch.nn.Module):
    def __init__(self, data_bits=16):
        super(iqSigmoidLayer, self).__init__()
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.data_bits = data_bits
        self.register_buffer('scale_o', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))

    def forward(self, input, local_scale_o, quant_mode=QuantMode.QValue):
        scale_x = ScalerBuffer(input.scale_data)
        local_scale_o = ScalerBuffer(local_scale_o)
        scale_o = ScalerBuffer(self.scale_o)
        running_o = ScalerBuffer(self.running_o)
        if isinstance(input, IQTensor):
            input = Requant.apply(
                input, input.bits, input.scale_data, self.data_bits)
            scale_x = ScalerBuffer(input.scale_data)
        z = iqSigmoid.apply(input, scale_x, local_scale_o, running_o, scale_o,
                            self.training, quant_mode, self.prefix, self.dump, self.path, self.data_bits)
        self.scale_o.fill_(scale_o.data)
        self.running_o.fill_(running_o.data)
        return z
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def sigmoid(tensor, *, out=None):
    is_iq_sigmoid = True
    if not isinstance(tensor, IQTensor):
        is_iq_sigmoid = False
    module_self = get_current_module()
    if module_self is None:
        is_iq_sigmoid = False
    if not is_iq_sigmoid:
        return torch_sigmoid(tensor)
    # assert tensor.bits == 8, 'iqsigmoid only support 8bit'
    quant_mode = getattr(module_self, LINGER_MODE, QuantMode.QValue)
    iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)

    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + \
        '_iqsigmoid_'+str(iname_index)
    iq_layer = None
    if hasattr(module_self, var_name):
        iq_layer = getattr(module_self, var_name)
    else:
        iq_layer = iqSigmoidLayer()
        iq_layer.training = module_self.training
        iq_layer = iq_layer.to(tensor.device)
        setattr(module_self, var_name, iq_layer)
    scale_z = None
    with torch.no_grad():
        z_f = torch_sigmoid(tensor)
        max_z = torch.max(torch.abs(z_f))
        if max_z == 0:
            scale_z = 1.0
        else:
            scale_z = 127 / max_z.item()
    return iq_layer(tensor, scale_z, quant_mode)


def sigmoid_(tensor, *, out=None):
    return sigmoid(tensor, out)


class iqTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale_x, local_scale_o, running_o, scale_o, training, quant_mode, prefix, dump, path, data_bits):
        ctx.save_for_backward(x)
        # x_int = x.quant_to_int8(scale_x())
        # x_int = x_int.contiguous().int()
        scale_z_iq = local_scale_o
        momentum = 0.1
        bound_value = 127

        if training:
            running_o.mul_(1-momentum).add_(momentum *
                                            (bound_value/local_scale_o()))
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(scale_z_iq(), 2))
                scale_z_iq = math.pow(2, scale_log)
            scale_z_iq = ScalerBuffer(scale_z_iq)
        else:
            assert running_o.data > 0, 'Must at least training one batch'
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(bound_value/running_o.data, 2))
                scale_z_iq = math.pow(2, scale_log)
            else:
                scale_z_iq = np.float32(bound_value / running_o.data)
            scale_z_iq = ScalerBuffer(scale_z_iq)
            scale_o.fill_(scale_z_iq())
        y_int = None

        if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
            q_input, _, _ = Quant().quant(
                x.data, data_bits, scale_x, mode=QuantMode.QValue, quant_data='input', iq_zero_point=x.zero_point)
            l_scale = 11 - int(math.log2(scale_x.data))

            if l_scale > 0:
                x_int = (q_input * pow(2, l_scale)).int()
            else:
                x_int = (q_input * pow(2, l_scale) + 0.5).floor().int()
            y_int = lingerext.luna_iqtanh(x_int.contiguous(), float(scale_x()))
            scale_z_iq.fill_(2**7)
            scale_o.fill_(scale_z_iq())
            running_o.fill_(1.0)
            y_float = Quant.dequant(y_int, scale_z_iq)
            return from_torch_tensor(y_float, scale_z_iq(), 8, zero_point=0)

        else:
            assert False, "linger only support luna quant."

    @staticmethod
    def backward(ctx, s):
        x, = ctx.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = torch_tanh(x)
            grad = torch.autograd.grad(y, x, s)
        return grad[0], None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, scale_x, local_scale_o, running_o, scale_o, training, quant_mode, prefix, dump, path, data_bits):
        param_dict = {'scale_x_f': scale_x(), 'scale_o_f': scale_o()}
        input_list = [x, ]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict['platform_quant_s'] = platform_quant
        param_dict['castor_mode_s'] = "luna"
        op = None
        op = g.op("thinker::iqTanh", *input_list, **param_dict)
        return op


class iqTanhLayer(torch.nn.Module):
    def __init__(self, data_bits=16):
        super(iqTanhLayer, self).__init__()
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.data_bits = data_bits
        self.register_buffer('scale_o', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))

    def forward(self, input, local_scale_o, quant_mode=QuantMode.QValue):
        scale_x = ScalerBuffer(input.scale_data)
        local_scale_o = ScalerBuffer(local_scale_o)
        scale_o = ScalerBuffer(self.scale_o)
        running_o = ScalerBuffer(self.running_o)
        if isinstance(input, IQTensor):
            input = Requant.apply(
                input, input.bits, input.scale_data, self.data_bits)
            scale_x = ScalerBuffer(input.scale_data)
        z = iqTanh.apply(input, scale_x, local_scale_o, running_o, scale_o,
                         self.training, quant_mode, self.prefix, self.dump, self.path, self.data_bits)
        self.scale_o.fill_(scale_o.data)
        self.running_o.fill_(running_o.data)
        return z
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def tanh(tensor, *, out=None):
    is_iq_tanh = True
    if not isinstance(tensor, IQTensor):
        is_iq_tanh = False
    module_self = get_current_module()
    if module_self is None:
        is_iq_tanh = False
    if not is_iq_tanh:
        return torch_tanh(tensor)
    # assert tensor.bits == 8, 'iqTanh only support 8bit'
    quant_mode = getattr(module_self, LINGER_MODE, QuantMode.QValue)
    iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)

    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + \
        '_iqtanh_'+str(iname_index)
    iq_layer = None
    if hasattr(module_self, var_name):
        iq_layer = getattr(module_self, var_name)
    else:
        iq_layer = iqTanhLayer()
        iq_layer.training = module_self.training
        iq_layer = iq_layer.to(tensor.device)
        setattr(module_self, var_name, iq_layer)
    scale_z = None
    with torch.no_grad():
        z_f = torch_tanh(tensor)
        max_z = torch.max(torch.abs(z_f))
        if max_z == 0:
            scale_z = 1.0
        else:
            scale_z = 127 / max_z.item()

    return iq_layer(tensor, scale_z, quant_mode)


def tanh_(tensor, *, out=None):
    return tanh(tensor, out)


class iqClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min, max, scale_x, local_scale_o, running_o, scale_o, training, quant_mode, prefix, dump, path):
        ctx.save_for_backward(x)
        ctx.min = min
        ctx.max = max
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
                scale_z_iq = np.float32(127 / running_o.data)
            scale_z_iq = ScalerBuffer(scale_z_iq)
            scale_o.fill_(scale_z_iq())
        y_int = None
        y_float = torch_clamp(x, min, max)
        if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
            Qx = math.log(127/max, 2)
            assert Qx.is_integer(
            ), "luna_quant max value don't support {} clamp, it must be (127 /2^n).".format(max)
            assert (min == -128/2**math.log(127/7.9375, 2)) or (min ==
                                                                0), "luna_quant min value don't support {} clamp, it must match with the max-value (-128/2^n) or 0.".format(min)
            scale_z_iq = 2**Qx
            scale_z_iq = ScalerBuffer(scale_z_iq)
            y_int = (x * scale_z_iq + 0.5).floor().int()
            if min == 0:
                y_int = torch_clamp(y_int, 0, 127)
            scale_o.fill_(scale_z_iq())

        else:
            assert False, 'platform_quant mode donot support for iqClamp'
        y_int = torch_clamp(y_int, -128, 127)
        y_float = Quant.dequant(y_int, scale_z_iq)
        if dump:
            name_list = ["input", "outputs", "q_outputs"]
            attr_list = [x, y_float, y_int]
            Dump.dump_file(prefix, ".iqClamp.", zip(
                name_list, attr_list), path)
        return from_torch_tensor(y_float, scale_z_iq(), 8)

    @staticmethod
    def backward(ctx, s):
        x, = ctx.saved_tensors
        min = ctx.min
        max = ctx.max
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = torch_clamp(x, min, max)
            grad = torch.autograd.grad(y, (x,), s)
        return grad[0], None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, min, max, scale_x, local_scale_o, running_o, scale_o, training, quant_mode, prefix, dump, path):
        param_dict = {'min_f': min, 'max_f': max,
                      'scale_x_f': scale_x(), 'scale_o_f': scale_o()}
        input_list = [x, ]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict['platform_quant_s'] = platform_quant

        return g.op("thinker::iqClamp", *input_list, **param_dict)


class iqClampLayer(torch.nn.Module):
    def __init__(self):
        super(iqClampLayer, self).__init__()
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.register_buffer('scale_o', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))

    def forward(self, input, min, max, local_scale_o, quant_mode=QuantMode.QValue):
        scale_x = ScalerBuffer(input.scale_data)
        local_scale_o = ScalerBuffer(local_scale_o)
        scale_o = ScalerBuffer(self.scale_o)
        running_o = ScalerBuffer(self.running_o)
        z = iqClamp.apply(input, min, max, scale_x, local_scale_o, running_o,
                          scale_o, self.training, quant_mode, self.prefix, self.dump, self.path)
        self.scale_o.fill_(scale_o.data)
        self.running_o.fill_(running_o.data)
        return z
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


def clamp(tensor, min=-math.inf, max=math.inf, out=None):
    is_iq_clamp = True
    if not isinstance(tensor, IQTensor):
        is_iq_clamp = False
    module_self = get_current_module()
    if module_self is None:
        is_iq_clamp = False
    if not is_iq_clamp:
        return torch_clamp(tensor, min, max)
    assert tensor.bits == 8, 'iqclamp only support 8bit'
    quant_mode = getattr(module_self, LINGER_MODE, QuantMode.QValue)
    iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)
    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + \
        '_iqclamp_'+str(iname_index)
    iq_layer = None
    if hasattr(module_self, var_name):
        iq_layer = getattr(module_self, var_name)
    else:
        iq_layer = iqClampLayer()
        iq_layer.training = module_self.training
        iq_layer = iq_layer.to(tensor.device)
        setattr(module_self, var_name, iq_layer)
    scale_z = None
    with torch.no_grad():
        z_f = torch_clamp(tensor, min, max)
        max_z = torch.max(torch.abs(z_f))
        if max_z == 0:
            scale_z = 1.0
        else:
            scale_z = 127 / max_z.item()
    return iq_layer(tensor, min, max, scale_z, quant_mode)


def clamp_(tensor, min=-math.inf, max=math.inf, out=None):
    return clamp(tensor, min, max, out)


def dropout(tensor, p: float, training: bool = False, inplace: bool = False):
    return torch_dropout(tensor, p, False, inplace)


def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    return input, lengths, batch_first, enforce_sorted


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    assert (padding_value == 0.0 and total_length is None), 'lstmint for pad_packed only support padding_value=0.0 and total_length=None'
    output, lengths = sequence
    return output, lengths


def transpose(*args, **kwargs):
    if forward_torch_tensor(args[0]):
        return torch_transpose(*args, **kwargs)
    assert isinstance(args[0], IQTensor)
    assert hasattr(args[0], 'scale_data')
    assert hasattr(args[0], 'bits')
    return iqTranspose.apply(args[0], args[1], args[2])


class iqMax(torch.autograd.Function):
    @staticmethod
    def forward(self, input, other, scale_x, scale_y, scale_o):
        self.save_for_backward(*(input, other))
        y = torch_max(input, other)

        return from_torch_tensor(y, scale_o, input.bits, input.zero_point)

    @staticmethod
    def backward(self, s):
        x, y = self.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)

        grad = None
        with torch.enable_grad():
            z = torch_max(x, y)
            grad = torch.autograd.grad(z, (x, y), s)
        return grad[0], grad[1], None, None, None

    @staticmethod
    def symbolic(g, input, other, scale_x, scale_y, scale_o):
        # torch.max(input, other)
        param_dict = {'scale_x_f': scale_x,
                      'scale_y_f': scale_y, 'scale_o_f': scale_o}
        input_list = [input, other, ]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict['platform_quant_s'] = platform_quant

        op = g.op("thinker::iqMax", *input_list, **param_dict)
        return op


def iqmax(*args, **kwargs):
    if len(args) != 2:
        return torch_max(*args, **kwargs)
    if not isinstance(args[0], IQTensor):
        return torch_max(*args, **kwargs)
    if not isinstance(args[1], IQTensor):
        return torch_max(*args, **kwargs)
    if len(kwargs) != 0:
        return torch_max(*args, **kwargs)
    assert isinstance(args[0], IQTensor)
    assert isinstance(args[1], IQTensor)

    assert hasattr(args[0], 'scale_data')
    assert hasattr(args[0], 'bits')
    assert hasattr(args[1], 'scale_data')
    assert hasattr(args[1], 'bits')

    max_scale_o = min(args[0].scale_data, args[1].scale_data)
    return iqMax.apply(args[0], args[1], args[0].scale_data, args[1].scale_data, max_scale_o)


def bmm(input, mat2, *, out=None):
    module_self = get_current_module()

    quant_mode = getattr(module_self, LINGER_MODE, QuantMode.QValue)

    out_bits = getattr(module_self, LINGER_OBIT,
                       None) if True else None
    iname_index = getattr(module_self, LINGER_FUNCINT_BMM_COUNTER)
    setattr(module_self, LINGER_FUNCINT_BMM_COUNTER, iname_index+1)
    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + \
        '_function_bmm_'+str(iname_index)
    quant_layer = None
    bmm_output = None
    if hasattr(module_self, var_name):
        quant_layer = getattr(module_self, var_name)
    else:
        quant_layer = BmmInt(data_bits=8, mode=quant_mode,)
        quant_layer.training = module_self.training
        quant_layer = quant_layer.to(input.device)
        setattr(module_self, var_name, quant_layer)
    quant_layer.clamp_data = None
    quant_layer.o_bits = out_bits
    bmm_output = quant_layer(input, mat2)
    return bmm_output


class iqVar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, unbiased, keepdim, scale_x, local_scale_o, running_o, scale_o, training, quant_mode, prefix, dump, path):
        ctx.save_for_backward(x)
        ctx.value = dim, unbiased, keepdim
        x_int = x.quant_to_int8(scale_x())
        x_int = x_int.contiguous()
        scale_z_iq = local_scale_o
        momentum = 0.1
        bound_value = 127

        if training:
            running_o.mul_(1-momentum).add_(momentum * (bound_value/local_scale_o()))
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(scale_z_iq(), 2))
                scale_z_iq = math.pow(2, scale_log)
            scale_z_iq = ScalerBuffer(scale_z_iq)
        else:
            assert running_o.data > 0, 'Must at least training one batch'
            if quant_mode == QuantMode.QValue:
                scale_log = round(math.log(bound_value/running_o.data, 2))
                scale_z_iq = math.pow(2, scale_log)
            else:
                scale_z_iq = np.float32(bound_value / running_o.data)
            scale_z_iq = ScalerBuffer(scale_z_iq)
            scale_o.fill_(scale_z_iq())
        y_int = None

        if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
            x_float = Quant.dequant(x_int, scale_x)
            y_float = torch_var(x_float, dim, unbiased, keepdim)
            y_int = (y_float * scale_z_iq()).round().int()
            y_int.clamp_(-128,127)

        y_float = Quant.dequant(y_int, scale_z_iq)

        if dump:
            name_list = ["input", "outputs", "q_input", "q_outputs"]
            attr_list = [x, y_float, x_int, y_int]
            Dump.dump_file(prefix, ".iqVar.", zip(name_list, attr_list), path)
        
        return from_torch_tensor(y_float, scale_z_iq(), 8)

    @staticmethod
    def backward(ctx, s):
        x, = ctx.saved_tensors
        dim, unbiased, keepdim  = ctx.value
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = torch_var(x, dim, unbiased, keepdim)
            grad = torch.autograd.grad(y, x, s)
        return grad[0], None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, x, dim, unbiased, keepdim, scale_x, local_scale_o, running_o, scale_o, training, quant_mode, prefix, dump, path):
        param_dict = {"scale_x_f": scale_x(), "scale_o_f": scale_o()}
        input_list = [x, ]
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        param_dict["platform_quant_s"] = platform_quant
        param_dict["castor_mode_s"] = "luna"
        param_dict["dim_i"] = dim
        param_dict["unbiased_i"] = unbiased
        op = None
        op = g.op("thinker::iqVar", *input_list, **param_dict)
        return op

class iqVarLayer(torch.nn.Module):
    def __init__(self, data_bits=8, mode=QuantMode.QValue,):
        super(iqVarLayer, self).__init__()
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.data_bits = data_bits
        self.mode = mode
        self.register_buffer('scale_o', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))

    def forward(self, input, dim, unbiased, keepdim, local_scale_o, quant_mode=QuantMode.QValue):
        scale_x = ScalerBuffer(input.scale_data)
        local_scale_o = ScalerBuffer(local_scale_o)
        scale_o = ScalerBuffer(self.scale_o)
        running_o = ScalerBuffer(self.running_o)
        if isinstance(input, IQTensor):
            if input.bits != self.data_bits:
                input = Requant.apply(
                    input, input.bits, input.scale_data, self.data_bits, self.mode)
            scale_x = ScalerBuffer(input.scale_data)
        z = iqVar.apply(input, dim, unbiased, keepdim, scale_x, local_scale_o, running_o, scale_o,
                            self.training, quant_mode, self.prefix, self.dump, self.path)
        self.scale_o.fill_(scale_o.data)
        self.running_o.fill_(running_o.data)
        return z

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

def var(tensor, dim, unbiased=True, keepdim=False, *, out=None):
    is_iq_var = True
    if not isinstance(tensor,IQTensor):
        is_iq_var = False
    module_self = get_current_module()
    if module_self is None:
        is_iq_var = False   
    if not is_iq_var:
        return torch_var(tensor)

    assert tensor.bits == 8, 'iqvar only support 8bit'
    quant_mode = getattr(module_self, LINGER_MODE, QuantMode.QValue)
    iname_index = getattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_IQTENSOR_LAYER_COUNTER, iname_index+1)
    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + '_iqvar_' + str(iname_index)
    
    iq_layer = None
    if hasattr(module_self,var_name):
        iq_layer = getattr(module_self,var_name)
    else:
        iq_layer = iqVarLayer()
        iq_layer.training = module_self.training
        iq_layer = iq_layer.to(tensor.device)
        setattr(module_self,var_name,iq_layer)
    scale_z = None
    with torch.no_grad():
        z_f = torch_var(tensor, dim, unbiased, keepdim)
        max_z = torch.max(torch.abs(z_f))
        if max_z == 0:
            scale_z = 1.0
        else:
            scale_z = 127 / max_z.item()
    return iq_layer(tensor, dim, unbiased, keepdim, scale_z, quant_mode)


torch.max_pool2d = max_pool2d
torch.relu = relu
torch.max = iqmax

torch.relu_ = relu_
torch.transpose = transpose
torch.nn.functional.pad = pad

__all__ = ['torch_relu', 'torch_relu_', 'torch_max_pool2d', 'torch_cat', 'iqCatLayer', 'cat', 'torch_sigmoid', 'torch_sigmoid_',
           'iqSigmoidLayer', 'sigmoid', 'sigmoid_', 'torch_tanh', 'torch_tanh_', 'iqTanhLayer', 'tanh', 'tanh_', 'torch_clamp', 'torch_clamp_', 'iqClampLayer', 'clamp', 'clamp_',
           'dropout', 'pack_padded_sequence', 'pad_packed_sequence', 'torch_pack_padded_sequence', 'torch_pad_packed_sequence', 'bmm',
           'torch_softmax', 'softmaxIntLayer', 'softmax',
           'torch_logsoftmax', 'logsoftmaxIntLayer', 'logsoftmax', 'var', 'channel_shuffle_quant', 'channel_shuffle']
