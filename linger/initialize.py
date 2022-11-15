import itertools
from typing import Tuple

import torch
import torch.nn as nn

import linger

from .config import config
from .modules import *
from .onnx import export as linger_export
from .ops import *
from .ops.iqtensor import iqAddLayer, iqDivLayer, iqMulLayer, iqSumLayer
from .ops.linger_functional import bmm as linger_bmm
from .ops.linger_functional import cat as linger_cat
from .ops.linger_functional import channel_shuffle_quant
from .ops.linger_functional import clamp as linger_clamp
from .ops.linger_functional import clamp_ as linger_clamp_
from .ops.linger_functional import dropout as linger_dropout
from .ops.linger_functional import (iqCatLayer, iqClampLayer, iqSigmoidLayer,
                                    iqTanhLayer, iqVarLayer)
from .ops.linger_functional import logsoftmax as linger_logsoftmax
from .ops.linger_functional import \
    pack_padded_sequence as linger_pack_padded_sequence
from .ops.linger_functional import \
    pad_packed_sequence as linger_pad_packed_sequence
from .ops.linger_functional import sigmoid as linger_sigmoid
from .ops.linger_functional import sigmoid_ as linger_sigmoid_
from .ops.linger_functional import softmax as linger_softmax
from .ops.linger_functional import tanh as linger_tanh
from .ops.linger_functional import tanh_ as linger_tanh_
from .ops.linger_functional import (torch_pack_padded_sequence,
                                    torch_pad_packed_sequence)
from .ops.linger_functional import var as linger_var
from .ops.module_self import hook_forward, hook_pre_forward
from .ops.ops_names import (LINGER_AHEAD_RELU, LINGER_AHEAD_SIGMOID,
                            LINGER_MIX_INT8_MANUAL_ROUND_LAYERS, LINGER_MODE,
                            LINGER_OBIT)
from .utils import QuantInfo, QuantMode, Singleton, get_device, logger

__all__ = ["disable_quant", "quant_module",
           "quant_module_by_type", "quant_tensor", "init"]


class _SingletonContainCustomModules(Singleton):
    customized_quant_list = {}
    _is_close_register = False

    def _close_register(self):
        self._is_close_register = True

    def _register(self, module, quant_info):
        assert isinstance(module, torch.nn.Module) or isinstance(module, list)
        modules = [module] if isinstance(module, torch.nn.Module) else module
        for each_mod in modules:
            if self._is_close_register:
                print("warning: module has initlized and linger.init may not work")
            self.customized_quant_list[each_mod] = quant_info

    def _is_registered(self, module):
        return (module in self.customized_quant_list.keys())

    def get(self, module):
        return self.customized_quant_list.get(module)

    def clear(self):
        if self._is_close_register:
            print("warning: module has initlized and linger.clear may not work")
        self.customized_quant_list.clear()


def fuse_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

    eps = 1e-5
    clamp_conv_name = prefix + 'conv'
    clamp_bn_name = prefix + 'bn'
    conv_int_name = prefix
    if clamp_conv_name + '.weight' in state_dict and clamp_bn_name + '.weight' in state_dict:
        b_mean = state_dict[clamp_bn_name + '.running_mean']
        b_var = state_dict[clamp_bn_name + '.running_var']
        b_w = state_dict[clamp_bn_name + '.weight']
        b_b = state_dict[clamp_bn_name + '.bias']
        sigma = 1 / torch.sqrt(b_var + eps)
        alpha = b_w * sigma
        beta = b_b - b_mean * alpha
        c_w = state_dict[clamp_conv_name + '.weight']
        state_dict[conv_int_name +
                   'weight'] = (c_w * alpha.view(-1, *([1]*(len(c_w.shape)-1))))
        if clamp_conv_name + '.bias' in state_dict:
            c_b = state_dict[clamp_conv_name + '.bias']
            state_dict[conv_int_name + 'bias'] = (c_b * alpha + beta)
            state_dict.pop(clamp_conv_name + '.bias')
        else:
            state_dict[conv_int_name + 'bias'] = beta
        state_dict.pop(clamp_bn_name + '.running_mean')
        state_dict.pop(clamp_bn_name + '.running_var')
        state_dict.pop(clamp_bn_name + '.weight')
        state_dict.pop(clamp_bn_name + '.bias')
        state_dict.pop(clamp_bn_name + '.num_batches_tracked')
        state_dict.pop(clamp_conv_name + '.weight')
    else:
        assert clamp_conv_name + '.weight' not in state_dict and clamp_bn_name + \
            '.weight' not in state_dict, 'load quanted model but contain float clamp params'


def _replaceOp(submodule, mode, in_data_bits, parameter_bits, out_bits=None):
    assert in_data_bits > 0 and in_data_bits <= 32, "in_data_bits should between 0 and 32"
    assert parameter_bits > 0 and parameter_bits <= 32, "parameter_bits should between 0 and 32"
    assert out_bits is None or out_bits > 0 and out_bits <= 32, "out_bits should between 0 and 32"
    assert mode == QuantMode.QValue, "mode support only q_value"
    if isinstance(submodule, tuple(SupportQuantTorchModules)):
        if isinstance(submodule, NormalizeFastGRU):
            gru = GRUInt(submodule.input_size, submodule.hidden_size, submodule.num_layers, submodule.bias, submodule.batch_first, submodule.dropout, submodule.bidirectional,
                         data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits,
                         clamp_data=submodule.normalize_data, clamp_weight=submodule.normalize_weight, clamp_bias=submodule.normalize_bias)
            return gru
        elif isinstance(submodule, NormalizeFastLSTM):
            lstm = LSTMInt(submodule.input_size, submodule.hidden_size, submodule.num_layers, submodule.bias, submodule.batch_first, submodule.dropout, submodule.bidirectional,
                           data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits,
                           clamp_data=submodule.normalize_data, clamp_weight=submodule.normalize_weight, clamp_bias=submodule.normalize_bias)
            return lstm
        elif isinstance(submodule, NormalizeConvBN1d):
            # ahead_relu = getattr(submodule,LINGER_AHEAD_RELU,False)
            conv = Conv1dInt(submodule.conv.in_channels, submodule.conv.out_channels, submodule.conv.kernel_size, submodule.conv.stride, submodule.conv.padding, submodule.conv.dilation, submodule.conv.groups,
                             True, submodule.conv.padding, data_bits=in_data_bits, parameter_bits=parameter_bits, o_bits=out_bits, mode=mode,
                             clamp_data=submodule.normalize_data, clamp_weight=submodule.normalize_weight, clamp_bias=submodule.normalize_bias, ahead_relu=submodule.ahead_relu)
            conv._register_load_state_dict_pre_hook(fuse_state_dict)
            return conv
        elif isinstance(submodule, NormalizeConvBN2d):
            # ahead_relu = getattr(submodule,LINGER_AHEAD_RELU,False)
            conv = Conv2dInt(submodule.conv.in_channels, submodule.conv.out_channels, submodule.conv.kernel_size, submodule.conv.stride, submodule.conv.padding, submodule.conv.dilation, submodule.conv.groups,
                             True, submodule.conv.padding, data_bits=in_data_bits, parameter_bits=parameter_bits, o_bits=out_bits, mode=mode,
                             clamp_data=submodule.normalize_data, clamp_weight=submodule.normalize_weight, clamp_bias=submodule.normalize_bias, ahead_relu=submodule.ahead_relu)
            conv._register_load_state_dict_pre_hook(fuse_state_dict)
            return conv
        elif isinstance(submodule, NormalizeConv1d):
            bias = True if submodule.bias is not None else False
            # ahead_relu = getattr(submodule,LINGER_AHEAD_RELU,False)
            conv = Conv1dInt(submodule.in_channels, submodule.out_channels, submodule.kernel_size, submodule.stride, submodule.padding, submodule.dilation, submodule.groups,
                             bias, submodule.padding_mode, data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits,
                             clamp_data=submodule.normalize_data, clamp_weight=submodule.normalize_weight, clamp_bias=submodule.normalize_bias, ahead_relu=submodule.ahead_relu)
            return conv
        elif isinstance(submodule, NormalizeConv2d):
            bias = True if submodule.bias is not None else False
            # ahead_relu = getattr(submodule,LINGER_AHEAD_RELU,False)
            conv = Conv2dInt(submodule.in_channels, submodule.out_channels, submodule.kernel_size, submodule.stride, submodule.padding, submodule.dilation, submodule.groups,
                             bias, submodule.padding_mode, data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits,
                             clamp_data=submodule.normalize_data, clamp_weight=submodule.normalize_weight, clamp_bias=submodule.normalize_bias,
                             ahead_relu=submodule.ahead_relu, ahead_sigmoid=submodule.ahead_sigmoid)
            return conv
        elif isinstance(submodule, NormalizeConvTranspose2d):
            bias = True if submodule.bias is not None else False
            conv_transpose = ConvTranspose2dInt(submodule.in_channels, submodule.out_channels, submodule.kernel_size, submodule.stride, submodule.padding, submodule.output_padding, submodule.groups,
                                                bias, submodule.dilation, submodule.padding_mode, data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits,
                                                clamp_data=submodule.normalize_data, clamp_weight=submodule.normalize_weight, clamp_bias=submodule.normalize_bias)
            return conv_transpose
        elif isinstance(submodule, NormalizeLinear):
            bias = True if submodule.bias is not None else False
            linear = LinearInt(submodule.in_features, submodule.out_features, bias, data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits,
                               clamp_data=submodule.normalize_data, clamp_weight=submodule.normalize_weight, clamp_bias=submodule.normalize_bias,
                               ahead_relu=submodule.ahead_relu, ahead_sigmoid=submodule.ahead_sigmoid)
            return linear
        elif isinstance(submodule, NormalizeBatchNorm2d):
            # ahead_relu = getattr(submodule,LINGER_AHEAD_RELU,False)
            if config.BnMomentumUpdate.disable:
                submodule_momentum = 0
            else:
                submodule_momentum = submodule.momentum
            bn = BatchNormInt(submodule.num_features, eps=submodule.eps, momentum=submodule_momentum, affine=submodule.affine, track_running_stats=submodule.track_running_stats,
                              data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits,
                              clamp_data=submodule.normalize_data, clamp_weight=submodule.normalize_weight, clamp_bias=submodule.normalize_bias, ahead_relu=submodule.ahead_relu)
            return bn
        elif isinstance(submodule, NormalizeLayerNorm):
            ahead_relu = getattr(submodule, LINGER_AHEAD_RELU, False)
            submodule_momentum = 0.1
            layer_norm = LayerNormInt(submodule.normalized_shape, submodule.eps, submodule_momentum, submodule.elementwise_affine,
                                      data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits, ahead_relu=ahead_relu)
            return layer_norm
        elif isinstance(submodule, NormalizeEmbedding):
            embedding = EmbeddingInt(submodule.num_embeddings, submodule.embedding_dim, submodule.padding_idx, submodule.max_norm, submodule.norm_type, submodule.scale_grad_by_freq, submodule.sparse,
                                     submodule.weight, data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits,
                                     clamp_data=submodule.normalize_data, clamp_weight=submodule.normalize_weight)
            return embedding
        elif isinstance(submodule, nn.Linear):
            bias = True if submodule.bias is not None else False
            ahead_relu = getattr(submodule, LINGER_AHEAD_RELU, False)
            ahead_sigmoid = getattr(
                submodule, LINGER_AHEAD_SIGMOID, False)
            linear = LinearInt(submodule.in_features, submodule.out_features, bias, data_bits=in_data_bits,
                               parameter_bits=parameter_bits, mode=mode, o_bits=out_bits, ahead_relu=ahead_relu, ahead_sigmoid=ahead_sigmoid)
            return linear
        elif isinstance(submodule, nn.Conv2d):
            bias = True if submodule.bias is not None else False
            ahead_relu = getattr(submodule, LINGER_AHEAD_RELU, False)
            ahead_sigmoid = getattr(
                submodule, LINGER_AHEAD_SIGMOID, False)
            conv = Conv2dInt(submodule.in_channels, submodule.out_channels, submodule.kernel_size, submodule.stride, submodule.padding, submodule.dilation, submodule.groups,
                             bias, submodule.padding_mode, data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits,
                             ahead_relu=ahead_relu, ahead_sigmoid=ahead_sigmoid)
            return conv
        elif isinstance(submodule, nn.BatchNorm2d):
            ahead_relu = getattr(submodule, LINGER_AHEAD_RELU, False)
            if config.BnMomentumUpdate.disable:
                submodule_momentum = 0
            else:
                submodule_momentum = submodule.momentum
            batch_norm = BatchNormInt(submodule.num_features, submodule.eps, submodule_momentum, submodule.affine, submodule.track_running_stats,
                                      data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits, ahead_relu=ahead_relu)
            return batch_norm

        elif isinstance(submodule, nn.LayerNorm):
            ahead_relu = getattr(submodule, LINGER_AHEAD_RELU, False)
            submodule_momentum = 0.1
            layer_norm = LayerNormInt(submodule.normalized_shape, submodule.eps, submodule_momentum, submodule.elementwise_affine,
                                      data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits, ahead_relu=ahead_relu)
            return layer_norm
        elif isinstance(submodule, nn.ConvTranspose2d):
            bias = True if submodule.bias is not None else False
            conv_transpose = ConvTranspose2dInt(submodule.in_channels, submodule.out_channels, submodule.kernel_size, submodule.stride, submodule.padding, submodule.output_padding, submodule.groups,
                                                bias, submodule.dilation, submodule.padding_mode, data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits)
            return conv_transpose
        elif isinstance(submodule, nn.GRU):
            gru = GRUInt(submodule.input_size, submodule.hidden_size, submodule.num_layers, submodule.bias, submodule.batch_first,
                         submodule.dropout, submodule.bidirectional, data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits)
            return gru
        elif isinstance(submodule, nn.LSTM):
            lstm = LSTMInt(submodule.input_size, submodule.hidden_size, submodule.num_layers, submodule.bias, submodule.batch_first,
                           submodule.dropout, submodule.bidirectional, data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits)
            return lstm
        elif isinstance(submodule, nn.AvgPool2d):
            avg_pool = AvgPool2dInt(submodule.kernel_size, submodule.stride, submodule.padding, submodule.ceil_mode,
                                    submodule.count_include_pad, submodule.divisor_override, data_bits=in_data_bits, mode=mode, o_bits=out_bits)
            return avg_pool
        elif isinstance(submodule, nn.Conv1d):
            bias = True if submodule.bias is not None else False
            ahead_relu = getattr(submodule, LINGER_AHEAD_RELU, False)
            conv1d = Conv1dInt(submodule.in_channels, submodule.out_channels, submodule.kernel_size, submodule.stride, submodule.padding, submodule.dilation, submodule.groups,
                               bias, submodule.padding_mode, data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits, ahead_relu=ahead_relu)
            return conv1d
        elif isinstance(submodule, nn.ReLU6):
            relu6 = ReLU6Int(data_bits=in_data_bits, mode=mode)
            return relu6
        elif isinstance(submodule, nn.Embedding):
            embedding = EmbeddingInt(submodule.num_embeddings, submodule.embedding_dim, submodule.padding_idx, submodule.max_norm, submodule.norm_type, submodule.scale_grad_by_freq, submodule.sparse,
                                     submodule.weight, data_bits=in_data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=out_bits)
            return embedding

    return None


def disable_quant(module: nn.Module):
    r"""禁用module及子module 任何量化策略(使用原有浮点精度),该接口支持inline使用(Example2)

    Args:
        module:需要不进行量化的module

    Example:        
        >>> net = shufflenet_v2_x1_0(pretrained=False)
        >>> linger.disable_quant(net)
    Example:
        >>> class Net(nn.Module):
        >>>     def __init__(self):
        >>>         super(Net, self).__init__()
        >>>         self.conv1 = nn.Sequential(
        >>>             nn.Conv2d(
        >>>                 in_channels=1,
        >>>                 out_channels=16,
        >>>                 kernel_size=5,
        >>>                 stride=1,
        >>>                 padding=2,
        >>>             ),
        >>>             nn.ReLU(),
        >>>             nn.MaxPool2d(kernel_size=2),
        >>>             nn.Conv2d(in_channels=16,
        >>>                 out_channels=16,
        >>>                 kernel_size=5,
        >>>                 stride=1,
        >>>                 padding=2,)
        >>>         )
        >>>         self.conv2 = nn.Sequential(
        >>>             nn.Conv2d(16, 32, 5, 1, 2),
        >>>             nn.ReLU(),
        >>>             nn.MaxPool2d(2),
        >>>         )
        >>>         self.out = nn.Linear(32 * 7 * 7, 10)
        >>>         linger.disable_quant(self.out)
    Notes:
        disable_quant 应该在linger.init函数之前调用才生效

        """
    queue = [module]
    while len(queue) > 0:
        node = queue.pop(0)
        if type(node) in SupportQuantTorchModules:
            _SingletonContainCustomModules()._register(node, None)
        for _, submodule in node.named_children():
            queue.append(submodule)


def quant_module(module: nn.Module, type_modules: Tuple = DefaultQuantIntXOP, mode: QuantMode = QuantMode.QValue, data_bits: int = 8, parameter_bits: int = 8, out_bits: int = None):
    r"""对module进行自定义量化设置,这是一个通用的量化策略设置接口,该接口支持inline使用

        Args:
            module: 需要量化的模型或者模块
            type_modules(tuple):量化针对的类型,默认为(nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.GRU, nn.LSTM)
            mode(QuantMode):量化模式,默认为Q值量化
            data_bits(int):推理过程中data(激活值)量化精度bit数，默认为8bit
            parameter_bits(int):权重的量化精度bit数，默认为8bit
            out_bits(int or None):输出数值的精度，默认为None，表示float精度 
        .. math:: 
            \text {save_weight_bits}=\begin{cases}
                8 & parameter\_bits\leq 8 \\
                16 & 8 < parameter\_bits\leq 16 \\
                32 & parameter\_bits \geq 16
            \end{cases}
        Notes:
            quant_module 应该在linger.init函数之前调用才生效
            如果module weights中有bias, 如果onnx 导出, 存储使用32bit
        """
    if not isinstance(type_modules, tuple):
        type_modules = (type_modules,)
    for t in type_modules:
        if t not in SupportQuantTorchModules:      # type_modules is a subset of SupportQuantTorchModules
            logger.fatal(str(t)+" is not supprt quant in linger now")
            exit(-1)

    queue = [module]
    while len(queue) > 0:
        node = queue.pop(0)
        if type(node) in type_modules and type(node) in SupportQuantTorchModules:      # and 后 判断多余
            qinfo = QuantInfo()
            qinfo.set_data_bits(data_bits)
            qinfo.set_parameter_bits(parameter_bits)
            qinfo.set_output_bits(out_bits)
            qinfo.set_mode(mode)
            _SingletonContainCustomModules()._register(node, qinfo)
        for _, submodule in node.named_children():
            queue.append(submodule)


def quant_module_by_type(module: nn.Module, type_modules: Tuple = DefaultQuantIntXOP, mode: QuantMode = QuantMode.QValue, data_bits: int = 8, parameter_bits: int = 8, out_bits: int = None):
    r"""对module进行自定义量化设置,包括激活值，weight以及输出激活值都是16bit,这是一个通用的量化策略设置接口,该接口支持inline使用

        Args:
            module: 需要量化的模型或者模块
            type_modules(tuple):量化针对的类型,默认为(nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.GRU, nn.LSTM)
            mode(QuantMode):量化模式,默认为Q值量化
            data_bits(int):推理过程中data(激活值)量化精度bit数，默认为8bit
            parameter_bits(int):权重的量化精度bit数，默认为8bit
            out_bits(int or None):输出数值的精度，默认为None，表示float精度 

        Notes:
            quant_module_by_type 应该在linger.init函数之前调用才生效
            如果module weights中有bias, 如果onnx 导出, 存储使用32bit
        """
    if type(type_modules) is not tuple:
        type_modules = (type_modules,)
    for user_module_type in type_modules:
        assert user_module_type in SupportQuantTorchModules, 'currently not support quant of ' + \
            str(user_module_type)
    queue = [module]
    while len(queue) > 0:
        node = queue.pop(0)
        for t in type_modules:
            if type(node) == t:
                qinfo = QuantInfo()
                qinfo.set_data_bits(data_bits)
                qinfo.set_parameter_bits(parameter_bits)
                qinfo.set_output_bits(out_bits)
                qinfo.set_mode(mode)
                _SingletonContainCustomModules()._register(node, qinfo)
        for _, submodule in node.named_children():
            queue.append(submodule)


def quant_tensor(module: nn.Module, x: torch.Tensor, name: str = '_default_layername', mode: QuantMode = QuantMode.QValue, bits: int = 8, zero_point: int = 0) -> torch.Tensor:
    r"""对tensor进行量化

    Args:
        module(torch.nn.Module):tensor 量化所在的module，如果是在forward代码里面，一般是self
        x(torch.Tensor):量化tensor的tensor
        name(str):量化后module名字，如果同一forward中出现多个需要量化的tensor，该名字应设置成不一样，默认为'_default_layername'
        mode(QuantMode):量化模式，默认QValue
        bits(int):tensor量化的bit数，默认为8bit
    Returns:
        返回量化后的x(tensor)
    Example:
        >>> def forward(self, x):        
        >>>     x = self.conv1(x)
        >>>     x = linger.quant_tensor(self,x,'i_am_the_key_code_line')
        >>>     x = self.conv2(x)
        >>>     x = x.view(x.size(0), -1)           
        >>>     output = self.out(x)
        >>> return output

    """
    var_name = LINGER_MIX_INT8_MANUAL_ROUND_LAYERS + '_iq_tensor_quant_' + name
    if hasattr(module, var_name):
        round_layer = getattr(module, var_name)
    else:
        round_layer = ScaledRoundLayer(
            mode=mode, bits=bits, zero_point=zero_point)
        round_layer.training = module.training
        round_layer = round_layer.to(x.device)
        setattr(module, var_name, round_layer)
    x = round_layer(x)
    return x


def hook_pre_pack_forward(module, input):
    input = list(input)
    if isinstance(input[0], tuple):
        orig_input = input[0]
        input_ori, lengths, batch_first, enforce_sorted = orig_input
        packed_input = torch_pack_padded_sequence(
            input_ori, lengths, batch_first, enforce_sorted)
        input[0] = packed_input
    return tuple(input)


def hook_pre_pad_forward(module, input, output):
    if isinstance(input[0], tuple):
        output_packed = output[0]
        output_unpack, lengths = torch_pad_packed_sequence(
            output_packed, module.batch_first)
        return (output_unpack, lengths), output[1]
    else:
        return output


def init(model: nn.Module, *, quant_modules: Tuple = DefaultQuantIntXOP, parameter_bits: int = 8, mode: QuantMode = QuantMode.QValue) -> nn.Module:
    data_bits = 8
    out_bits = 8
    assert data_bits > 0 and data_bits <= 32, "data_bits should between 0 and 32"
    assert parameter_bits > 0 and parameter_bits <= 32, "parameter_bits should between 0 and 32"
    assert data_bits + parameter_bits <= 32, "data_bits + parameter_bits less than 32"
    assert out_bits is None or out_bits > 0 and out_bits <= 32, "out_bits should between 0 and 32"
    if type(quant_modules) is not tuple:
        quant_modules = (quant_modules,)
    quant_modules = set(list(quant_modules) + [NormalizeConvBN1d, NormalizeConvBN2d, NormalizeConv2d, NormalizeConv1d, NormalizeConvTranspose2d,
                        NormalizeLinear, nn.ReLU6, NormalizeFastGRU, NormalizeFastLSTM, NormalizeBatchNorm2d, NormalizeEmbedding, NormalizeLayerNorm])
    for user_module_type in quant_modules:
        assert user_module_type in SupportQuantTorchModules, 'currently not support quant of ' + \
            str(user_module_type)

    if config.IQTensor.iqcat:
        torch.cat = linger_cat
    if config.IQTensor.iqclamp:
        torch.clamp = linger_clamp
        torch.clamp_ = linger_clamp_
    if config.IQTensor.iqsigmoid:
        torch.sigmoid = linger_sigmoid
        torch.sigmoid_ = linger_sigmoid_
    if config.IQTensor.iqtanh:
        torch.tanh = linger_tanh
        torch.tanh_ = linger_tanh_
    if config.IQTensor.softmaxint:
        torch.softmax = linger_softmax
        torch.nn.functional.softmax = linger_softmax
    if config.IQTensor.logsoftmaxint:
        torch.log_softmax = linger_logsoftmax
        torch.nn.functional.log_softmax = linger_logsoftmax
    if config.IQTensor.iqvar:
        torch.var = linger_var
    if config.FunctionQuant.bmm:
        torch.bmm = linger_bmm
    if config.FunctionQuant.channel_shuffle:
        linger.channel_shuffle = channel_shuffle_quant
    torch.nn.functional.dropout = linger_dropout
    torch.onnx.export = linger_export
    if nn.LSTM in quant_modules or nn.GRU in quant_modules:
        torch.nn.utils.rnn.pack_padded_sequence = linger_pack_padded_sequence
        torch.nn.utils.rnn.pad_packed_sequence = linger_pad_packed_sequence
    device = get_device(model)
    queue = [model]
    while len(queue) > 0:
        node = queue.pop(0)
        # add for get father module in current forward
        node.register_forward_pre_hook(hook_pre_forward)
        node.register_forward_hook(hook_forward)
        setattr(node, LINGER_MODE, mode)
        setattr(node, LINGER_OBIT, out_bits)
        for name, submodule in node.named_children():
            # customed by user, set Singleton Class for get module register_keys by class instance
            if _SingletonContainCustomModules()._is_registered(submodule):
                quant_info = _SingletonContainCustomModules().get(submodule)
                if quant_info == None:  # disabled module
                    if isinstance(submodule, (nn.LSTM, nn.GRU, NormalizeFastGRU, NormalizeFastLSTM)):
                        submodule.register_forward_pre_hook(
                            hook_pre_pack_forward)
                        submodule.register_forward_hook(hook_pre_pad_forward)
                    continue
                else:
                    r_module = _replaceOp(submodule, quant_info.mode, quant_info.data_bits,
                                          quant_info.parameter_bits, quant_info.output_bits)
                    assert r_module is not None
                    setattr(node, name, r_module)
            elif type(submodule) in SupportQuantTorchModules and type(submodule) in quant_modules:  # not customed
                r_module = _replaceOp(
                    submodule, mode, data_bits, parameter_bits, out_bits)
                assert r_module is not None
                setattr(node, name, r_module)
            else:
                queue.append(submodule)

    def quant_tensor_pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def quant_tensor_layer(module, prefix=''):
            local_name_params = itertools.chain(
                module._parameters.items(), module._buffers.items())
            local_state = {k: v for k, v in local_name_params if v is not None}
            for key in state_dict.keys():
                if LINGER_MIX_INT8_MANUAL_ROUND_LAYERS in key:
                    if key.startswith(prefix):
                        full_input_name = key[len(prefix):]
                        # get the name of param/buffer/child
                        input_name = full_input_name.split('.', 1)[0]
                        if input_name not in module._modules and input_name not in local_state:
                            if '_iqadd_' in input_name:
                                iq_layer = iqAddLayer()
                                iq_layer.training = model.training
                                iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_iqmul_' in input_name:
                                iq_layer = iqMulLayer()
                                iq_layer.training = model.training
                                iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_iqcat_' in input_name:
                                iq_layer = iqCatLayer()
                                iq_layer.training = model.training
                                iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_iqsigmoid_' in input_name:
                                iq_layer = iqSigmoidLayer()
                                iq_layer.training = model.training
                                iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_iqtanh_' in input_name:
                                iq_layer = iqTanhLayer()
                                iq_layer.training = model.training
                                iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_iqclamp_' in input_name:
                                iq_layer = iqClampLayer()
                                iq_layer.training = model.training
                                iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_iqdiv_' in input_name:
                                iq_layer = iqDivLayer()
                                iq_layer.training = model.training
                                iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_iqsum_' in input_name:
                                iq_layer = iqSumLayer()
                                iq_layer.training = model.training
                                iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_iqvar_' in input_name:
                                iq_layer = iqVarLayer()
                                iq_layer.training = model.training
                                iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_iq_tensor_quant_' in input_name:
                                round_layer = ScaledRoundLayer(
                                    mode=mode, bits=data_bits)
                                round_layer.training = model.training
                                round_layer = round_layer.to(device)
                                setattr(module, input_name, round_layer)
                            elif '_function_bmm_' in input_name:
                                bmm_layer = BmmInt(data_bits=8, mode=mode,)
                                bmm_layer.training = model.training
                                bmm_layer = bmm_layer.to(device)
                                setattr(module, input_name, bmm_layer)
                            elif '_SoftmaxInt_' in input_name:
                                softmax_layer = softmaxIntLayer(
                                    data_bits=8, mode=mode,)
                                softmax_layer.training = model.training
                                softmax_layer = softmax_layer.to(device)
                                setattr(module, input_name, softmax_layer)

            for name, children in module._modules.items():
                if children is not None:
                    quant_tensor_layer(children, prefix + name + '.')

        quant_tensor_layer(model, prefix)
        quant_tensor_layer = None

    model._register_load_state_dict_pre_hook(quant_tensor_pre_hook)
    if model.training:
        model.train()
    else:
        model.eval()
    model.to(device)
    return model
