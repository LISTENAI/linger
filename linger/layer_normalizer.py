#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from typing import Tuple

import torch
import torch.nn as nn

from .config import config
from .modules import *
from .ops.ops_names import LINGER_AHEAD_RELU, LINGER_AHEAD_SIGMOID
from .utils import ClampInfo, Singleton, get_device, logger


class _SingletonContainClampModules(Singleton):
    clamped_quant_list = {}
    _is_close_register = False

    def _close_register(self):
        self._is_close_register = True

    def _register(self, module, quant_info):
        assert isinstance(module, torch.nn.Module) or isinstance(module, list)
        modules = [module] if isinstance(module, torch.nn.Module) else module
        for each_mod in modules:
            if self._is_close_register:
                print("warning: module has initlized and linger.init may not work")
            self.clamped_quant_list[each_mod] = quant_info

    def _is_registered(self, module):
        return (module in self.clamped_quant_list.keys())

    def get(self, module):
        return self.clamped_quant_list.get(module)

    def clear(self):
        if self._is_close_register:
            print("warning: module has initlized and linger.clear may not work")
        self.clamped_quant_list.clear()


def disable_normalize(module: nn.Module):
    r"""
        禁用clamp策略替换网络，该接口支持inline 使用
    Notes:
        disable_normalize 应该在linger.normalize_layers函数之前调用才生效
    """
    queue = [module]
    while len(queue) > 0:
        node = queue.pop(0)
        if type(node) in SupportNormalizeTorchModules:
            _SingletonContainClampModules()._register(node, None)
        for _, submodule in node.named_children():
            queue.append(submodule)


def normalize_module(module: nn.Module, type_modules: Tuple = DefaultNormalizeIntXModule, normalize_weight_value: float = 8, normalize_bias_value: float = 8, normalize_output_value: float = None):
    r"""
        对module进行自定义clamp设置，通用的clamp策略接口，该接口支持inline使用
    """
    if not isinstance(type_modules, tuple):
        type_modules = (type_modules, )
    for t in type_modules:
        if t not in SupportNormalizeTorchModules:
            logger.fatal(str(t) + 'is not support clamp in linger now')
            exit(-1)
    queue = [module]
    while len(queue) > 0:
        node = queue.pop(0)
        if type(node) in type_modules and type(node) in SupportNormalizeTorchModules:
            cinfo = ClampInfo()
            cinfo.set_clamp_weight_value(normalize_weight_value)
            cinfo.set_clamp_bias_value(normalize_bias_value)
            cinfo.set_clamp_output_value(normalize_output_value)
            _SingletonContainClampModules()._register(node, cinfo)
        for _, submodule in node.named_children():
            queue.append(submodule)


def _replaceModule(submodule, normalize_weight_value, normalize_bias_value, normalize_output_value):
    if isinstance(submodule, tuple(SupportNormalizeTorchModules)):
        if isinstance(submodule, NormalizeConvBN2d):
            bias = True if submodule.conv.bias is not None else False
            if config.BnMomentumUpdate.disable:
                submodule_bn_momentum = 0
            else:
                submodule_bn_momentum = submodule.bn.momentum
            convbn = NormalizeConvBN2d(submodule.conv.in_channels, submodule.conv.out_channels, submodule.conv.kernel_size, submodule.conv.stride, submodule.conv.padding, submodule.conv.dilation,
                                       submodule.conv.groups, bias, submodule.conv.padding_mode, submodule.bn.eps, submodule_bn_momentum, submodule.bn.affine, submodule.bn.track_running_stats,
                                       normalize_data=normalize_output_value, normalize_weight=normalize_weight_value, normalize_bias=normalize_bias_value, ahead_relu=submodule.ahead_relu)
            return convbn
        elif isinstance(submodule, NormalizeConvBN1d):
            bias = True if submodule.conv.bias is not None else False
            if config.BnMomentumUpdate.disable:
                submodule_bn_momentum = 0
            else:
                submodule_bn_momentum = submodule.bn.momentum
            convbn = NormalizeConvBN1d(submodule.conv.in_channels, submodule.conv.out_channels, submodule.conv.kernel_size, submodule.conv.stride, submodule.conv.padding, submodule.conv.dilation,
                                       submodule.conv.groups, bias, submodule.conv.padding_mode, submodule.bn.eps, submodule_bn_momentum, submodule.bn.affine, submodule.bn.track_running_stats,
                                       normalize_data=normalize_output_value, normalize_weight=normalize_weight_value, normalize_bias=normalize_bias_value, ahead_relu=submodule.ahead_relu)
            return convbn
        elif isinstance(submodule, nn.Conv2d):
            bias = True if submodule.bias is not None else False
            ahead_relu = getattr(submodule, LINGER_AHEAD_RELU, False)
            ahead_sigmoid = getattr(
                submodule, LINGER_AHEAD_SIGMOID, False)

            conv = NormalizeConv2d(submodule.in_channels, submodule.out_channels, submodule.kernel_size, submodule.stride, submodule.padding, submodule.dilation,
                                   submodule.groups, bias, submodule.padding_mode, normalize_data=normalize_output_value, normalize_weight=normalize_weight_value, normalize_bias=normalize_bias_value, ahead_relu=ahead_relu, ahead_sigmoid=ahead_sigmoid)
            return conv
        elif isinstance(submodule, nn.Linear):
            bias = True if submodule.bias is not None else False
            ahead_relu = getattr(submodule, LINGER_AHEAD_RELU, False)
            ahead_sigmoid = getattr(
                submodule, LINGER_AHEAD_SIGMOID, False)

            linear = NormalizeLinear(submodule.in_features, submodule.out_features, bias, normalize_data=normalize_output_value, normalize_weight=normalize_weight_value,
                                     normalize_bias=normalize_bias_value, ahead_relu=ahead_relu, ahead_sigmoid=ahead_sigmoid)
            return linear
        elif isinstance(submodule, nn.Embedding):
            embed = NormalizeEmbedding(submodule.num_embeddings, submodule.embedding_dim, submodule.padding_idx, submodule.max_norm, submodule.norm_type, submodule.scale_grad_by_freq,
                                       submodule.sparse, submodule.weight, normalize_data=normalize_output_value, normalize_weight=normalize_weight_value)
            return embed
        elif isinstance(submodule, nn.ConvTranspose2d):
            bias = True if submodule.bias is not None else False
            convtranspose = NormalizeConvTranspose2d(submodule.in_channels, submodule.out_channels, submodule.kernel_size, submodule.stride, submodule.padding,
                                                     submodule.output_padding, submodule.groups, bias, submodule.dilation, submodule.padding_mode, normalize_data=normalize_output_value, normalize_weight=normalize_weight_value, normalize_bias=normalize_bias_value)
            return convtranspose
        elif isinstance(submodule, nn.BatchNorm2d):
            ahead_relu = getattr(submodule, LINGER_AHEAD_RELU, False)
            if config.BnMomentumUpdate.disable:
                submodule_momentum = 0
            else:
                submodule_momentum = submodule.momentum
            bn = NormalizeBatchNorm2d(submodule.num_features, eps=submodule.eps, momentum=submodule_momentum, affine=submodule.affine, track_running_stats=submodule.track_running_stats,
                                      normalize_data=normalize_output_value, normalize_weight=normalize_weight_value, normalize_bias=normalize_bias_value, ahead_relu=ahead_relu)
            return bn

        elif isinstance(submodule, nn.Conv1d):
            bias = True if submodule.bias is not None else False
            ahead_relu = getattr(submodule, LINGER_AHEAD_RELU, False)
            conv = NormalizeConv1d(submodule.in_channels, submodule.out_channels, submodule.kernel_size, submodule.stride, submodule.padding, submodule.dilation,
                                   submodule.groups, bias, submodule.padding_mode, normalize_data=normalize_output_value, normalize_weight=normalize_weight_value, normalize_bias=normalize_bias_value, ahead_relu=ahead_relu)
            return conv
        elif isinstance(submodule, nn.GRU):
            gru = NormalizeFastGRU(submodule.input_size, submodule.hidden_size, submodule.num_layers, submodule.bias, submodule.batch_first, submodule.dropout, submodule.bidirectional,
                                   normalize_data=normalize_output_value, normalize_weight=normalize_weight_value, normalize_bias=normalize_bias_value)
            return gru
        elif isinstance(submodule, nn.LSTM):
            lstm = NormalizeFastLSTM(submodule.input_size, submodule.hidden_size, submodule.num_layers, submodule.bias, submodule.batch_first, submodule.dropout, submodule.bidirectional,
                                     normalize_data=normalize_output_value, normalize_weight=normalize_weight_value, normalize_bias=normalize_bias_value)
            return lstm

    return None


def normalize_layers(model: nn.Module, *, normalize_modules: Tuple = DefaultNormalizeIntXModule, normalize_weight_value: float = 8, normalize_bias_value: float = None, normalize_output_value: float = 8) -> nn.Module:
    r"""对模型进行clamp 处理以方便进行更好的量化，
        默认支持nn.Conv2d, nn.Linear, nn.ConvTranspose2d, Conv-BN融合Clamp

    """
    if type(normalize_modules) is not tuple:
        normalize_modules = (normalize_modules, )
    normalize_modules = set(list(normalize_modules) +
                            [NormalizeConvBN2d, NormalizeConvBN1d])
    for user_module_type in normalize_modules:
        assert user_module_type in SupportNormalizeTorchModules, 'Currently not support clamp of ' + \
            str(user_module_type)
    device = get_device(model)
    queue = [model]
    while len(queue) > 0:
        node = queue.pop(0)
        for name, submodule in node.named_children():
            if _SingletonContainClampModules()._is_registered(submodule):
                clamp_info = _SingletonContainClampModules().get(submodule)
                if clamp_info == None:
                    continue
                else:
                    r_module = _replaceModule(submodule, clamp_info.clamp_weight_value, clamp_info.clamp_bias_value,
                                              clamp_info.clamp_output_value)
                    assert r_module is not None
                    setattr(node, name, r_module)
            elif type(submodule) in SupportNormalizeTorchModules and type(submodule) in normalize_modules:
                r_module = _replaceModule(
                    submodule, normalize_weight_value, normalize_bias_value, normalize_output_value)
                assert r_module is not None
                setattr(node, name, r_module)
            else:
                queue.append(submodule)

    if model.training:
        model.train()
    else:
        model.eval()
    model.to(device)
    return model
