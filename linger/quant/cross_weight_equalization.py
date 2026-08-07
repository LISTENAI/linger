import copy
import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


def cross_weight_equalization(model: nn.Module, *dummy_args) -> nn.Module:
    """
    基于 Hook 的跨层权重均衡实现

    流程:
    1. 通过 hook 追踪模型执行（融合前）
    2. 替换 ReLU6 为 ReLU
    3. 分析并融合 Conv-BN 对（用于后续偏置吸收）
    4. 找到连续的 Conv 层组（Conv -> {BN, ReLU, PReLU} -> Conv）
    5. 执行跨层权重均衡
    6. 吸收 BN 偏置到相邻层（仅当 prev_layer 后有 BN 时）
    """
    from linger.layer_tracer import (
        _record_execution, _analyze, _apply_fusion,
        _calc_gap_threshold, _is_conv, _is_bn, _connected
    )
    from linger.constrain import ConvBN2d, ConvBN1d

    if len(dummy_args) == 0:
        raise ValueError("需要提供 dummy_input 用于 hook 追踪")

    dummy_input = dummy_args[0]
    device = next(model.parameters()).device
    model.eval()

    # 1. 替换 ReLU6 为 ReLU（提前做，不影响后续层关系分析）
    replace_relu6_to_relu(model)

    # 2. 使用 hook 记录执行（融合前的原始层关系）
    records_before_fusion = _record_execution(model, dummy_input)

    if len(records_before_fusion) < 2:
        print("Warning: 记录数太少，可能模型没有正确执行")
        return model

    # 3. 分析并融合 Conv-BN（用于偏置吸收）
    gap = _calc_gap_threshold(records_before_fusion)
    pairs = _analyze(records_before_fusion, gap, model, fuse_bn=True)

    # 保存 conv_bn_dict 用于后续偏置吸收: conv -> bn 映射
    conv_bn_dict = {}
    for fb in pairs:
        conv_bn_dict[fb.conv] = fb.bn

    # 4. 基于融合前的 records 找到连续的卷积层对
    # 这里使用融合前的记录，因为融合后 ConvBN2d 不在 _HOOK_TYPES 中
    conv_pairs_from_records = _find_consecutive_conv_layers_with_middle_ops_from_records(
        records_before_fusion)

    # 5. 执行 Conv-BN 融合
    if pairs:
        _apply_fusion(model, pairs)

    # 6. 将 records 中的原始 Conv2d 模块映射到融合后的模块
    # 融合后，原始的 Conv2d 在 ConvBN2d.conv 中（但 .conv 是新创建的对象）
    # 需要找到融合后对应位置的模块
    conv_pairs = _map_records_to_fused_model(model, conv_pairs_from_records)

    # 7. 执行跨层权重均衡
    scale_factor_groups = []
    for prev_conv, curr_conv in conv_pairs:
        # 获取 prev_layer 的 bias（如果存在）
        # 对于 ConvBN2d，需要通过 .conv 访问原始 conv 层
        prev_conv_weight, prev_conv_bias = _get_conv_weight_and_bias(prev_conv)
        curr_conv_weight, curr_conv_bias = _get_conv_weight_and_bias(curr_conv)

        if prev_conv_bias is None:
            prev_bias = torch.zeros(prev_conv.out_channels, device=device)
        else:
            prev_bias = prev_conv_bias.data.clone()

        # 执行均衡
        new_weight_first, new_weight_second, new_bias, scale_factor = \
            _layer_equalization(prev_conv_weight, curr_conv_weight, prev_bias)

        # 更新权重和偏置（对于 ConvBN2d，需要更新 .conv）
        if isinstance(prev_conv, (ConvBN2d, ConvBN1d)):
            prev_conv.conv.weight = torch.nn.Parameter(new_weight_first, requires_grad=True)
            prev_conv.conv.bias = torch.nn.Parameter(new_bias, requires_grad=True)
        else:
            prev_conv.weight = torch.nn.Parameter(new_weight_first, requires_grad=True)
            prev_conv.bias = torch.nn.Parameter(new_bias, requires_grad=True)

        if isinstance(curr_conv, (ConvBN2d, ConvBN1d)):
            curr_conv.conv.weight = torch.nn.Parameter(new_weight_second, requires_grad=True)
        else:
            curr_conv.weight = torch.nn.Parameter(new_weight_second, requires_grad=True)

        scale_factor_groups.append(scale_factor)

    # 8. 吸收偏置（仅当 prev_layer 后有 BN 时才执行）
    if conv_pairs and scale_factor_groups:
        _bias_absorb(conv_pairs, conv_bn_dict, scale_factor_groups)

    # 9. 恢复模型设备状态
    model.to(device)

    return model


def _get_conv_weight_and_bias(module):
    """获取卷积层的权重和偏置，处理 ConvBN 融合层的情况"""
    if hasattr(module, 'conv'):
        # ConvBNXd fused layer - access internal conv
        return module.conv.weight.data, module.conv.bias.data if module.conv.bias is not None else None
    else:
        # Regular conv layer
        return module.weight.data, module.bias.data if module.bias is not None else None


def _map_records_to_fused_model(model, conv_pairs_from_records):
    """
    将融合前 records 中的 Conv2d 模块映射到融合后的模块

    融合后，原始的 Conv2d 被替换为 ConvBNXd，权重也被融合了
    需要找到融合后对应位置的模块（可能在 .conv 属性中）
    """
    from linger.constrain import ConvBN2d, ConvBN1d
    from linger.layer_tracer import EmptyBatchNorm

    mapped_pairs = []

    for prev_orig, curr_orig in conv_pairs_from_records:
        # 找到 prev_orig 对应的融合模块
        prev_fused = _find_fused_module_for_original(model, prev_orig)
        curr_fused = _find_fused_module_for_original(model, curr_orig)

        if prev_fused is not None and curr_fused is not None:
            mapped_pairs.append((prev_fused, curr_fused))

    return mapped_pairs


def _find_fused_module_for_original(model, original_conv):
    """
    在融合后的模型中找到与原始 Conv 对应的融合模块

    通过检查模块路径和位置来匹配
    """
    from linger.constrain import ConvBN2d, ConvBN1d

    # 构建原始 conv 的路径映射
    original_conv_path = None
    for name, module in model.named_modules():
        if module is original_conv:
            original_conv_path = name
            break

    if original_conv_path is None:
        return None

    # 找到同名位置的 ConvBN2d/ConvBN1d 模块
    for name, module in model.named_modules():
        if isinstance(module, (ConvBN2d, ConvBN1d)):
            if name == original_conv_path:
                return module

    return None


def replace_relu6_to_relu(model: nn.Module) -> nn.Module:
    """将 nn.Hardswish 和 nn.ReLU6 替换为 nn.ReLU"""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU6):
            setattr(model, name, nn.ReLU())
        else:
            replace_relu6_to_relu(module)
    return model


def _is_conv(m):
    """检查模块是否为卷积层"""
    return isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d))


def _find_consecutive_conv_layers_with_middle_ops_from_records(records: List) -> List[Tuple[nn.Conv2d, nn.Conv2d]]:
    """
    基于 hook 记录找到连续的卷积层对（中间允许有线性操作）

    支持的中间操作：ReLU, PReLU, BN, EmptyBatchNorm 等线性缩放操作
    返回: List[(prev_conv, curr_conv)]

    参考原始 FX 代码：
    cls_supported_activations = (torch.nn.ReLU, torch.nn.PReLU, torch.nn.BatchNorm2d)

    注意：与参考FX实现不同，hook-based方法在残差网络中可能因skip connections导致
    tensor ID不匹配，因此不使用 _connected 检查，而是直接查找顺序执行模式。
    """
    from linger.layer_tracer import _is_conv as _is_conv_from_tracer
    from linger.layer_tracer import _is_bn, EmptyBatchNorm

    # 支持的中间操作（线性操作，可以跨层均衡）
    _SUPPORTED_MIDDLE_OPS = (
        nn.ReLU,
        nn.PReLU, nn.LeakyReLU, nn.Sigmoid,
        nn.BatchNorm1d, nn.BatchNorm2d,
        EmptyBatchNorm
    )

    def _is_supported_middle_op(mod):
        return isinstance(mod, _SUPPORTED_MIDDLE_OPS)

    # 分析 records，找到 Conv -> [中间操作*] -> Conv 的模式
    # 注意：在残差网络中，跳过连接会导致 tensor ID 不匹配，
    # 因此这里不使用 _connected 检查，而是匹配参考 FX 实现的行为：
    # 从输出节点向后遍历时，查找连续的 Conv-[激活/BN]-Conv 模式
    pairs = []
    n = len(records)

    # 创建 module -> index 的映射，便于查找
    conv_indices = [i for i, rec in enumerate(records) if _is_conv_from_tracer(rec.mod)]

    for idx in range(len(conv_indices) - 1):
        first_idx = conv_indices[idx]
        second_idx = conv_indices[idx + 1]

        # 检查两个 conv 之间是否都是支持的中间操作
        all_supported = True
        for k in range(first_idx + 1, second_idx):
            if not _is_supported_middle_op(records[k].mod):
                all_supported = False
                break

        if all_supported:
            pairs.append((records[first_idx].mod, records[second_idx].mod))

    return pairs


def _layer_equalization(weight_first, weight_second, bias_first, s_range=(1e-8, 1e8), signed=True, eps=0):
    """跨层权重均衡核心算法"""
    with torch.no_grad():
        # Clone weights to avoid in-place modification issues
        weight_first = weight_first.clone()
        weight_second = weight_second.clone()
        if bias_first is not None:
            bias_first = bias_first.clone()

        num_group = 1
        if weight_first.shape[0] != weight_second.shape[1]:
            # group convolution
            num_group = weight_first.shape[0] // weight_second.shape[1]

        group_channels_i = weight_first.shape[0] // num_group
        group_channels_o = weight_second.shape[0] // num_group

        S = torch.zeros(weight_first.size(0))

        for g in range(num_group):
            c_start_i = g * group_channels_i
            c_end_i = (g + 1) * group_channels_i
            weight_first_group = weight_first[c_start_i:c_end_i]

            c_start_o = g * group_channels_o
            c_end_o = (g + 1) * group_channels_o
            weight_second_group = weight_second[c_start_o:c_end_o]

            for ii in range(weight_second_group.shape[1]):
                if signed:
                    range_1 = torch.max(torch.abs(weight_first_group[ii]))
                    range_2 = torch.max(torch.abs(weight_second_group[:, ii]))
                else:
                    range_1 = torch.max(weight_first_group[ii]) - torch.min(weight_first_group[ii])
                    range_2 = torch.max(weight_second_group[:, ii]) - torch.min(weight_second_group[:, ii])

                s = (1 / (range_1 + eps)) * torch.sqrt(range_1 * range_2 + eps)
                s = max(s_range[0], min(s_range[1], s))
                S[c_start_i + ii] = s

                weight_first[c_start_i + ii] = weight_first[c_start_i + ii] * s

                if bias_first is not None:
                    bias_first[c_start_i + ii] = bias_first[c_start_i + ii] * s

                weight_second[c_start_o:c_end_o, ii] = weight_second[c_start_o:c_end_o, ii] * (1/s)

        return torch.nn.Parameter(weight_first, requires_grad=True), \
               torch.nn.Parameter(weight_second, requires_grad=True), \
               torch.nn.Parameter(bias_first, requires_grad=True), S


def _bias_absorb(group_layers, conv_bn_dict, scale_factor_groups):
    """吸收 BN 偏置到相邻卷积层"""
    from linger.constrain import ConvBN2d, ConvBN1d

    def _get_absorb_bias(cur_weight, cur_bias, prev_bias, eps):
        with torch.no_grad():
            sum_weight = torch.sum(cur_weight.view(cur_weight.size(0), cur_weight.size(1), -1), -1)
            num_groups = prev_bias.size(0) // cur_weight.size(1)
            for group_idx in range(num_groups):
                start_idx = group_idx * cur_weight.size(1)
                end_idx = (group_idx + 1) * cur_weight.size(1)
                cur_bias[start_idx:end_idx] = cur_bias[start_idx:end_idx] + torch.matmul(
                    sum_weight[start_idx:end_idx], eps[start_idx:end_idx])
            prev_bias = prev_bias - eps
        return torch.nn.Parameter(cur_bias, requires_grad=True), torch.nn.Parameter(prev_bias, requires_grad=True)

    def _get_layer_bias(layer):
        """获取层的 bias，处理 ConvBN2d 的情况"""
        if isinstance(layer, (ConvBN2d, ConvBN1d)):
            return layer.conv.bias
        return layer.bias

    def _set_layer_bias(layer, bias):
        """设置层的 bias，处理 ConvBN2d 的情况"""
        if isinstance(layer, (ConvBN2d, ConvBN1d)):
            layer.conv.bias = bias
        else:
            layer.bias = bias

    def _get_layer_weight(layer):
        """获取层的 weight，处理 ConvBN2d 的情况"""
        if isinstance(layer, (ConvBN2d, ConvBN1d)):
            return layer.conv.weight
        return layer.weight

    for i, (prev_layer, curr_layer) in enumerate(group_layers):
        prev_bias = _get_layer_bias(prev_layer)
        curr_bias = _get_layer_bias(curr_layer)

        if prev_bias is None or curr_bias is None or prev_layer not in conv_bn_dict:
            continue

        if hasattr(curr_layer, "relu_before") and curr_layer.relu_before:
            eps = conv_bn_dict[prev_layer].bias - conv_bn_dict[prev_layer].weight.abs() * 3
            eps[eps < 0] = 0
            eps = eps * scale_factor_groups[i]
        else:
            eps = conv_bn_dict[prev_layer].bias * scale_factor_groups[i]

        curr_weight = _get_layer_weight(curr_layer)
        new_curr_bias, new_prev_bias = _get_absorb_bias(
            curr_weight, curr_bias.data.clone(), prev_bias.data.clone(), eps)
        _set_layer_bias(curr_layer, new_curr_bias)
        _set_layer_bias(prev_layer, new_prev_bias)


__all__ = ["cross_weight_equalization", "replace_relu6_to_relu", "_find_consecutive_conv_layers"]