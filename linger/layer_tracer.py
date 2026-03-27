"""基于 Hook 的算子融合 tracer，替代 JIT-based trace_layers。

支持融合模式：convbn / convbnrelu / convrelu / bnrelu / linearrelu / linear_sigmoid / conv_sigmoid
兼容 transformer 等含动态控制流的网络。

权重加载：
  方案B - 自动：load_state_dict 时 pre_hook 自动重映射 key
  方案C - 显式：返回 CheckpointAdapter，可手动转换并序列化
"""

import torch
import torch.nn as nn
from typing import List

from .utils import ActivationType, LINGER_ACTIVATION_TYPE, LINGER_IGNORE_PAMAMTER
from .constrain import ConvBN1d, ConvBN2d, ConvTransposeBN1d, ConvTransposeBN2d
from .config import QUANT_CONFIGS

_LINGER_TRACED = '_linger_traced'
_LINGER_TRACE_HOOK_HANDLE = '_linger_trace_hook_handle'

# Conv 类型 -> 合法的 BN 类型
_CONV_BN_MATCH = {
    nn.Conv1d: nn.BatchNorm1d, nn.Conv2d: nn.BatchNorm2d,
    nn.ConvTranspose1d: nn.BatchNorm1d, nn.ConvTranspose2d: nn.BatchNorm2d,
}
# Conv 类型 -> 融合后的 ConvBN 模块类
_CONV_TO_FUSED = {
    nn.Conv1d: ConvBN1d, nn.Conv2d: ConvBN2d,
    nn.ConvTranspose1d: ConvTransposeBN1d, nn.ConvTranspose2d: ConvTransposeBN2d,
}
_BN_PARAMS = ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked')
_CONV_PARAMS = ('weight', 'bias')


# ==================== 与 layer_tracer.py 一致的基础类 ====================

class FuseableConvBN():
    """记录一对可融合的 Conv-BN 信息"""
    def __init__(self, conv_f, conv, bn_f, bn, root_model=None):
        self.conv_f = conv_f        # conv 的父模块
        self.conv = conv
        self.bn_f = bn_f            # bn 的父模块
        self.bn = bn
        self.scope_conv = None      # conv 在模型树中的路径
        self.scope_bn = None        # bn 在模型树中的路径
        self.root_model = root_model

    def set_root_model(self, root_model):
        self.root_model = root_model


class EmptyBatchNorm(torch.nn.Module):
    """融合后 BN 的占位符，不做任何 Tensor 操作"""
    def __init__(self):
        super(EmptyBatchNorm, self).__init__()
        setattr(self, LINGER_IGNORE_PAMAMTER,
                torch.nn.Parameter(torch.zeros([1])))

    def forward(self, input):
        return input

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        pass


# ==================== 执行记录器：hook + 全局操作计数 ====================

class _OpCounterMode(torch.overrides.TorchFunctionMode):
    """拦截所有 torch 操作递增计数器，用于检测 module 间的隐式操作"""
    def __init__(self, counter_ref):
        self._ref = counter_ref

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self._ref[0] += 1
        return func(*(args or ()), **(kwargs or {}))


class _Rec:
    """单次模块执行的快照"""
    __slots__ = ('name', 'mod', 'out_id', 'out_ver', 'inp_id', 'inp_ver', 'seq')
    def __init__(self, name, mod, inp_id, inp_ver, out_id, out_ver, seq):
        self.name, self.mod = name, mod
        self.inp_id, self.inp_ver = inp_id, inp_ver
        self.out_id, self.out_ver = out_id, out_ver
        self.seq = seq


# 需要挂 hook 的模块类型
_HOOK_TYPES = (
    nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d,
    nn.BatchNorm1d, nn.BatchNorm2d,
    nn.ReLU, nn.Sigmoid, nn.Linear,
)


def _record_execution(model, *args):
    """执行一次 forward，返回所有目标模块的执行记录"""
    records = []
    counter = [0]  # 用列表以便闭包可修改
    hooks = []

    def _make_hook(name):
        def hook(mod, inp, out):
            it = inp[0] if isinstance(inp, tuple) else inp
            ot = out[0] if isinstance(out, tuple) else out
            if isinstance(it, torch.Tensor) and isinstance(ot, torch.Tensor):
                records.append(_Rec(name, mod, id(it), it._version,
                                    id(ot), ot._version, counter[0]))
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, _HOOK_TYPES):
            hooks.append(mod.register_forward_hook(_make_hook(name)))

    mode = _OpCounterMode(counter)
    try:
        mode.__enter__()
        with torch.no_grad():
            # training = model.training
            # model.train()
            model(*args)
            # model.train(training)
    finally:
        mode.__exit__(None, None, None)
        for h in hooks:
            h.remove()
    return records


def _calc_gap_threshold(records):
    """自动标定相邻 Conv-BN 之间正常 op 间隔，返回阈值"""
    max_gap = 0
    for i in range(len(records) - 1):
        a, b = records[i], records[i + 1]
        if _is_conv(a.mod) and _is_bn(b.mod):
            if a.out_id == b.inp_id and a.out_ver == b.inp_ver:
                max_gap = max(max_gap, b.seq - a.seq)
    return int(max_gap * 1.5) + 1 if max_gap > 0 else 100


# ==================== 邻接分析 ====================

def _is_conv(m):
    return isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d))

def _is_bn(m):
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))

def _connected(a, b, gap):
    """三重校验：tensor id + _version + op 间隔"""
    return a.out_id == b.inp_id and a.out_ver == b.inp_ver and (b.seq - a.seq) <= gap

def _get_parent(model, dotted_name):
    """'a.b.c' -> (model.a.b, 'c')"""
    parts = dotted_name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _analyze(records, gap, model, fuse_bn):
    """分析执行记录，找出需要融合的 Conv-BN 对，同时标记 activation type。
    返回 List[FuseableConvBN]（仅 convbn 融合对，activation 信息直接写在模块属性上）。
    """
    n = len(records)
    pairs = []
    fused_conv_ids, fused_bn_ids = set(), set()

    # 第一遍：Conv -> BN 融合（含 Conv -> BN -> ReLU/Sigmoid）
    if fuse_bn:
        for i in range(n - 1):
            a, b = records[i], records[i + 1]
            if not (_is_conv(a.mod) and _is_bn(b.mod)):
                continue
            if not isinstance(b.mod, _CONV_BN_MATCH.get(type(a.mod), type(None))):
                continue
            if not _connected(a, b, gap):
                continue
            # 检查 BN 后是否紧跟 ReLU/Sigmoid
            act = ActivationType.none
            if i + 2 < n and _connected(b, records[i + 2], gap):
                if isinstance(records[i + 2].mod, nn.ReLU):
                    act = ActivationType.Relu
                elif isinstance(records[i + 2].mod, nn.Sigmoid):
                    act = ActivationType.Sigmoid
            if act != ActivationType.none:
                setattr(a.mod, LINGER_ACTIVATION_TYPE, act)
            # 构建 FuseableConvBN
            conv_f, _ = _get_parent(model, a.name)
            bn_f, _ = _get_parent(model, b.name)
            fb = FuseableConvBN(conv_f, a.mod, bn_f, b.mod)
            fb.scope_conv, fb.scope_bn = a.name, b.name
            pairs.append(fb)
            fused_conv_ids.add(id(a.mod))
            fused_bn_ids.add(id(b.mod))

    # 第二遍：独立 Conv -> ReLU / Conv -> Sigmoid
    for i in range(n - 1):
        a, b = records[i], records[i + 1]
        if _is_conv(a.mod) and id(a.mod) not in fused_conv_ids and _connected(a, b, gap):
            if isinstance(b.mod, nn.ReLU):
                setattr(a.mod, LINGER_ACTIVATION_TYPE, ActivationType.Relu)
            elif isinstance(b.mod, nn.Sigmoid):
                setattr(a.mod, LINGER_ACTIVATION_TYPE, ActivationType.Sigmoid)

    # 第三遍：独立 BN -> ReLU
    for i in range(n - 1):
        a, b = records[i], records[i + 1]
        if _is_bn(a.mod) and id(a.mod) not in fused_bn_ids and _connected(a, b, gap):
            if isinstance(b.mod, nn.ReLU):
                setattr(a.mod, LINGER_ACTIVATION_TYPE, ActivationType.Relu)

    # 第四遍：Linear -> ReLU / Linear -> Sigmoid
    for i in range(n - 1):
        a, b = records[i], records[i + 1]
        if isinstance(a.mod, nn.Linear) and _connected(a, b, gap):
            if isinstance(b.mod, nn.ReLU):
                setattr(a.mod, LINGER_ACTIVATION_TYPE, ActivationType.Relu)
            elif isinstance(b.mod, nn.Sigmoid):
                setattr(a.mod, LINGER_ACTIVATION_TYPE, ActivationType.Sigmoid)

    return pairs


# ==================== 模块替换 ====================

def _create_convbn(conv_m, bn_m):
    """根据 Conv 类型创建对应的 ConvBN 融合模块"""
    cls = _CONV_TO_FUSED[type(conv_m)]
    has_bias = conv_m.bias is not None
    kw = dict(
        in_channels=conv_m.in_channels, out_channels=conv_m.out_channels,
        kernel_size=conv_m.kernel_size, stride=conv_m.stride,
        padding=conv_m.padding, groups=conv_m.groups,
        bias=has_bias, padding_mode=conv_m.padding_mode,
        eps=bn_m.eps, momentum=bn_m.momentum,
        affine=bn_m.affine, track_running_stats=bn_m.track_running_stats,
        constrain=None,
    )
    if isinstance(conv_m, (nn.ConvTranspose1d, nn.ConvTranspose2d)):
        kw['output_padding'] = conv_m.output_padding
    else:
        kw['dilation'] = conv_m.dilation
    return cls(**kw)


def _apply_fusion(model, pairs):
    """执行模块替换：Conv -> ConvBNXd, BN -> EmptyBatchNorm"""
    device = QUANT_CONFIGS.device
    for fb in pairs:
        # 创建融合模块并拷贝 activation type
        act = getattr(fb.conv, LINGER_ACTIVATION_TYPE, ActivationType.none)
        convbn = _create_convbn(fb.conv, fb.bn).to(device)
        setattr(convbn, LINGER_ACTIVATION_TYPE, act)
        # 替换 conv -> ConvBNXd
        conv_parent, conv_attr = _get_parent(model, fb.scope_conv)
        setattr(conv_parent, conv_attr, convbn)
        # 替换 bn -> EmptyBatchNorm
        bn_parent, bn_attr = _get_parent(model, fb.scope_bn)
        setattr(bn_parent, bn_attr, EmptyBatchNorm())


# ==================== 方案C: CheckpointAdapter ====================

class CheckpointAdapter:
    """独立的 state_dict 格式转换器，可序列化保存。

    用法::
        adapter = trace_layers_hook(model, dummy_input)
        adapter.save('adapter.pth')

        # 之后在其他脚本中使用：
        adapter = CheckpointAdapter.load('adapter.pth')
        sd = torch.load('unfused_checkpoint.pth')
        adapter.adapt(sd)
        model.load_state_dict(sd)
    """
    def __init__(self, mappings=None):
        self._mappings = mappings or []  # List[(conv_scope, bn_scope)]

    def add(self, conv_scope, bn_scope):
        self._mappings.append((conv_scope, bn_scope))

    def adapt(self, state_dict, prefix=''):
        """将 unfused state_dict 原地转换为 fused 格式，返回同一 dict"""
        for conv_scope, bn_scope in self._mappings:
            _remap_keys(state_dict, prefix + conv_scope, prefix + bn_scope)
        return state_dict

    def save(self, path):
        torch.save({'mappings': self._mappings}, path)

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, weights_only=False)['mappings'])

    def __repr__(self):
        return f'CheckpointAdapter({len(self._mappings)} fusions)'


def _remap_keys(sd, conv_scope, bn_scope):
    """显式 key 重映射：unfused -> fused ConvBN 格式

    conv_scope.weight -> conv_scope.conv.weight
    bn_scope.weight   -> conv_scope.bn.weight
    （其余 bias / running_mean / running_var / num_batches_tracked 同理）
    """
    # 已经是融合格式则跳过
    if conv_scope + '.conv.weight' in sd:
        return
    if bn_scope + '.' + LINGER_IGNORE_PAMAMTER in sd:
        return
    # 重映射 conv 参数
    for p in _CONV_PARAMS:
        old = conv_scope + '.' + p
        if old in sd:
            sd[conv_scope + '.conv.' + p] = sd.pop(old)
    # 重映射 bn 参数
    for p in _BN_PARAMS:
        old = bn_scope + '.' + p
        if old in sd:
            sd[conv_scope + '.bn.' + p] = sd.pop(old)


# ==================== 方案B: 自动加载 hook ====================

def _register_load_hook(model, adapter):
    """注册 state_dict 预加载 hook，加载时自动重映射 key"""
    # 移除旧 hook
    old = getattr(model, _LINGER_TRACE_HOOK_HANDLE, None)
    if old is not None:
        old.remove()

    def pre_hook(state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        adapter.adapt(state_dict, prefix=prefix)

    handle = model._register_load_state_dict_pre_hook(pre_hook)
    setattr(model, _LINGER_TRACE_HOOK_HANDLE, handle)


# ==================== 主入口 ====================

def trace_layers(model, *args, fuse_bn=True):
    r"""基于 Hook 的算子融合 tracer，替代 JIT-based trace_layers。

    与 trace_layers 的关键差异：
      - 使用 forward hook 而非 torch.jit.trace，兼容动态控制流
      - 三重校验（tensor id + _version + op计数）防止小算子误融合
      - 权重加载同时支持 方案B（自动hook）和 方案C（CheckpointAdapter）

    Args:
        model: 需要 trace 的模型或子模型
        \*args: forward 所需的示例输入
        fuse_bn: 是否执行 Conv-BN 融合

    Returns:
        CheckpointAdapter: 可用于显式转换 unfused state_dict

    Examples::
        # 方案B：自动加载（load_state_dict 时 hook 自动重映射）
        adapter = trace_layers_hook(model, dummy_input)
        model.load_state_dict(torch.load('unfused.pth'))

        # 方案C：显式转换
        adapter = trace_layers_hook(model, dummy_input)
        sd = torch.load('unfused.pth')
        adapter.adapt(sd)
        model.load_state_dict(sd)
    """
    if getattr(model, _LINGER_TRACED, False):
        print("Warning: trace_layers_hook 已在此模块上调用过，将替换之前的融合。")

    # 清理旧的 activation type
    for mod in model.modules():
        if hasattr(mod, LINGER_ACTIVATION_TYPE):
            delattr(mod, LINGER_ACTIVATION_TYPE)

    # 记录执行序列
    records = _record_execution(model, *args)
    adapter = CheckpointAdapter()

    if len(records) >= 2:
        gap = _calc_gap_threshold(records)
        pairs = _analyze(records, gap, model, fuse_bn)
        if pairs:
            _apply_fusion(model, pairs)
            for fb in pairs:
                adapter.add(fb.scope_conv, fb.scope_bn)
            # 方案B：注册自动加载 hook
            _register_load_hook(model, adapter)

    setattr(model, _LINGER_TRACED, True)
    return adapter


__all__ = ['trace_layers_hook', 'CheckpointAdapter',
           'EmptyBatchNorm', 'FuseableConvBN']
