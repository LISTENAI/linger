import itertools
import operator
from contextlib import contextmanager
from fnmatch import fnmatch

import torch
import torch.nn as nn
# from torch.fx import symbolic_trace, GraphModule

from .quant.qtensor import QTensor
from .quant.ops import *
from .config import QuantConfig, QUANT_CONFIGS
from .quant.ops.qconfig import _QMODULE_TABLE, _QTENSOR_OP_TABLE, quantize_module, quantize_tensor
from .constrain.cmodule import constrain_module, _CMODULE_TABLE
from typing import Any, Dict, List, Optional, Union

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

@contextmanager
def calibration():
    # 保存旧值
    # old_a_calibrate_name = QUANT_CONFIGS.quant_info.a_calibrate_name
    # old_w_calibrate_name = QUANT_CONFIGS.quant_info.w_calibrate_name
    try:
        QUANT_CONFIGS.calibration = True
        # QUANT_CONFIGS.quant_info.a_calibrate_name = a_calibrate_name
        # QUANT_CONFIGS.quant_info.w_calibrate_name = w_calibrate_name
        yield  # <<< 关键点：控制权交给 with 块
    finally:
        QUANT_CONFIGS.calibration = False

def const_module(module: nn.Module, c_activation_val: float = 8.0, c_weight_val: float = 8.0, c_bias_val = None, c_weight_factor = None):
    for name, m in module.named_modules():
        if hasattr(m, 'clamp_weight'):
            m.clamp_weight = c_weight_val
            m.clamp_factor = c_weight_factor
        if hasattr(m, 'clamp_bias'):
            m.clamp_bias = c_bias_val
        if hasattr(m, 'clamp_activation'):
            m.clamp_activation = c_activation_val

def quant_module(module: nn.Module, c_activation_val: float = 8.0, c_weight_val: float = 8.0, c_bias_val = None, c_weight_factor = None, data_bits: int = 8, weight_bits: int = 8, bias_bits: int = 32, out_bits: int = 8):
    for name, m in module.named_modules():
        if hasattr(m, 'input_quantizer') and m.input_quantizer is not None:
            m.input_quantizer.data_bits = data_bits
        if hasattr(m, 'weight_quantizer') and m.weight_quantizer is not None:
            m.weight_quantizer.data_bits = weight_bits
            m.weight_quantizer.clamp_weight_value = c_weight_val
            m.weight_quantizer.clamp_factor_value = c_weight_factor
        if hasattr(m, 'bias_quantizer') and m.bias_quantizer is not None:
            m.bias_quantizer.data_bits = bias_bits
            m.bias_quantizer.clamp_bias_value = c_bias_val
        if hasattr(m, 'output_quantizer') and m.output_quantizer is not None:
            m.output_quantizer.data_bits = out_bits
            m.output_quantizer.clamp_activation_value = c_activation_val

def constrain(model: nn.Module, config_file: str = None, disable_module=None, disable_submodel=None):
    c_configs = QUANT_CONFIGS
    if config_file is not None:
        c_configs._load_from_yaml(config_file)

    if disable_module is not None:
        for name in disable_module:
            if _CMODULE_TABLE.get(name, None) is not None:
                _CMODULE_TABLE.pop(name)

    for name, m in model.named_modules():
        if disable_submodel is not None and any(fnmatch(name, pattern) for pattern in disable_submodel):
            continue
        _constrain_submodule(model, name, m, c_configs.clamp_info.to_dict())

    model.to(c_configs.device)
    return model

def init(model: nn.Module, config_file: str = None, disable_module=None, disable_submodel=None):

    q_configs = QUANT_CONFIGS
    if config_file is not None:
        q_configs._load_from_yaml(config_file)
    
    if disable_module is not None:
        for name in disable_module:
            if _QMODULE_TABLE.get(name, None) is not None:
                _QMODULE_TABLE.pop(name)
                
    # traced_model = symbolic_trace(model)
    # model = _replace_ops(traced_model, q_configs)

    has_replaced = []
    for name, m in model.named_modules():
        if disable_submodel is not None and any(fnmatch(name, pattern) for pattern in disable_submodel):
            continue
        if any(name.startswith(p + ".") for p in has_replaced):
            continue
        
        m.register_forward_pre_hook(hook_pre_forward)
        m.register_forward_hook(hook_forward)

        is_replaced = _quantize_submodule(model, name, m, weights_cfg=q_configs.quant_info.to_dict(), activations_cfg=q_configs.quant_info.to_dict(), bias_cfg=q_configs.quant_info.to_dict(), constrain =  q_configs.clamp_info.to_dict())
        if is_replaced:
            has_replaced.append(name)

    def quant_tensor_pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    
        def quant_tensor_layer(module, prefix=''):
            local_name_params = itertools.chain(
                module._parameters.items(), module._buffers.items())
            local_state = {k: v for k, v in local_name_params if v is not None}
            for key in state_dict.keys():
                if LINGER_QTENSOR_LAYERS_PREIFX in key:
                    if key.startswith(prefix):
                        full_input_name = key[len(prefix):]
                        # get the name of param/buffer/child
                        input_name = full_input_name.split('.', 1)[0]
                        if input_name not in module._modules and input_name not in local_state:
                            # quant_info = getattr(module, LINGER_QUANTINFO, QuantInfo())
                            activate_cfg  = q_configs.quant_info.to_dict()
                            if '_qadd_' in input_name:
                                iq_layer = QAdd(activate_config=activate_cfg, num_input=2)
                                iq_layer.training = model.training
                                # iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_qmul_' in input_name:
                                iq_layer = QMul(activate_config=activate_cfg, num_input=2)
                                iq_layer.training = model.training
                                # iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_qcat_' in input_name:
                                iq_layer = QCat(activate_config=activate_cfg, num_input=2, is_cat=True)
                                iq_layer.training = model.training
                                # iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_qbmm_' in input_name:
                                iq_layer = QBmm(activate_config=activate_cfg, num_input=2)
                                iq_layer.training = model.training
                                # iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_qmatmul_' in input_name:
                                iq_layer = QMatmul(activate_config=activate_cfg, num_input=2)
                                iq_layer.training = model.training
                                # iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_qsigmoid_' in input_name:
                                iq_layer = QSigmoid(activate_config=activate_cfg, num_input=1)
                                iq_layer.training = model.training
                                # iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_qtanh_' in input_name:
                                iq_layer = QTanh(activate_config=activate_cfg, num_input=1)
                                iq_layer.training = model.training
                                # iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            elif '_qsoftmax_' in input_name:
                                iq_layer = QSoftmax(activate_config=activate_cfg, num_input=1)
                                iq_layer.training = model.training
                                # iq_layer = iq_layer.to(device)
                                setattr(module, input_name, iq_layer)
                            else:
                                pass

            for name, children in module._modules.items():
                if children is not None:
                    quant_tensor_layer(children, prefix + name + '.')

        quant_tensor_layer(model, prefix)
        quant_tensor_layer = None

    model._register_load_state_dict_pre_hook(quant_tensor_pre_hook)

    model.to(q_configs.device)
    return model


# 关闭QTensor类算子时使用
def disable_quant_ops(qmodule_list = [], qtensor_list = []):
    """
    删除_QMODULE_TABLE, _QTENSOR_OP_TABLE中的量化操作
    """
    pop_list = []
    for name in qmodule_list:
        for k in _QMODULE_TABLE.keys():
            if name == k:
                pop_list.append(k)
    for k in pop_list:
        _QMODULE_TABLE.pop(k)
    
    pop_list = []
    for name in qtensor_list:
        for k in _QTENSOR_OP_TABLE.keys():
            if name in str(k):
                pop_list.append(k)
    for k in pop_list:
        _QTENSOR_OP_TABLE.pop(k)

def get_quant_ops_name():
    return _QMODULE_TABLE.keys(), _QTENSOR_OP_TABLE.keys()

def config_save_to_yaml(yaml_save_path):
    QUANT_CONFIGS._save_to_yaml(yaml_save_path)



def _set_module_by_name(parent_module, name, child_module):
    module_names = name.split(".")
    if len(module_names) == 1:
        setattr(parent_module, name, child_module)
    else:
        parent_module_name = name[: name.rindex(".")]
        parent_module = parent_module.get_submodule(parent_module_name)
        setattr(parent_module, module_names[-1], child_module)

def _quantize_submodule(
    model: torch.nn.Module,
    name: str,
    module: torch.nn.Module,
    weights_cfg: Optional[Union[str]] = None,
    activations_cfg: Optional[Union[str]] = None,
    bias_cfg: Optional[Union[str]] = None,
    constrain: Optional[Union[str]] = None,
):
    qmodule = quantize_module(module, weights_cfg=weights_cfg, activations_cfg=activations_cfg, bias_cfg = bias_cfg, dim = getattr(module, "dim", None), constrain = constrain)
    if isinstance(module, ConvBN1d) or isinstance(module, CConvBN1d)  \
        or isinstance(module, ConvBN2d) or isinstance(module, CConvBN2d) \
        or isinstance(module, ConvTransposeBN1d) or isinstance(module, CConvTransposeBN1d) \
        or isinstance(module, ConvTransposeBN2d) or isinstance(module, CConvTransposeBN2d):
        qmodule._register_load_state_dict_pre_hook(fuse_state_dict)

    if qmodule is not None:
        _set_module_by_name(model, name, qmodule)
        qmodule.name = name
        for name, param in module.named_parameters():
            # Save device memory by clearing parameters
            setattr(module, name, None)
            del param
        return True
    return False

def _constrain_submodule(
    model: torch.nn.Module,
    name: str,
    module: torch.nn.Module,
    constrain: Optional[Union[str]] = None,
):
    cmodule = constrain_module(module, constrain=constrain)
    if cmodule is not None:
        _set_module_by_name(model, name, cmodule)
        cmodule.name = name
        for name, param in module.named_parameters():
            # Save device memory by clearing parameters
            setattr(module, name, None)
            del param


# def _replace_ops(gm: GraphModule, quant_cfg: QuantConfig) -> GraphModule:
#     graph = gm.graph
#     qtensor_counter = 0
#     activate_cfg  = quant_cfg.quant_info.to_dict()
#     constrain_cfg = quant_cfg.clamp_info.to_dict()
#     for node in list(graph.nodes):
#         if node.op == "call_function":
#             new_node_mod = None
#             new_node_name = None
#             new_node = None
#             new_node_mod = quantize_tensor(node.target, activate_cfg, num_input = len(node.args), dim = node.kwargs.get('dim', None))

#             if new_node_mod is not None:
#                 new_node_name = f"{new_node_mod._get_name()}_{qtensor_counter}"
#                 qtensor_counter += 1
#                 with graph.inserting_after(node):
#                     gm.add_module(new_node_name, new_node_mod)
#                     new_node = graph.call_module(new_node_name, args=node.args)
#                     node.replace_all_uses_with(new_node)
#                     graph.erase_node(node)
#         elif node.op == "call_module":
#             old_mod = gm.get_submodule(node.target)
#             weights_cfg = quant_cfg.quant_info.to_dict()
#             activate_cfg  = quant_cfg.quant_info.to_dict()
#             bias_cfg = quant_cfg.quant_info.to_dict()
#             new_node_mod = quantize_module(old_mod, activate_cfg, weights_cfg = weights_cfg, bias_cfg = bias_cfg, constrain = constrain_cfg, dim = getattr(old_mod, "dim", None))
#             if new_node_mod is not None and new_node_mod is not old_mod:
#                 gm.add_submodule(node.target, new_node_mod)
#     graph.lint()
#     gm.recompile()
#     return gm
