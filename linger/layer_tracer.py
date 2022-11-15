import torch.nn as nn

from .conv_bn_fuser import *


def trace_layers(root_model: nn.Module, target_model: nn.Module, *args, fuse_bn: bool = True, ahead_conv_relu: bool = True, ahead_bn_relu: bool = True, ahead_linear_relu: bool = True, ahead_conv_sigmoid: bool = True, ahead_linear_sigmoid: bool = True) -> nn.Module:
    r"""对模型进行trace 同时进行模型提前进行fusion训练,root_model为原始的根module,target_model为目标trace的子model

    Args:
        root_model(torch.nn.Module): 原始根模型
        target_model(torch.nn.Module): Trace 子模型
        args(torch.Tensor or Tuple or List): 模型Trace的位置参数
        fuse_bn(bool)：是否融合BN 
        ahead_conv_relu(bool): 是否统计Conv输出正值scale
        ahead_bn_relu(bool): 是否统计BN输出正值scale 
        ahead_linear_relu(bool): 是否统计Linear输出正值scale
    returns:
        返回融合BN后的module

    Examples:
        test/test_trace_layers.py

    Note:
        trace_layer 只支持一次调用，对于多次调用时，会直接覆盖前一次trace_layer的实现，但并未体现在输出的net的网络结构中，内部hook会被清空，导致加载的融合参数无法生效

    """
    if SingletonConvFusedBnModules().has_fuseable_items():
        print("Warning: trace_layers only support one-time call, the latest call will overwrite the previous call and this may cause errors, please check !")
    FuseConvBNAheadRelu(target_model, *args, fused_bn=fuse_bn, ahead_conv_relu=ahead_conv_relu, ahead_bn_relu=ahead_bn_relu,
                        ahead_linear_relu=ahead_linear_relu, ahead_conv_sigmoid=ahead_conv_sigmoid, ahead_linear_sigmoid=ahead_linear_sigmoid)
    if SingletonConvFusedBnModules().has_fuseable_items():
        SingletonConvFusedBnModules().build_normalize_convbn2d_scope(root_model)
        SingletonConvFusedBnModules().build_empty_bn_scope(root_model)

    def pre_hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        SingletonConvFusedBnModules().fuse_state_dicts(state_dict)
        SingletonConvFusedBnModules().clear()
    if SingletonConvFusedBnModules().has_fuseable_items():
        root_model._register_load_state_dict_pre_hook(pre_hook)
    return root_model
