import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME
from typing import Optional, Union, Dict, Any

class QAvgPool2dOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, qparam_dict = None):
        return F.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    @staticmethod
    def symbolic(g,  input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, qparam_dict = None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        is_input_qtensor = qparam_dict.get("is_input_qtensor", None)
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop('op_type', None)
        qparam_dict.pop('is_input_qtensor', None)
        if is_input_qtensor is False or is_input_qtensor is None:
            op_inner = quantlinear(g, input, qparam_dict['scale_x_f'], qparam_dict['platform_s'], qparam_dict['x_bits_i'], 0)
            input_list = [op_inner]
        else:
            input_list = [input]
        return g.op(
                node_name,
                *input_list,
                **qparam_dict
            )

@register_qmodule(nn.AvgPool2d)
class QAvgPool2d(QModuleMixin, nn.AvgPool2d):
    @classmethod
    def qcreate(
        cls,
        module,
        activations_cfg: Optional[Dict[str, Any]] = None,
        weights_cfg: Optional[Dict[str, Any]] = None,
        bias_cfg: Optional[Dict[str, Any]] = None,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[Dict[str, Any]] = None,
    ):
        return cls(
            kernel_size = module.kernel_size,
            stride = module.stride,
            padding = module.padding,
            ceil_mode = module.ceil_mode,
            count_include_pad = module.count_include_pad,
            divisor_override = module.divisor_override,

            device = device,
            activations_cfg=activations_cfg,
            weights_cfg=None,
            constrain=None, 
            bias_cfg=None,
            open_ihook = True,
            open_ohook = True,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, False)
            return QAvgPool2dOnnxFunction.apply(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override, qparam_dict)
        return F.avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )
        

