import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from .qavgpool_chip import simulate_avgpool1d
from ..qconfig import register_qmodule
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME
from ....utils import QuantMode
from typing import Optional, Dict, Any

class QAvgPool1dOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, ceil_mode, count_include_pad, qparam_dict = None):
        return F.avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    @staticmethod
    def symbolic(g,  input, kernel_size, stride, padding, ceil_mode, count_include_pad, qparam_dict = None):
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

@register_qmodule(nn.AvgPool1d)
class QAvgPool1d(QModuleMixin, nn.AvgPool1d):
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
            return QAvgPool1dOnnxFunction.apply(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad, qparam_dict)
        ref_out = F.avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )
        input_scale = 1.0
        round_mode = QuantMode.floor_add
        if hasattr(self, "input_quantizer"):
            input_scale = float(self.input_quantizer.scale)
            round_mode = self.input_quantizer.round_mode
        aligned_out = simulate_avgpool1d(
            input,
            scale=input_scale,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=None,
            round_mode=round_mode,
        )
        if self.training:
            return ref_out + (aligned_out - ref_out).detach()
        return aligned_out
        
