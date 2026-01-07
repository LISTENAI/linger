import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME
from typing import Optional, Union, Dict, Any

class QLinearOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, qparam_dict = None):
        return F.linear(input, weights, bias)
    @staticmethod
    def symbolic(g,  input, weights, bias, qparam_dict = None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        is_input_qtensor = qparam_dict.get("is_input_qtensor", None)
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop('op_type', None)
        qparam_dict.pop('is_input_qtensor', None)
        if is_input_qtensor is False or is_input_qtensor is None:
            op_inner = quantlinear(g, input, qparam_dict['scale_x_f'], qparam_dict['platform_s'], qparam_dict['x_bits_i'], 0)
            input_list = [op_inner, weights]
        else:
            input_list = [input, weights]
        if bias is not None:
            input_list.append(bias)
        return g.op(
                node_name,
                *input_list,
                **qparam_dict
            )

@register_qmodule(torch.nn.Linear)
class QLinear(QModuleMixin, nn.Linear):
    @classmethod
    def qcreate(
        cls,
        module,
        activations_cfg: Optional[Dict[str, Any]] = None,
        weights_cfg: Optional[Dict[str, Any]] = None,
        bias_cfg: Optional[Dict[str, Any]] = None,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype=module.weight.dtype,
            device=device,
            activations_cfg=activations_cfg,
            weights_cfg=weights_cfg,
            bias_cfg=bias_cfg if module.bias is not None else None,
            constrain = constrain, 
        )

    def forward(self, input):
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, False)
            return QLinearOnnxFunction.apply(input, self.qweight, self.qbias, qparam_dict)
        return F.linear(input, self.qweight, bias=self.qbias)

