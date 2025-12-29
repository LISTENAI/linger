import torch
import torch.nn.functional as F
from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME
from typing import Optional, Union, Dict, Any

class QConv1dOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, stride, padding, dilation, groups, qparam_dict = None):
        return F.conv1d(input, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    @staticmethod
    def symbolic(g,  input, weights, bias, stride, padding, dilation, groups, qparam_dict = None):
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

@register_qmodule(torch.nn.Conv1d)
class QConv1d(QModuleMixin, torch.nn.Conv1d):
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
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            dtype=module.weight.dtype,
            device=device,
            activations_cfg=activations_cfg,
            weights_cfg=weights_cfg,
            bias_cfg=bias_cfg,
            constrain = constrain, 
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, False)
            return QConv1dOnnxFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, qparam_dict)
        return self._conv_forward(input, self.qweight, self.qbias)

