import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from ....constrain import ConvTransposeBN1d, CConvTransposeBN1d
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME
from typing import Optional, Union, Dict, Any

class QConvTranspose1dOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding, groups, dilation, qparam_dict = None):
        return F.conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation)
    @staticmethod
    def symbolic(g,  input, weight, bias, stride, padding, output_padding, groups, dilation, qparam_dict= None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        is_input_qtensor = qparam_dict.get("is_input_qtensor", None)
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop('op_type', None)
        qparam_dict.pop('is_input_qtensor', None)
        if is_input_qtensor is False or is_input_qtensor is None:
            op_inner = quantlinear(g, input, qparam_dict['scale_x_f'], qparam_dict['platform_s'], qparam_dict['x_bits_i'], 0)
            input_list = [op_inner, weight]
        else:
            input_list = [input, weight]
        if bias is not None:
            input_list.append(bias)
        return g.op(
                node_name,
                *input_list,
                **qparam_dict
            )

@register_qmodule(ConvTransposeBN1d)
@register_qmodule(CConvTransposeBN1d)
@register_qmodule(torch.nn.ConvTranspose1d)
class QConvTranspose1d(QModuleMixin, torch.nn.ConvTranspose1d):
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
        if "ConvTransposeBN1d" in module._get_name():
            return cls(
                in_channels=module.conv.in_channels,
                out_channels=module.conv.out_channels,
                kernel_size=module.conv.kernel_size,
                stride=module.conv.stride,
                padding=module.conv.padding,
                output_padding=module.conv.output_padding,
                groups=module.conv.groups,
                bias=True,
                dilation=module.conv.dilation,
                padding_mode=module.conv.padding_mode,
                dtype=module.conv.weight.dtype,
                
                device=device,
                activations_cfg=activations_cfg,
                weights_cfg=weights_cfg,
                bias_cfg=bias_cfg,
                constrain=constrain, 
            )
        else:
            return cls(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                output_padding=module.output_padding,
                groups=module.groups,
                bias=module.bias is not None,
                dilation=module.dilation,
                padding_mode=module.padding_mode,
                dtype=module.weight.dtype,
                device=device,
                activations_cfg=activations_cfg,
                weights_cfg=weights_cfg,
                bias_cfg=bias_cfg,
                constrain = constrain, 
            )

    def forward(self, input: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, False)
            return QConvTranspose1dOnnxFunction.apply(input, self.qweight, self.qbias, self.stride, self.padding, self.output_padding, self.groups, self.dilation, qparam_dict)
        return F.conv_transpose1d(
            input, 
            self.qweight, 
            self.qbias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,)

