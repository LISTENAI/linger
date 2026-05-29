import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .qmodule import QModuleMixin
from .qavgpool_chip import simulate_global_avgpool2d
from ..qconfig import register_qmodule
from ...quantizer import AQuantizer
from ...qtensor import from_tensor_to_qtensor
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME
from ....utils import QuantMode


def _normalize_output_size(output_size):
    if isinstance(output_size, int):
        return (output_size, output_size)
    if isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        return tuple(output_size)
    raise ValueError(f"unsupported AdaptiveAvgPool2d output_size: {output_size}")


def _check_global_output_size(output_size):
    if _normalize_output_size(output_size) != (1, 1):
        raise NotImplementedError("QAdaptiveAvgPool2d only supports output_size=(1, 1)")


class QGlobalAveragePool2dOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qparam_dict=None):
        return input.mean(dim=(-2, -1), keepdim=True)

    @staticmethod
    def symbolic(g, input, qparam_dict=None):
        op_type = qparam_dict.get("op_type", "QGlobalAveragePool2d")
        is_input_qtensor = qparam_dict.get("is_input_qtensor", None)
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop("op_type", None)
        qparam_dict.pop("is_input_qtensor", None)

        if is_input_qtensor is False or is_input_qtensor is None:
            input = quantlinear(
                g,
                input,
                qparam_dict["scale_x_f"],
                qparam_dict["platform_s"],
                qparam_dict["x_bits_i"],
                0,
            )
        return g.op(node_name, input, **qparam_dict)


@register_qmodule(nn.AdaptiveAvgPool2d)
class QAdaptiveAvgPool2d(QModuleMixin, nn.AdaptiveAvgPool2d):
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
        _check_global_output_size(module.output_size)
        qmodule = cls(
            output_size=module.output_size,
            device=device,
            activations_cfg=activations_cfg,
            weights_cfg=None,
            constrain=None,
            bias_cfg=None,
            open_ihook=True,
            open_ohook=False,
        )
        qmodule.add_module("output_quantizer", AQuantizer(activations_cfg, constrain))
        return qmodule

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        _check_global_output_size(self.output_size)
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, False)
            output = QGlobalAveragePool2dOnnxFunction.apply(input, qparam_dict)
            return from_tensor_to_qtensor(output, self.output_quantizer.scale, self.output_quantizer.data_bits)

        ref_out = input.mean(dim=(-2, -1), keepdim=True)
        ref_out = self.output_quantizer(ref_out)
        input_scale = 1.0
        input_round_mode = QuantMode.floor_add
        if hasattr(self, "input_quantizer"):
            input_scale = float(self.input_quantizer.scale)
            input_round_mode = self.input_quantizer.round_mode
        output_scale = float(self.output_quantizer.scale)
        output_round_mode = self.output_quantizer.round_mode
        aligned_out = simulate_global_avgpool2d(
            input,
            input_scale=input_scale,
            output_scale=output_scale,
            input_round_mode=input_round_mode,
            output_round_mode=output_round_mode,
            output_quant_min=self.output_quantizer.quant_min,
            output_quant_max=self.output_quantizer.quant_max,
        )
        if self.training:
            aligned_out = ref_out + (aligned_out - ref_out).detach()
        return from_tensor_to_qtensor(
            aligned_out,
            self.output_quantizer.scale,
            self.output_quantizer.data_bits,
        )
