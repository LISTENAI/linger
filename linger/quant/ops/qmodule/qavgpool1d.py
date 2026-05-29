import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from .qavgpool_chip import simulate_avgpool1d
from ..qconfig import register_qmodule
from ...quantizer import AQuantizer
from ...qtensor import from_tensor_to_qtensor
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME
from ....utils import QuantMode
from ....config import QUANT_CONFIGS
from ....utils import PlatForm
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
        _kernel_size = module.kernel_size
        _stride = module.stride
        qmodule = cls(
            kernel_size=_kernel_size if _kernel_size is None else None,
            stride=_stride if _stride is None else None,
            padding = module.padding,
            ceil_mode = module.ceil_mode,
            count_include_pad = module.count_include_pad,

            device = device,
            activations_cfg=activations_cfg,
            weights_cfg=None,
            constrain=None, 
            bias_cfg=None,
            open_ihook = True,
            open_ohook = False,
        )
        # Restore kernel_size and stride after parent __init__
        qmodule.kernel_size = _kernel_size
        qmodule.stride = _stride

        qmodule.add_module("output_quantizer", AQuantizer(activations_cfg, constrain))
        return qmodule

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Validate kernel_size and stride in forward
        if hasattr(self, "kernel_size") and self.kernel_size is not None:
            kernel_size = self.kernel_size
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size,)
            input_dim = input.shape[-1]  # For 1D, check last dimension
            if QUANT_CONFIGS.platform in {PlatForm.venusA, PlatForm.arcs, PlatForm.mars}:
                for k in kernel_size:
                    if k != input_dim:
                        assert 1 <= k <= 12, f"kernel size of venusA/arcs/mars should be in range [1, 12], got {k}"
            elif QUANT_CONFIGS.platform in {PlatForm.venus}:
                for k in kernel_size:
                    if k != input_dim:
                        assert 1 <= k <= 5, f"kernel size of venus should be in range [1, 5], got {k}"

        if hasattr(self, "stride") and self.stride is not None:
            stride = self.stride
            if isinstance(stride, int):
                stride = (stride,)
            input_dim = input.shape[-1]  # For 1D, check last dimension
            for s in stride:
                if s != input_dim:
                    assert s in {1, 2, 4}, f"stride of venus/arcs/mars/venusA should be in [1, 2, 4], got {s}"

        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, False)
            output = QAvgPool1dOnnxFunction.apply(input, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad, qparam_dict)
            return from_tensor_to_qtensor(output, self.output_quantizer.scale, self.output_quantizer.data_bits)
        ref_out = F.avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )
        ref_out = self.output_quantizer(ref_out)
        input_scale = 1.0
        input_round_mode = QuantMode.floor_add
        if hasattr(self, "input_quantizer"):
            input_scale = float(self.input_quantizer.scale)
            input_round_mode = self.input_quantizer.round_mode
        output_scale = float(self.output_quantizer.scale)
        output_round_mode = self.output_quantizer.round_mode
        aligned_out = simulate_avgpool1d(
            input,
            input_scale=input_scale,
            output_scale=output_scale,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=None,
            input_round_mode=input_round_mode,
            output_round_mode=output_round_mode,
            output_quant_min=self.output_quantizer.quant_min,
            output_quant_max=self.output_quantizer.quant_max,
        )
        if self.training:
            aligned_out = ref_out + (aligned_out - ref_out).detach()
        return from_tensor_to_qtensor(aligned_out, self.output_quantizer.scale, self.output_quantizer.data_bits)
        
