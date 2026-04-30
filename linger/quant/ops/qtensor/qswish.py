import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .qtensor_mod import QModuleTensor
from ..qconfig import register_qmodule
from ....config import QUANT_CONFIGS
from ....utils import PlatForm
from ....onnx import generate_onnx_qparam_dict, quant_qtensor_symbolic_input, QDOMAIN_NAME

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

VENUSA_SWISH_BANDS = (
    0, 24654376, 49696796, 75597255, 102943334, 132562612, 165787474, 205189411,
    257698662, 386828914, 411601113, 503293022, 600623812, 719278864, 886211418,
    1200463913, 2147483648,
)
VENUSA_SWISH_SLOPE0S = (
    73253038, 85425077, 97268252, 108569000, 119102644, 128623281, 136842248,
    143368061, 147330902, 146163034, 144088169, 140854555, 138082720, 135988566,
    134689659, 134226579,
)
VENUSA_SWISH_SLOPE1S = (
    60964690, 48792651, 36949476, 25648728, 15115084, 5594447, -2624520, -9150333,
    -13113174, -11945306, -9870441, -6636827, -3864992, -1770838, -471931, -8851,
)
VENUSA_SWISH_BIASES = (
    -18406, -297641, -845322, -1640240, -2649106, -3823039, -5089876, -6333243,
    -7277526, -6866303, -6053889, -4539545, -2991709, -1593178, -528753, -24754,
)


def _shift_floor_x05_int64(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift >= 64:
        return torch.zeros_like(x)
    if shift > 0:
        x = x >> (shift - 1)
        x = (x & 0x1) + (x >> 1)
    return x


def venusa_swish_i32(x_q27: torch.Tensor) -> torch.Tensor:
    x_i64 = x_q27.to(torch.int64)
    sign_mask = x_i64 < 0
    absx = torch.where(x_i64 == INT32_MIN, torch.full_like(x_i64, INT32_MAX), torch.abs(x_i64))
    out = torch.zeros_like(x_i64)

    for i in range(16):
        upper = VENUSA_SWISH_BANDS[i + 1]
        if i == 0:
            seg_mask = absx <= upper
        else:
            seg_mask = (absx > VENUSA_SWISH_BANDS[i]) & (absx <= upper)

        pos_mask = seg_mask & (~sign_mask)
        neg_mask = seg_mask & sign_mask

        pos_val = _shift_floor_x05_int64(
            x_i64 * int(VENUSA_SWISH_SLOPE0S[i]) + (int(VENUSA_SWISH_BIASES[i]) << 30), 27
        )
        neg_val = _shift_floor_x05_int64(
            x_i64 * int(VENUSA_SWISH_SLOPE1S[i]) + (int(VENUSA_SWISH_BIASES[i]) << 30), 27
        )
        out = torch.where(pos_mask, pos_val, out)
        out = torch.where(neg_mask, neg_val, out)

    out = torch.clamp(out, min=INT32_MIN, max=INT32_MAX)
    return out.to(torch.int32)


class QSwishOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qparam_dict=None):
        return F.silu(input)

    @staticmethod
    def symbolic(g, input, qparam_dict=None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop("op_type", None)
        input = quant_qtensor_symbolic_input(g, input, qparam_dict, 0)
        output = g.op(node_name, input, **qparam_dict)
        if input.type() is not None:
            output.setType(input.type())
        return output


class QSwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_quantizer):
        ctx.save_for_backward(input)

        if QUANT_CONFIGS.platform != PlatForm.venusA:
            raise RuntimeError("QSwish only supports venusA platform")

        x = input.contiguous()
        scale_x = input_quantizer.scale
        quant_x = input_quantizer.quant_round(x * scale_x, input_quantizer.round_mode)

        l_scale = 27 - int(math.log2(scale_x.data))
        if l_scale > 0:
            x_q27 = (quant_x * pow(2, l_scale)).to(torch.int32)
        else:
            x_q27 = (quant_x * pow(2, l_scale) + 0.5).floor().to(torch.int32)
        x_q27.clamp_(-2**31, 2**31 - 1)

        output_q27 = venusa_swish_i32(x_q27.contiguous())
        return output_q27.float() / float(2**27)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            y = F.silu(x)
            grad = torch.autograd.grad(y, x, grad_output)
        return grad[0], None


@register_qmodule(nn.SiLU)
class QSwish(QModuleTensor):
    @classmethod
    def qcreate(
        cls,
        module: torch.nn.Module,
        activate_config: Optional[Dict[str, Any]] = None,
        num_input: int = 1,
        dim: int = -1
    ):
        return cls(
            activate_config=activate_config,
            num_input=num_input
        )

    def forward(self, x, *args, **kwargs):
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, True)
            return QSwishOnnxFunction.apply(x, qparam_dict)
        if QUANT_CONFIGS.calibration:
            return F.silu(x)
        if isinstance(self.input_quantizer, nn.ModuleList):
            return QSwishFunction.apply(x, self.input_quantizer[0])
        return QSwishFunction.apply(x, self.input_quantizer)
