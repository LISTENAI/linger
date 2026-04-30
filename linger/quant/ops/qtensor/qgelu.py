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

VENUSA_GELU_BANDS = (
    0, 14115017, 28375324, 42953268, 58054959, 73951497, 91038024, 109967415,
    132036451, 160959227, 226844274, 267242587, 306191336, 350131144, 407973369,
    516356188, 2147483648,
)
VENUSA_GELU_SLOPE0S = (
    72728288, 83897176, 94867369, 105507160, 115674329, 125208681, 133918928,
    141554901, 147729276, 151456513, 147845039, 143733750, 139917095, 136824421,
    134818073, 134217887,
)
VENUSA_GELU_SLOPE1S = (
    61489440, 50320552, 39350359, 28710568, 18543399, 9009047, 298800, -7337173,
    -13511548, -17238785, -13627311, -9516022, -5699367, -2606693, -600345, -159,
)
VENUSA_GELU_BIASES = (
    -10758, -157495, -447240, -872623, -1421992, -2078164, -2815981, -3596982,
    -4354386, -4913744, -4157062, -3134007, -2046605, -1040047, -281784, -3183,
)


def _shift_floor_x05_int64(x: torch.Tensor, shift: int) -> torch.Tensor:
    if shift >= 64:
        return torch.zeros_like(x)
    if shift > 0:
        x = x >> (shift - 1)
        x = (x & 0x1) + (x >> 1)
    return x


def venusa_gelu_i32(x_q27: torch.Tensor) -> torch.Tensor:
    x_i64 = x_q27.to(torch.int64)
    sign_mask = x_i64 < 0
    absx = torch.where(x_i64 == INT32_MIN, torch.full_like(x_i64, INT32_MAX), torch.abs(x_i64))
    out = torch.zeros_like(x_i64)

    for i in range(16):
        upper = VENUSA_GELU_BANDS[i + 1]
        if i == 0:
            seg_mask = absx <= upper
        else:
            seg_mask = (absx > VENUSA_GELU_BANDS[i]) & (absx <= upper)

        pos_mask = seg_mask & (~sign_mask)
        neg_mask = seg_mask & sign_mask

        pos_val = _shift_floor_x05_int64(
            x_i64 * int(VENUSA_GELU_SLOPE0S[i]) + (int(VENUSA_GELU_BIASES[i]) << 30), 27
        )
        neg_val = _shift_floor_x05_int64(
            x_i64 * int(VENUSA_GELU_SLOPE1S[i]) + (int(VENUSA_GELU_BIASES[i]) << 30), 27
        )
        out = torch.where(pos_mask, pos_val, out)
        out = torch.where(neg_mask, neg_val, out)

    out = torch.clamp(out, min=INT32_MIN, max=INT32_MAX)
    return out.to(torch.int32)


class QGeluOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qparam_dict=None):
        return F.gelu(input)

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


class QGeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_quantizer):
        ctx.save_for_backward(input)

        if QUANT_CONFIGS.platform != PlatForm.venusA:
            raise RuntimeError("QGelu only supports venusA platform")

        x = input.contiguous()
        scale_x = input_quantizer.scale
        quant_x = input_quantizer.quant_round(x * scale_x, input_quantizer.round_mode)

        l_scale = 27 - int(math.log2(scale_x.data))
        if l_scale > 0:
            x_q27 = (quant_x * pow(2, l_scale)).to(torch.int32)
        else:
            x_q27 = (quant_x * pow(2, l_scale) + 0.5).floor().to(torch.int32)
        x_q27.clamp_(-2**31, 2**31 - 1)

        output_q27 = venusa_gelu_i32(x_q27.contiguous())
        return output_q27.float() / float(2**27)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            y = F.gelu(x)
            grad = torch.autograd.grad(y, x, grad_output)
        return grad[0], None


@register_qmodule(nn.GELU)
class QGelu(QModuleTensor):
    @classmethod
    def qcreate(
        cls,
        module: torch.nn.Module,
        activate_config: Optional[Dict[str, Any]] = None,
        num_input: int = 1,
        dim: int = -1
    ):
        if module is not None and getattr(module, "approximate", "none") != "none":
            return None
        return cls(
            activate_config=activate_config,
            num_input=num_input
        )

    def forward(self, x, *args, **kwargs):
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, True)
            return QGeluOnnxFunction.apply(x, qparam_dict)
        if QUANT_CONFIGS.calibration:
            return F.gelu(x)
        if isinstance(self.input_quantizer, nn.ModuleList):
            return QGeluFunction.apply(x, self.input_quantizer[0])
        return QGeluFunction.apply(x, self.input_quantizer)
