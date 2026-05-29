import math

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .qtensor_mod import QModuleTensor
from ...qtensor import from_tensor_to_qtensor
from ....config import QUANT_CONFIGS
from ....utils import PlatForm
from ....onnx import generate_onnx_qparam_dict, quant_qtensor_symbolic_input, QDOMAIN_NAME

import lingerext

TANH_Q31 = 31
TANH_VENUSA_RAW_Q = 58
RUNTIME_TANH_PLATFORMS = {PlatForm.arcs, PlatForm.mars, PlatForm.venusA}


def _shift_floor_x05_int64(x, shift):
    if shift <= 0:
        return x << (-shift)
    if shift >= 64:
        return torch.zeros_like(x)
    val = x >> (shift - 1)
    return (val & 1) + (val >> 1)


def tanh_runtime_qbits(platform):
    if platform in {PlatForm.arcs, PlatForm.mars}:
        return TANH_Q31
    if platform == PlatForm.venusA:
        return TANH_VENUSA_RAW_Q
    raise RuntimeError(f"QTanh does not support platform {platform}")


def tanh_requant_to_qbits(value, src_q_bits, q_bits):
    src_q_bits = int(src_q_bits)
    q_bits = int(q_bits)
    quant_min = -(1 << q_bits)
    quant_max = (1 << q_bits) - 1
    q = _shift_floor_x05_int64(value.to(torch.int64), src_q_bits - q_bits)
    return q.clamp(quant_min, quant_max)


def _fixed_scale_fake_quant(output, quantizer, out_scale):
    if quantizer.data_bits <= 24:
        return quantizer(output, out_scale)

    scale = torch.tensor(float(1 << (quantizer.data_bits - 1)), dtype=torch.float64, device=output.device)
    output_int = quantizer.quant_round(output.to(torch.float64) * scale, quantizer.round_mode)
    output_int = output_int.clamp(quantizer.quant_min, quantizer.quant_max)
    quantizer.scale.fill_(float(scale))
    return output_int / scale


class QTanhOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, qparam_dict = None):
        return torch.tanh(input)
    @staticmethod
    def symbolic(g, input, qparam_dict = None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop('op_type', None)
        input_list = [quant_qtensor_symbolic_input(g, input, qparam_dict, 0)]
        return g.op(
                node_name,
                *input_list,
                **qparam_dict
            )

class QTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_quantizer, output_bits):
        ctx.save_for_backward(input)
        
        x = input.contiguous()
        
        scale_x = input_quantizer.scale
        quant_x = input_quantizer.quant_round(x * scale_x, input_quantizer.round_mode)
        
        if QUANT_CONFIGS.platform == PlatForm.venus:    # Q11->Q15
            l_scale = 11 - int(math.log2(scale_x.data))
            if l_scale > 0:
                x_q11 = (quant_x * pow(2, l_scale)).int()
            else:
                x_q11 = (quant_x * pow(2, l_scale) + 0.5).floor().int()
            output_q15 = lingerext.venus_qtanh_forward(x_q11.contiguous())
            # 转换为浮点数 (Q15 -> float)
            output = output_q15.float() / (1 << 15)
        else:
            # 转换为Q27格式的int32
            l_scale = 27 - int(math.log2(scale_x.data))
            if l_scale > 0:
                x_q27 = (quant_x * pow(2, l_scale)).to(torch.int64)
            else:
                x_q27 = (quant_x * pow(2, l_scale) + 0.5).floor().to(torch.int64)
            x_q27.clamp_(-2**31, 2**31-1)

            if QUANT_CONFIGS.platform == PlatForm.arcs or QUANT_CONFIGS.platform == PlatForm.mars:
                runtime_output = lingerext.arcs_qtanh_forward(x_q27.contiguous().int())
            elif QUANT_CONFIGS.platform == PlatForm.venusA:
                runtime_output = lingerext.venusa_qtanh_forward(x_q27.contiguous().int())
            else:
                raise RuntimeError(f"QTanh does not support platform {QUANT_CONFIGS.platform}")

            output_q_bits = TANH_Q31 if output_bits is None else int(output_bits) - 1
            output_q = tanh_requant_to_qbits(
                runtime_output,
                tanh_runtime_qbits(QUANT_CONFIGS.platform),
                output_q_bits,
            )
            output_dtype = torch.float64 if output_bits is not None and output_q_bits > 24 else torch.float32
            return output_q.to(output_dtype) / float(1 << output_q_bits)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        
        # 使用标准tanh的梯度近似
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = torch.tanh(x)
            grad = torch.autograd.grad(y, x, grad_output)
        
        return grad[0], None, None

# @register_qmodule(nn.Tanh)
class QTanh(QModuleTensor):
    @classmethod
    def qcreate(
        cls,
        module: torch.nn.Module,
        activate_config: Optional[Dict[str, Any]] = None,
        num_input: int = 1,
        dim: int = -1
    ):
        return cls(
            activate_config = activate_config,
            num_input = num_input
        )

    def quantize_output(self, module, input, output):
        if (
            not torch.onnx.is_in_onnx_export()
            and QUANT_CONFIGS.platform in RUNTIME_TANH_PLATFORMS
        ):
            out_scale = torch.tensor(1 << (self.output_quantizer.data_bits - 1), dtype=torch.float32, device=output.device)
            fake_output = _fixed_scale_fake_quant(output, self.output_quantizer, out_scale)
            return from_tensor_to_qtensor(fake_output, self.output_quantizer.scale, self.output_quantizer.data_bits)
        return super().quantize_output(module, input, output)
    
    def forward(self, x, *args, **kwargs):
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, True)
            return QTanhOnnxFunction.apply(x, qparam_dict)
        elif QUANT_CONFIGS.calibration:
            return torch.tanh(x)
        else:
            if isinstance(self.input_quantizer, nn.ModuleList):
                return QTanhFunction.apply(x, self.input_quantizer[0], self.output_quantizer.data_bits)
            else:
                return QTanhFunction.apply(x, self.input_quantizer, self.output_quantizer.data_bits)
