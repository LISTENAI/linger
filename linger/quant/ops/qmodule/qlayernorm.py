import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from ...quantizer import WQuantizer, AQuantizer, BQuantizer
from ....config import QUANT_CONFIGS
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME

import lingerext

class QLayerNormOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps, qparam_dict=None):
        return F.layer_norm(input, normalized_shape, weight, bias, eps)
        
    @staticmethod
    def symbolic(g,  input, normalized_shape, weight, bias, eps, qparam_dict=None):
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


class QLayerNormFunction(torch.autograd.Function):
    """QLayerNormFunction
    input---quant_data
    weight---quant_data
    bias---float_data
    """
    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps,
                input_quantizer, weight_quantizer, bias_quantizer, training):
        
        # === 1 reshape input to (M, N)
        N = 1
        for d in normalized_shape:
            N *= d
        M = input.numel() // N

        scale_x = input_quantizer.scale
        scale_w = weight_quantizer.scale

        q_input = input_quantizer.quant_round(input * scale_x, input_quantizer.round_mode)
        q_input = q_input.clamp(input_quantizer.quant_min, input_quantizer.quant_max)
        q_weight = weight_quantizer.quant_round(weight * scale_w, weight_quantizer.round_mode)
        q_weight = q_weight.clamp(weight_quantizer.quant_min, weight_quantizer.quant_max)
        x_2d = q_input.contiguous().view(M, N).long().long()

        # === 2 调用底层算子
        sum_x = x_2d.clone().sum(-1, keepdim=True)
        sum_x2 = x_2d.clone().pow(2).sum(-1, keepdim=True)
        denominator = N * sum_x2 - sum_x * sum_x
        scale_eps = 2 * scale_x.log2()
        q_eps = math.floor(eps * pow(2, scale_eps) * N * N + 0.5)
        # q_eps = input_quantizer.quant_round(eps * pow(2, scale_eps) * N * N, input_quantizer.round_mode)
        # q_eps = (torch.clamp(q_eps, input_quantizer.quant_min, input_quantizer.quant_max)).long()
        denominator = denominator + q_eps
        numerator = N * x_2d
        numerator = numerator - sum_x
        q_y_normal = lingerext.qlayernorm_kernel_forward(numerator.int(), denominator.long(), math.log2(scale_x.data))
        scale_y_normal = 2**10

        # === 3 reshape back to original
        q_y_normal = q_y_normal.view(input.shape)
        q_y_normal.clamp_(-2**15, 2**15-1)
        q_output = q_y_normal * q_weight
        if bias is not None:
            q_bias = bias_quantizer.quant_round(bias * scale_y_normal * scale_w, bias_quantizer.round_mode)
            q_bias = torch.clamp(q_bias, bias_quantizer.quant_min, bias_quantizer.quant_max)
        else:
            q_bias = None
        
        q_output = q_output + q_bias
        q_output.clamp_(-2**31, 2**31-1)
        outputs = q_output.float() / (scale_y_normal * scale_w)

        saved_tensors = []
        if training:
            ctx.normalized_shape = normalized_shape
            ctx.eps = eps
            ctx.scale_x = scale_x
            ctx.scale_w = scale_w
            saved_tensors += [input, weight, bias]
            ctx.save_for_backward(*saved_tensors)

        return outputs
    
    @staticmethod
    def backward(ctx, gradOutput):
        input, weight, bias = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        input = input.detach().requires_grad_(True)
        weight = weight.detach().requires_grad_(True)
        bias = bias.detach().requires_grad_(True)

        with torch.enable_grad():
            z = F.layer_norm(input, normalized_shape, weight, bias, eps)
            grads = torch.autograd.grad(
                z, (input, weight, bias) if bias is not None else (input, weight),
                gradOutput,
                retain_graph=False,
                allow_unused=True
            )

        gradInput = grads[0]
        gradWeight = grads[1]
        gradBias = grads[2] if bias is not None else None

        return gradInput, gradWeight, gradBias, None, None, None, None, None, None

@register_qmodule(torch.nn.LayerNorm)
class QLayerNorm(QModuleMixin, nn.LayerNorm):
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
            normalized_shape = module.normalized_shape,
            eps = module.eps,
            elementwise_affine = module.elementwise_affine,
            # bias = module.bias is not None,
            device = device,
            dtype = module.weight.dtype,
            activations_cfg = activations_cfg,
            weights_cfg = weights_cfg,
            bias_cfg = bias_cfg,
            constrain = constrain
        )

    def forward(self, input):
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, False)
            return QLayerNormOnnxFunction.apply(input, self.normalized_shape, self.qweight, self.qbias, self.eps, qparam_dict)
        if QUANT_CONFIGS.calibration:
            return F.layer_norm(input, self.normalized_shape, self.qweight, self.qbias, self.eps)
        else:
            return QLayerNormFunction.apply(
                input, self.qweight, self.bias, self.normalized_shape, self.eps,
                self.input_quantizer, self.weight_quantizer, self.bias_quantizer, self.training
            )

