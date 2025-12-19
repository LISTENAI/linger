import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .qtensor_mod import QModuleTensor
from ..qconfig import register_qmodule
from ....config import QUANT_CONFIGS
from ....utils import PlatForm

import lingerext

class QTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, input_quantizer):
        ctx.save_for_backward(input)
        
        x = input.contiguous()
        
        scale_x = input_quantizer.scale
        quant_x = input_quantizer.quant_round(x * scale_x, input_quantizer.round_mode)
        
        # 转换为Q27格式的int32
        l_scale = 27 - int(math.log2(scale_x.data))
        if l_scale > 0:
            x_q27 = (quant_x * pow(2, l_scale)).to(torch.int32)
        else:
            x_q27 = (quant_x * pow(2, l_scale) + 0.5).floor().to(torch.int32)
        x_q27.clamp_(-2**31, 2**31-1)

        output_q31 = None
        if QUANT_CONFIGS.platform == PlatForm.venus:
            output_q31 = lingerext.venusa_qtanh_forward(x_q27.contiguous())
        elif QUANT_CONFIGS.platform == PlatForm.arcs or QUANT_CONFIGS.platform == PlatForm.mars:
            output_q31 = lingerext.arcs_qtanh_forward(x_q27.contiguous())
        elif QUANT_CONFIGS.platform == PlatForm.venusA:
            output_q31 = lingerext.venusa_qtanh_forward(x_q27.contiguous())
        
        # 转换为浮点数 (Q31 -> float)
        output = output_q31.float() / (1 << 31)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        
        # 使用标准tanh的梯度近似
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = F.tanh(x)
            grad = torch.autograd.grad(y, x, grad_output)
        
        return grad, None

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
    
    def qforward(self, x, *args, **kwargs):
        if QUANT_CONFIGS.calibration:
            return torch.tanh(x)
        else:
            if isinstance(self.input_quantizer, nn.ModuleList):
                return QTanhFunction.apply(x, self.input_quantizer[0])
            else:
                return QTanhFunction.apply(x, self.input_quantizer)

