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

class QSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.dim = dim
        ctx.save_for_backward(input)
        
        x = input.contiguous()
        # 处理负维度
        dim = dim if dim >= 0 else dim + input.dim()

        # ---- Step 1: reshape input -> 2D tensor [outer, size] ----
        ndim = x.dim()
        size = x.size(dim)
        outer = int(x.numel() / size)
        permute_order = list(range(ndim))
        # move softmax dim to last
        if dim != ndim - 1:
            permute_order[dim], permute_order[-1] = permute_order[-1], permute_order[dim]
            x = x.permute(permute_order)
        x_2d = x.reshape(outer, size)
        
        # 转换为Q25格式的int32
        x_q25 = (x_2d * (1 << 25) + 0.5).floor().to(torch.int32)
        x_q25.clamp_(-2**31, 2**31-1)

        output_q15 = None
        if QUANT_CONFIGS.platform == PlatForm.venus:
            output_q15 = lingerext.arcs_qsoftmax_forward(x_q25, dim)
        elif QUANT_CONFIGS.platform == PlatForm.arcs or QUANT_CONFIGS.platform == PlatForm.mars:
            output_q15 = lingerext.arcs_qsoftmax_forward(x_q25, dim)
        elif QUANT_CONFIGS.platform == PlatForm.venusA:
            output_q15 = lingerext.venusa_qsoftmax_forward(x_q25, dim)
        
        # 转换为浮点数 (Q15 -> float)
        y = output_q15.float() / (1 << 15)

        # ---- reshape back to original shape ----
        output = y.reshape_as(x)
        if dim != ndim - 1:
            # inverse permutation
            inv_perm = [0] * ndim
            for i, p in enumerate(permute_order):
                inv_perm[p] = i
            output = output.permute(inv_perm).contiguous()
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        dim = ctx.dim
        
        # 使用标准softmax的梯度近似
        x = x.detach().clone().requires_grad_(True)
        grad = None
        with torch.enable_grad():
            y = F.softmax(x, dim=dim)
            grad = torch.autograd.grad(y, x, grad_output)
        
        return grad[0], None

@register_qmodule(nn.Softmax)
class QSoftmax(QModuleTensor):
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
            num_input = 2,
            dim = dim
        )
    
    def qforward(self, x, *args, **kwargs):
        if QUANT_CONFIGS.calibration:
            return F.softmax(x, self.dim)
        else:
            return QSoftmaxFunction.apply(x, self.dim)

