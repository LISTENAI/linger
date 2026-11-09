"""Native MOM fake quantization with the original arithmetic and STE order."""

import torch

from ..utils import QuantMode


class NativeMOMFakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, quant_min, quant_max, round_mode):
        scaled = input * scale
        if round_mode == QuantMode.floor_add:
            rounded = (scaled + 0.5).floor()
        elif round_mode == QuantMode.floor:
            rounded = scaled.floor()
        else:
            rounded = scaled.round()
        # The subtraction/addition is part of the existing floating-point
        # behavior. Do not replace it with just the rounded tensor.
        quantized = (rounded - scaled).detach() + scaled
        mask = (quantized >= quant_min) & (quantized <= quant_max)
        if torch.is_tensor(scale):
            ctx.save_for_backward(mask, scale)
            ctx.scalar_scale = None
        else:
            ctx.save_for_backward(mask)
            ctx.scalar_scale = scale
        return quantized.clamp(quant_min, quant_max) / scale

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scalar_scale is None:
            mask, scale = ctx.saved_tensors
        else:
            mask, = ctx.saved_tensors
            scale = ctx.scalar_scale
        # Match DivBackward -> ClampBackward -> MulBackward, including
        # rounding with non-power-of-two residual/output scales.
        grad_input = grad_output / scale
        grad_input = torch.where(mask, grad_input, 0)
        grad_input = grad_input * scale
        return grad_input, None, None, None, None
