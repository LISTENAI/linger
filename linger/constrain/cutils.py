import math
import torch

def static_clip(input, clip_data, training=True, is_weight=True):
    return torch.clamp(input, min = -clip_data, max = clip_data)

def dyn_clip_weight(weight, factor):
    with torch.no_grad():
        if factor is None:
            factor = 3
        clamp_data = factor * weight.abs().mean()
        abs_max = weight.abs().max()
        clamp_data = torch.min(clamp_data, abs_max)
    return torch.clamp(weight, min=-clamp_data, max=clamp_data)


__all__ = ['static_clip', 'dyn_clip_weight']
