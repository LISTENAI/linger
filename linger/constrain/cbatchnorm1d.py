import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cmodule import CModuleMixin, register_cmodule
from typing import Optional, Union, Dict, Any

from .cutils import static_clip, dyn_clip_weight

@register_cmodule(torch.nn.BatchNorm1d)
class CBatchNorm1d(CModuleMixin, nn.BatchNorm1d):
    @classmethod
    def ccreate(
        cls,
        module,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[Dict[str, Any]] = None,
    ):
        return cls(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
            dtype=module.weight.dtype,
            device=device,
            constrain=constrain,
        )

    def forward(self, input):
        # cweight = self.cweight
        # cbias = self.cbias
        # BatchNorm1d input shape: (N, C) or (N, C, L)
        if input.dim() == 2:
            # Input shape: (N, C)
            batchsize, channels = input.shape
            size = batchsize
            sum_dims = 0
        else:
            # Input shape: (N, C, L)
            batchsize, channels, length = input.shape
            size = batchsize * length
            sum_dims = (0, 2)

        if self.training:
            mean = input.sum(sum_dims, keepdim=True) / size
            var = input.pow(2).sum(sum_dims, keepdim=True) / size - \
                (input.sum(sum_dims, keepdim=True) / size).pow(2)
            var = torch.clamp(var, min=0.0)
            self.running_mean = (
                1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze().detach()
            self.running_var = (1 - self.momentum) * self.running_var + \
                self.momentum * var.squeeze().detach()
        else:
            if input.dim() == 2:
                mean = self.running_mean.reshape(1, -1)
                var = self.running_var.reshape(1, -1)
            else:
                mean = self.running_mean.reshape(1, -1, 1)
                var = self.running_var.reshape(1, -1, 1)
        sigma = 1 / torch.sqrt(var + self.eps)

        if input.dim() == 2:
            alpha = self.weight.view(1, -1) * sigma
            beta = self.bias.view(1, -1) - mean * alpha
        else:
            alpha = self.weight.view(1, -1, 1) * sigma
            beta = self.bias.view(1, -1, 1) - mean * alpha

        if self.clamp_weight is not None:
            alpha = static_clip(alpha, self.clamp_weight)
        else:
            alpha = dyn_clip_weight(alpha, self.clamp_factor)

        if self.clamp_bias is not None:
            beta = static_clip(beta, self.clamp_bias)

        out = alpha * input + beta
        return out

