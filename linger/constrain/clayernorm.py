import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cmodule import CModuleMixin, register_cmodule
from typing import Optional, Union, Dict, Any

@register_cmodule(torch.nn.LayerNorm)
class CLayerNorm(CModuleMixin, nn.LayerNorm):
    @classmethod
    def ccreate(
        cls,
        module,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[Dict[str, Any]] = None,
    ):
        return cls(
            module.normalized_shape,
            module.eps,
            module.elementwise_affine,
            None if module.bias is None else True,
            dtype=module.weight.dtype,
            device=device,
            constrain=constrain,
        )

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.cweight, self.cbias, self.eps
        )

