import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cmodule import CModuleMixin, register_cmodule
from typing import Optional, Union, Dict, Any

@register_cmodule(torch.nn.Linear)
class CLinear(CModuleMixin, nn.Linear):
    @classmethod
    def ccreate(
        cls,
        module,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[Dict[str, Any]] = None,
    ):
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype=module.weight.dtype,
            device=device,
            constrain=constrain,
        )

    def forward(self, input):
        return F.linear(input, self.cweight, bias=self.cbias)

