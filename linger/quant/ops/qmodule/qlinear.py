import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Union, Dict, Any

@register_qmodule(torch.nn.Linear)
class QLinear(QModuleMixin, nn.Linear):
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
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype=module.weight.dtype,
            device=device,
            activations_cfg=activations_cfg,
            weights_cfg=weights_cfg,
            bias_cfg=bias_cfg if module.bias is not None else None,
            constrain = constrain, 
        )

    def qforward(self, input):
        return F.linear(input, self.qweight, bias=self.qbias)

