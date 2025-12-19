import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Union, Dict, Any

from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor

@register_qmodule(nn.ReLU)
class QRelu(QModuleMixin, nn.ReLU):
    @classmethod
    def qcreate(
        cls,
        module,
        activations_cfg: Optional[Dict[str, Any]] = None,
        weights_cfg: Optional[Dict[str, Any]] = None,
        bias_cfg: Optional[Dict[str, Any]] = None,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[Dict[str, Any]] = None,
    ):
        return cls(
            inplace = module.inplace,
            
            device = device,
            activations_cfg=activations_cfg,
            weights_cfg=None,
            constrain=None,
            bias_cfg=None,
            open_ihook = False,
            open_ohook = False,
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, QTensor):
            tmp_input = from_qtensor_to_tensor(input)
            scale = input.scale.detach()
            data_bits = input.data_bits
            out = F.relu(tmp_input, inplace=self.inplace)
            return from_tensor_to_qtensor(out, scale, data_bits)
        else:
            return F.relu(input, inplace=self.inplace)

