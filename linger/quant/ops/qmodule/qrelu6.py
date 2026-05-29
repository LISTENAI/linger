import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Dict, Any

from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor


@register_qmodule(nn.ReLU6)
class QRelu6(QModuleMixin, nn.ReLU6):
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
            inplace=module.inplace,
            device=device,
            activations_cfg=activations_cfg,
            weights_cfg=None,
            constrain=None,
            bias_cfg=None,
            open_ihook=False,
            open_ohook=False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, QTensor):
            tmp_input = from_qtensor_to_tensor(input)
            scale = input.scale.detach()
            data_bits = input.data_bits
            out = F.relu6(tmp_input, inplace=self.inplace)
            return from_tensor_to_qtensor(out, scale, data_bits)
        return F.relu6(input, inplace=self.inplace)
