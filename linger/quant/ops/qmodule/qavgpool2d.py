import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor
from typing import Optional, Union, Dict, Any

@register_qmodule(nn.AvgPool2d)
class QAvgPool2d(QModuleMixin, nn.AvgPool2d):
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
            kernel_size = module.kernel_size,
            stride = module.stride,
            padding = module.padding,
            ceil_mode = module.ceil_mode,
            count_include_pad = module.count_include_pad,
            divisor_override = module.divisor_override,

            device = device,
            activations_cfg=activations_cfg,
            weights_cfg=None,
            constrain=None, 
            bias_cfg=None,
            open_ihook = True,
            open_ohook = True,
        )

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )
        

