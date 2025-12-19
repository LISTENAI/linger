import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from .cmodule import CModuleMixin, register_cmodule
from typing import Optional, Union, Dict, Any

@register_cmodule(torch.nn.ConvTranspose1d)
class QConvTranspose1d(CModuleMixin, torch.nn.ConvTranspose1d):
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
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            output_padding=module.output_padding,
            groups=module.groups,
            bias=module.bias is not None,
            dilation=module.dilation,
            padding_mode=module.padding_mode,
            dtype=module.weight.dtype,
            device=device,
            activations_cfg=activations_cfg,
            weights_cfg=weights_cfg,
            bias_cfg=bias_cfg,
            constrain = constrain, 
        )

    def qforward(self, input: torch.Tensor, output_size: Optional[List[int]] = None) -> torch.Tensor:
        return F.conv_transpose1d(
            input, 
            self.qweight, 
            self.qbias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,)

