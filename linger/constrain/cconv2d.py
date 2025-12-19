import torch
from .cmodule import CModuleMixin, register_cmodule
from typing import Optional, Union, Dict, Any

@register_cmodule(torch.nn.Conv2d)
class CConv2d(CModuleMixin, torch.nn.Conv2d):
    @classmethod
    def ccreate(
        cls,
        module,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[Dict[str, Any]] = None,
    ):
        return cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            dtype=module.weight.dtype,
            device=device,
            constrain=constrain,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.cweight, self.cbias)

