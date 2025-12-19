import torch
from typing import Dict, Any, Optional
from .qtensor_mod import QModuleTensor

class QMul(QModuleTensor):
    r"""量化乘法算子封装

    """
    @classmethod
    def qcreate(
        cls,
        module: torch.nn.Module,
        activate_config: Optional[Dict[str, Any]] = None,
        num_input: int = 2,
        dim: int = -1
    ):
        return cls(
            activate_config=activate_config,
            num_input=num_input
        )
    
    def qforward(self, x, y):
        if self.training:
            self.output_quantizer.min_scale = torch.min(self.input_quantizer[0].scale, self.input_quantizer[1].scale)
        else:
            if self.output_quantizer.scale != self.input_quantizer[0].scale:
                int_x = self.input_quantizer[0].quant_round(x * self.output_quantizer.scale, self.input_quantizer[0].round_mode)
                x = int_x / self.output_quantizer.scale
            if self.output_quantizer.scale != self.input_quantizer[1].scale:
                int_y = self.input_quantizer[1].quant_round(y * self.output_quantizer.scale, self.input_quantizer[1].round_mode)
                y = int_y / self.output_quantizer.scale
        return x * y
