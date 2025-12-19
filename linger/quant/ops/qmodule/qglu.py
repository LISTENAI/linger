import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Union, Dict, Any

from ..qtensor import QSigmoidFunction
from ...quantizer import AQuantizer, BQuantizer

from ....config import QUANT_CONFIGS

@register_qmodule(nn.GLU)
class QGLU(QModuleMixin, nn.GLU):
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
        glu_module = cls(
            dim = module.dim,
            
            device = device,
            activations_cfg=activations_cfg,
            weights_cfg=None,
            constrain=constrain,
            bias_cfg=None,
        )

        glu_module.register_module("sigmoid_quantizer", BQuantizer(activations_cfg, None))
        return glu_module

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
        if QUANT_CONFIGS.calibration:
            return F.glu(input, self.dim)
        else:
            input_a, input_b = torch.chunk(input, 2, dim=self.dim)
            input_b = QSigmoidFunction.apply(input_b, self.input_quantizer) # int32 dequant
            input_b = self.sigmoid_quantizer(input_b, torch.tensor(2**15, dtype=torch.float32))   # Q15 for thinker forward
            output = input_a * input_b
            return output

