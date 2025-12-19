import torch
from torch.onnx import is_in_onnx_export
from typing import Dict, Any, Optional

from .qtensor_mod import QModuleTensor
from ..qconfig import register_qmodule
from ...qtensor import QTensor
from ...quantizer import AQuantizer
from ....config import *

# def qbmm(module, x, y, name="_default"):
#     assert isinstance(x, QTensor)
#     assert isinstance(y, QTensor)

#     quant_info = getattr(module, LINGER_QUANTINFO, QuantInfo())

#     var_name = name
#     iq_layer = None
#     if hasattr(module, var_name):
#         iq_layer = getattr(module, var_name)
#     else:
#         iq_layer = QBmm(quant_info=quant_info)
#         iq_layer.training = module.training
#         iq_layer = iq_layer.to(x.device)
#         setattr(module, var_name, iq_layer)

#     return iq_layer(x, y)

# @register_qmodule(torch.bmm)
class QBmm(QModuleTensor):
    # def __init__(self, activate_config: Optional[Dict[str, Any]] = None, num_input: int = 2):
    #     super(QModuleTensor, self).__init__()

    #     self.prefix         = ""
    #     self.dump           = False
    #     self.path           = ""
    #     self.a_config       = activate_config

    @classmethod
    def qcreate(
        cls,
        module: torch.nn.Module,
        activate_config: Optional[Dict[str, Any]] = None,
        num_input: int = 2,
        dim: int = -1
    ):
        return cls(
            activate_config = activate_config,
            num_input = num_input
        )

    def forward(self, x, y):
        return torch.bmm(x, y)

        
