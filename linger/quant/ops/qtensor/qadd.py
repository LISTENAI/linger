import torch
from typing import Dict, Any, Optional
from .qtensor_mod import QModuleTensor
from ....config import QUANT_CONFIGS
from ....utils import PlatForm
# def add(module, x, y, name="_default"):
#     assert isinstance(x, QTensor)
#     assert isinstance(y, (QTensor, float, int))

#     quant_info = getattr(module, LINGER_QUANTINFO, QuantInfo())

#     var_name = name
#     iq_layer = None
#     if hasattr(module, var_name):
#         iq_layer = getattr(module, var_name)
#     else:
#         iq_layer = QAdd(quant_info=quant_info)
#         iq_layer.training = module.training
#         iq_layer = iq_layer.to(x.device)
#         setattr(module, var_name, iq_layer)

#     return iq_layer(x, y)

# @register_qmodule(torch.add)
# @register_qmodule(operator.add)
# @register_qmodule(torch.ops.aten.add.Tensor)
class QAdd(QModuleTensor):
    r"""对iqadd的layer封装

    """
    # def __init__(self, activate_config: Optional[Dict[str, Any]] = None, num_input: int = 2):
    #     super(QAdd, self).__init__(activate_config, num_input)

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
        return x + y
        
