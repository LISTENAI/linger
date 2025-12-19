import torch
from typing import Dict, Any, Optional
from .qtensor_mod import QModuleTensor

class QMatmul(QModuleTensor):
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
       return torch.matmul(x, y)