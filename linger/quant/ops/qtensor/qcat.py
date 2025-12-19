import torch
from typing import Dict, Any, Optional
from .qtensor_mod import QModuleTensor

class QCat(QModuleTensor):
    r"""对cat的layer封装

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
            activate_config = activate_config,
            num_input = num_input,
            is_cat = True
        )

    def qforward(self, x, y):
        return torch.cat(x, dim=y)
        
