import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cmodule import CModuleMixin, register_cmodule
from typing import Optional, Union, Dict, Any

@register_cmodule(torch.nn.Embedding)
class CEmbedding(CModuleMixin, nn.Embedding):
    @classmethod
    def ccreate(
        cls,
        module,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[Dict[str, Any]] = None,
    ):
        return cls(
            module.num_embeddings,
            module.embedding_dim,
            module.padding_idx,
            module.max_norm,
            module.norm_type,
            module.scale_grad_by_freq,
            module.sparse,
            None, # _weight参数永远设置为None
            None, # _freeze参数永远设置为None
            dtype = module.weight.dtype,
            device=device,
            constrain=constrain,
            open_ihook = False
        )

    def forward(self, input):
        return F.embedding(
                    input,
                    self.cweight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )

