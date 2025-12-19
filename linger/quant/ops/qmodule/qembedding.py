import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Union, Dict, Any
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor

@register_qmodule(torch.nn.Embedding)
class QEmbedding(QModuleMixin, nn.Embedding):
    @classmethod
    def qcreate(
        cls,
        module,
        activations_cfg: Optional[Dict[str, Any]] = None,
        weights_cfg: Optional[Dict[str, Any]] = None,
        bias_cfg: Optional[Dict[str, Any]] = None,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
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
            device = device,
            activations_cfg = activations_cfg,
            weights_cfg = weights_cfg,
            bias_cfg = bias_cfg,
            constrain = constrain,
            open_ihook = False,
            open_ohook = False
        )

    def qforward(self, input):
        out_q =  F.embedding(
                    input,
                    self.qweight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
        return from_tensor_to_qtensor(out_q, self.weight_quantizer.scale, self.weight_quantizer.data_bits)

