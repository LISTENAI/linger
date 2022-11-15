from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..quant import NormalizeFunction


class NormalizeEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None,  normalize_data=None, normalize_weight=None) -> None:

        assert normalize_data is None or normalize_data > 0, 'normalize value is None or must >0'
        assert normalize_weight is None or normalize_weight > 0, 'normalize value is None or must >0'
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                              max_norm, norm_type, scale_grad_by_freq, sparse, _weight)
        self.normalize_data = normalize_data
        self.normalize_weight = normalize_weight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_weight = self.weight
        if self.normalize_weight is not None:
            normalized_weight = NormalizeFunction.apply(
                normalized_weight, self.normalize_weight, self.training)

        out = None
        out = F.embedding(input, normalized_weight, self.padding_idx, self.max_norm,
                        self.norm_type, self.scale_grad_by_freq, self.sparse)

        if self.normalize_data is not None:
            out = NormalizeFunction.apply(out, self.normalize_data, self.training, False)
        return out

    def extra_repr(self):
        s = nn.Embedding.extra_repr(self)
        extra_s = ',normalize_data:{normalize_data},normalize_weight:{normalize_weight}}'.format(
            **self.__dict__)
        return s+extra_s


__all__ = ['NormalizeEmbedding']
