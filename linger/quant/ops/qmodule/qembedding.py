import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Union, Dict, Any
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME

class QEmbeddingOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, qparam_dict= None):
        return F.embedding(
                    input,
                    weight,
                    padding_idx,
                    max_norm,
                    norm_type,
                    scale_grad_by_freq,
                    sparse,
                )
        
    @staticmethod
    def symbolic(g,  input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, qparam_dict= None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        is_input_qtensor = qparam_dict.get("is_input_qtensor", None)
        node_name = f"{QDOMAIN_NAME}::Gather"
        qparam_dict.pop('op_type', None)
        qparam_dict.pop('is_input_qtensor', None)
        if is_input_qtensor is False or is_input_qtensor is None:
            op_inner = quantlinear(g, input, qparam_dict['scale_x_f'], qparam_dict['platform_s'], qparam_dict['x_bits_i'], 0)
            input_list = [op_inner, weight]
        else:
            input_list = [input, weight]
        return g.op(
                node_name,
                *input_list,
                **qparam_dict
            )

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

    def forward(self, input):
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, False)
            return QEmbeddingOnnxFunction.apply(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse, qparam_dict)
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

