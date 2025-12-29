import torch
from typing import Dict, Any, Optional
from .qtensor_mod import QModuleTensor
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor
from ....onnx import generate_onnx_qparam_dict, QDOMAIN_NAME

class QSqueezeOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, input, *args):
        return torch.squeeze(input, *args)

    @staticmethod
    def symbolic(g, input, *args):
        input_list = [input]
        qparam_dict = {}
        node_name = f"{QDOMAIN_NAME}::Squeeze"
        qparam_dict['axes_i'] = args
        return g.op(
                node_name,
                *input_list,
                **qparam_dict
            )
