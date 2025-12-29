import torch
from typing import Dict, Any, Optional
from .qtensor_mod import QModuleTensor
from ....onnx import generate_onnx_qparam_dict, QDOMAIN_NAME

class QCatOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, qparam_dict = None):
        return torch.cat(input, dim=dim)
    @staticmethod
    def symbolic(g, input, dim, qparam_dict = None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop('op_type', None)
        input_list = input
        qparam_dict['axis_i'] = int(dim)
        return g.op(
                node_name,
                *input_list,
                **qparam_dict
            )

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

    def forward(self, input, dim):
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, True)
            return QCatOnnxFunction.apply(input, dim, qparam_dict)
        return torch.cat(input, dim=dim)
        
