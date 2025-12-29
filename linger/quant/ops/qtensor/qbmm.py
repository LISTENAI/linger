import torch
from typing import Dict, Any, Optional
from .qtensor_mod import QModuleTensor
from ....onnx import generate_onnx_qparam_dict, QDOMAIN_NAME

class QBmmOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, qparam_dict = None):
        return torch.bmm(x, y)
    @staticmethod
    def symbolic(g, x, y, qparam_dict = None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop('op_type', None)
        input_list = [x, y]
        return g.op(
                node_name,
                *input_list,
                **qparam_dict
            )

class QBmm(QModuleTensor):
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
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, True)
            return QBmmOnnxFunction.apply(x, y, qparam_dict)
        return torch.bmm(x, y)

        
