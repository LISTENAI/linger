import torch
from typing import Dict, Any, Optional
from .qtensor_mod import QModuleTensor
from ....onnx import generate_onnx_qparam_dict, quant_qtensor_symbolic_input, QDOMAIN_NAME

class QVarOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, unbiased, keepdim, qparam_dict=None):
        return torch.var(x, dim, unbiased, keepdim)

    @staticmethod
    def symbolic(g, x, dim, unbiased, keepdim, qparam_dict=None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop('op_type', None)
        qparam_dict['dims_i'] = dim
        qparam_dict['unbiased_i'] = unbiased
        qparam_dict['keepdim_i'] = keepdim
        input_list = [
            quant_qtensor_symbolic_input(g, x, qparam_dict, 0),
        ]
        return g.op(
            node_name,
            *input_list,
            **qparam_dict
        )

class QVar(QModuleTensor):
    r"""对QVar的layer封装

    """
    @classmethod
    def qcreate(
        cls,
        module: torch.nn.Module,
        activate_config: Optional[Dict[str, Any]] = None,
        num_input: int = 1,
        dim: int = -1,
        unbiased: bool = True,
        keepdim: bool = True
    ):
        return cls(
            activate_config=activate_config,
            num_input=num_input,
            dim=dim,
            unbiased=unbiased,
            keepdim=keepdim
        )

    def __init__(self, activate_config=None, num_input=1, dim=-1, unbiased=True, keepdim=False):
        super().__init__(activate_config=activate_config, num_input=num_input)
        self.dim = dim
        self.unbiased = unbiased
        self.keepdim = keepdim

    def forward(self, x, dim=-1, unbiased=True, keepdim=True):
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, True)
            qparam_dict['dims_i'] = dim
            qparam_dict['unbiased_i'] = unbiased
            qparam_dict['keepdim_i'] = keepdim
            return QVarOnnxFunction.apply(x, dim, unbiased, keepdim, qparam_dict)
        elif self.training:
            self.output_quantizer.min_scale = self.input_quantizer[0].scale
        else:
            return torch.var(x, dim, unbiased, keepdim)