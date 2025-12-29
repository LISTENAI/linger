import torch
from typing import Dict, Any, Optional
from .qtensor_mod import QModuleTensor
from ....onnx import generate_onnx_qparam_dict, QDOMAIN_NAME

class QAddOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, qparam_dict = None):
        return x + y
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

class QAdd(QModuleTensor):
    r"""对QAdd的layer封装

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
            num_input = num_input
        )

    def forward(self, x, y):
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, True)
            return QAddOnnxFunction.apply(x, y, qparam_dict)
        elif self.training:
            self.output_quantizer.min_scale = torch.min(self.input_quantizer[0].scale, self.input_quantizer[1].scale)
        else:
            if self.output_quantizer.scale != self.input_quantizer[0].scale:
                int_x = self.input_quantizer[0].quant_round(x * self.output_quantizer.scale, self.input_quantizer[0].round_mode)
                x = int_x / self.output_quantizer.scale
            if self.output_quantizer.scale != self.input_quantizer[1].scale:
                int_y = self.input_quantizer[1].quant_round(y * self.output_quantizer.scale, self.input_quantizer[1].round_mode)
                y = int_y / self.output_quantizer.scale
        return x + y
        
