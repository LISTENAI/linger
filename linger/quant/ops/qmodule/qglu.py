import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Union, Dict, Any

from ..qtensor import QSigmoidFunction
from ...quantizer import AQuantizer, BQuantizer
from ....config import QUANT_CONFIGS
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME

class QGLUOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, qparam_dict = None):
        return F.glu(input, dim)
        
    @staticmethod
    def symbolic(g,  input, dim, qparam_dict = None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        is_input_qtensor = qparam_dict.get("is_input_qtensor", None)
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop('op_type', None)
        qparam_dict.pop('is_input_qtensor', None)
        if is_input_qtensor is False or is_input_qtensor is None:
            op_inner = quantlinear(g, input, qparam_dict['scale_x_f'], qparam_dict['platform_s'], qparam_dict['x_bits_i'], 0)
            input_list = [op_inner]
        else:
            input_list = [input]
        return g.op(
                node_name,
                *input_list,
                **qparam_dict
            )

@register_qmodule(nn.GLU)
class QGLU(QModuleMixin, nn.GLU):
    @classmethod
    def qcreate(
        cls,
        module,
        activations_cfg: Optional[Dict[str, Any]] = None,
        weights_cfg: Optional[Dict[str, Any]] = None,
        bias_cfg: Optional[Dict[str, Any]] = None,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[Dict[str, Any]] = None,
    ):
        glu_module = cls(
            dim = module.dim,
            
            device = device,
            activations_cfg=activations_cfg,
            weights_cfg=None,
            constrain=constrain,
            bias_cfg=None,
        )

        glu_module.register_module("sigmoid_quantizer", BQuantizer(activations_cfg, None))
        return glu_module

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, False)
            return QGLUOnnxFunction.apply(input, self.dim, qparam_dict)
        if QUANT_CONFIGS.calibration:
            return F.glu(input, self.dim)
        else:
            input_a, input_b = torch.chunk(input, 2, dim=self.dim)
            input_b = QSigmoidFunction.apply(input_b, self.input_quantizer) # int32 dequant
            input_b = self.sigmoid_quantizer(input_b, torch.tensor(2**15, dtype=torch.float32))   # Q15 for thinker forward
            output = input_a * input_b
            return output

