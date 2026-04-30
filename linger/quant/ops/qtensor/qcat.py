import torch
from typing import Dict, Any, Optional
from .qtensor_mod import QModuleTensor
from ....onnx import QDOMAIN_NAME, quant_qtensor_symbolic_input
from ....config import QUANT_CONFIGS

class QCatOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dim, qparam_dict, *args):
        tensors = args
        return torch.cat(tensors, dim=dim)
    @staticmethod
    def symbolic(g, dim, qparam_dict, *args):
        input_list = []
        node_name = f"{QDOMAIN_NAME}::QCat"
        param_dict = {'axis_i': dim}
        input_count = qparam_dict.get("input_count", len(args))

        for i, tensor in enumerate(args[:input_count]):
            scale_key = f"scale_x_{i}_f"
            bits_key = f"x_{i}_bits_i"
            scale_x = float(qparam_dict[scale_key])
            data_bits = int(qparam_dict[bits_key])

            input_list.append(quant_qtensor_symbolic_input(g, tensor, qparam_dict, i))
            param_dict[scale_key] = scale_x
            param_dict[bits_key] = data_bits

        param_dict['platform_s'] = str(qparam_dict['platform_s'])
        param_dict['quant_mode_s'] = str(qparam_dict['quant_mode_s'])
        param_dict['o_bits_i'] = int(qparam_dict['o_bits_i'])
        param_dict['scale_o_f'] = float(qparam_dict['scale_o_f'])

        return g.op(
                node_name,
                *input_list,
                **param_dict
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
        param_list = []
        if torch.onnx.is_in_onnx_export():
            qparam_dict = {
                "input_count": len(input),
                "platform_s": str(QUANT_CONFIGS.platform.name),
                "quant_mode_s": str(self.output_quantizer.round_mode.name),
                "o_bits_i": int(self.output_quantizer.data_bits),
                "scale_o_f": float(self.output_quantizer.scale),
            }
            for s in input:
                param_list.append(s)
            for i, _ in enumerate(input):
                qparam_dict[f"scale_x_{i}_f"] = float(self.input_quantizer[i].scale)
                qparam_dict[f"x_{i}_bits_i"] = int(self.input_quantizer[i].data_bits)
                qparam_dict[f"is_x_{i}_qtensor_i"] = int(getattr(self.input_quantizer[i], "is_qtensor", False))
            return QCatOnnxFunction.apply(dim, qparam_dict, *param_list)
        return torch.cat(input, dim=dim)
        
