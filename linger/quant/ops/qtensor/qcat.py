import torch
from typing import Dict, Any, Optional
from .qtensor_mod import QModuleTensor
from ....onnx import generate_onnx_qparam_dict, QDOMAIN_NAME
from ....config import QUANT_CONFIGS

class QCatOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dim, output_quantizer, *args):
        tensors = args[0:len(args)//2]
        return torch.cat(tensors, dim=dim)
    @staticmethod
    def symbolic(g, dim, output_quantizer, *args):
        tensors = args[0:len(args)//2]
        input_list = tensors
        node_name = f"{QDOMAIN_NAME}::QCat"
        param_dict = {'axis_i': dim}
        for i, value in enumerate(args[len(args)//2:]):
            param_dict['scale_x_'+str(i)+"_f"] = value
            param_dict['x_'+str(i)+"_bits_i"] = 8

        param_dict['platform_s'] = str(QUANT_CONFIGS.platform.name)
        param_dict['quant_mode_s'] = str(output_quantizer.round_mode.name)

        param_dict['o_bits_i'] = int(output_quantizer.data_bits)
        param_dict['scale_o_f'] = float(output_quantizer.scale)

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
        param_list =[]
        if torch.onnx.is_in_onnx_export():
            for s in input:
                param_list.append(s)
            for i, s in enumerate(input):
                param_list.append(float(self.input_quantizer[i].scale))
            # qparam_dict = generate_onnx_qparam_dict(self, True)
            return QCatOnnxFunction.apply(dim, self.output_quantizer, *param_list)
        return torch.cat(input, dim=dim)
        
