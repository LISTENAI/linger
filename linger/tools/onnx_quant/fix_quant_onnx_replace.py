from typing import List

import onnx

from .transform.onnx_utils import remove_identity_node
from .transform.replace_add import replace_add_to_iqadd
from .transform.replace_avgpoolint import replace_avgpool2dint
from .transform.replace_clip import replace_clip_attr
from .transform.replace_conv import replace_conv2dint
from .transform.replace_iqsigmoid import replace_iqsigmoid
from .transform.replace_linearint import replace_linearint

supported_replace_ops = ['Conv', 'ConvTranspose', 'Add', 'Clip', 'AveragePool', 'MaxPool', 'Relu', 'Gemm', 'Transpose', 'Reshape', 'Squeeze', 'Unsqueeze',
                         'Split', "Sigmoid"]


def onnx_quant(model_path, quant_ops_type: List[str], remove_ops_type: List[str] = [], scale_x=16.0, scale_y=16.0, scale_w=64.0, scale_o=16.0, platform_quant="luna_quant"):
    model = onnx.load(model_path)
    for remove_op_type in remove_ops_type:
        model = remove_identity_node(model, remove_op_type)
    for quant_op_type in quant_ops_type:
        if quant_op_type == "Conv" or quant_op_type == "ConvTranspose":
            model = replace_conv2dint(
                model, scale_x, scale_w, scale_o, platform_quant)
        if quant_op_type == "Add":
            model = replace_add_to_iqadd(
                model, scale_x, scale_y, scale_o, platform_quant)
        if quant_op_type == "Clip":
            model = replace_clip_attr(model, scale_o)
        if quant_op_type == "Sigmoid":
            model = replace_iqsigmoid(model, scale_x, 256.0, platform_quant)
        if quant_op_type == "AveragePool":
            model = replace_avgpool2dint(
                model, scale_x, scale_o, platform_quant)
        if quant_op_type == "Gemm":
            model = replace_linearint(
                model, scale_x, scale_w, scale_o, platform_quant)

    # change inp/outp type to int8, ignore quant/dequant op
    for i in range(len(model.graph.input)):
        if model.graph.input[i].type.tensor_type.elem_type == 1:
            model.graph.input[i].type.tensor_type.elem_type = 3
    for i in range(len(model.graph.output)):
        if model.graph.output[i].type.tensor_type.elem_type == 1:
            if "Sigmoid" in quant_ops_type:
                model.graph.output[i].type.tensor_type.elem_type = 2
            else:
                model.graph.output[i].type.tensor_type.elem_type = 3

    onnx.save(model, model_path[:-5]+"_quant.onnx")
