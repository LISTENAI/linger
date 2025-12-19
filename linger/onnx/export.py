import torch
import torch.nn
import torch.onnx
import onnx
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from torch.onnx import symbolic_helper as sym_help
from onnx import shape_inference, helper
from io import BytesIO
from typing import Any, Callable, Collection, Mapping, Sequence, TYPE_CHECKING

from ..config import QUANT_CONFIGS
from ..utils import _single, _pair, _triple

from .update_dequant import parser_dequant

torch_onnx_export = torch.onnx.export

QDOMAIN_NAME = 'linger'

import os
import numpy as np
from onnx import numpy_helper
def quant_weight_bias(f_data, data_bits, scale):
    if data_bits <= 8:
        q_val = np.round(f_data * scale).astype(np.int8)
    elif data_bits <= 16:
        q_val = np.round(f_data * scale).astype(np.int16)
    else:
        q_val = np.round(f_data * scale).astype(np.int32)
    return q_val

def change_onnx_to_linger2_0(onnx_path):
    onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    # 构建 initializer 查找表
    initializer_map = {init.name: init for init in graph.initializer}

    for node in graph.node:
        if 'QGRU' in node.op_type or 'QLSTM' in node.op_type:
            scale_i = None
            scale_h = None
            scale_iw = None
            scale_hw = None
            for attr in node.attribute:
                if attr.name == 'scale_x':
                    scale_i = attr.f
                    attr.name = 'scale_i'
                elif attr.name == 'scale_h':
                    scale_h = attr.f
                elif attr.name == 'scale_iw':
                    scale_iw = attr.f
                elif attr.name == 'scale_hw':
                    scale_hw = attr.f
                elif attr.name == 'x_bits':
                    attr.name = 'data_bits'
                elif attr.name == 'w_bits':
                    attr.name = 'parameter_bits'
            for i in range(1, 3):
                if i == 1:
                    scale_w = scale_iw
                else:
                    scale_w = scale_hw
                weight_name = node.input[i]
                weight_f = numpy_helper.to_array(initializer_map[weight_name])
                weight_q = quant_weight_bias(weight_f, 8, scale_w)
                weight_init_new = numpy_helper.from_array(weight_q, weight_name)
                graph.initializer.remove(initializer_map[weight_name])
                graph.initializer.append(weight_init_new)
            if len(node.input) > 3:
                for i in range(3, 5):
                    if i == 3:
                        scale_b = scale_i * scale_iw
                    else:
                        scale_b = scale_h * scale_hw
                    bias_name = node.input[i]
                    bias_f = numpy_helper.to_array(initializer_map[bias_name])
                    bias_q = quant_weight_bias(bias_f, 32, scale_b)
                    bias_init_new = numpy_helper.from_array(bias_q, bias_name)
                    graph.initializer.remove(initializer_map[bias_name])
                    graph.initializer.append(bias_init_new)

            if 'QGRU' in node.op_type:
                node.op_type = 'GRUInt'
                node.name = node.name + '_GRUInt'
            elif 'QLSTM' in node.op_type:
                node.op_type = 'LSTMInt'
                node.name = node.name + '_LSTMInt'
            continue

        scale_x = None
        scale_w = None
        for attr in node.attribute:
            if attr.name == 'scale_x':
                scale_x = attr.f
            elif attr.name == 'scale_w':
                scale_w = attr.f
            elif attr.name == 'x_bits':
                attr.name = 'data_bits'
            elif attr.name == 'w_bits':
                attr.name = 'parameter_bits'

        if len(node.input) > 1 and 'weight' in str(node.input[1]):
            weight_name = node.input[1]
            weight_f = numpy_helper.to_array(initializer_map[weight_name])
            weight_q = quant_weight_bias(weight_f, 8, scale_w)
            weight_init_new = numpy_helper.from_array(weight_q, weight_name)

            graph.initializer.remove(initializer_map[weight_name])
            graph.initializer.append(weight_init_new)

        if len(node.input) > 2 and 'bias' in str(node.input[2]):
            bias_name = node.input[2]
            bias_f = numpy_helper.to_array(initializer_map[bias_name])
            bias_q = quant_weight_bias(bias_f, 32, scale_x * scale_w)
            bias_init_new = numpy_helper.from_array(bias_q, bias_name)
            graph.initializer.remove(initializer_map[bias_name])
            graph.initializer.append(bias_init_new)

        if 'QConv2d' in node.op_type or 'QConvBN2d' in node.op_type:
            node.op_type = 'Conv2dInt'
            node.name = node.name + '_Conv2dInt'
        elif 'QConv1d' in node.op_type or 'QConvBN1d' in node.op_type:
            node.op_type = 'Conv1dInt'
            node.name = node.name + '_Conv1dInt'
        elif 'QAvgPool1d' in node.op_type:
            node.op_type = 'AvgPool1dInt'
            node.name = node.name + '_AvgPool1dInt'
        elif 'QAvgPool2d' in node.op_type:
            node.op_type = 'AvgPool2dInt'
            node.name = node.name + '_AvgPool2dInt'
        elif 'QConvTranspose1d' in node.op_type:
            node.op_type = 'ConvTranspose1dInt'
            node.name = node.name + '_ConvTranspose1dInt'
        elif 'QConvTranspose2d' in node.op_type:
            node.op_type = 'ConvTranspose2dInt'
            node.name = node.name + '_ConvTranspose2dInt'
        elif 'QLinear' in node.op_type:
            node.op_type = 'LinearInt'
            node.name = node.name + '_LinearInt'
        elif 'QEmbedding' in node.op_type:
            node.op_type = 'Gather'
            node.name = node.name + '_Gather'
        elif 'QLayerNorm' in node.op_type:
            node.op_type = 'LayerNormInt'
            node.name = node.name + '_LayerNormInt'
        elif 'QAdd' in node.op_type:
            node.op_type = 'iqAdd'
            node.name = node.name + '_iqAdd'
        elif 'QMul' in node.op_type:
            node.op_type = 'iqMul'
            node.name = node.name + '_iqMul'
        elif 'QBmm' in node.op_type:
            node.op_type = 'BmmInt'
            node.name = node.name + '_BmmInt'
        elif 'QCat' in node.op_type:
            node.op_type = 'iqCat'
            node.name = node.name + '_iqCat'
            for attr in node.attribute:
                if attr.name == 'axis':
                    attr.name = 'dim'
        elif 'QSigmoid' in node.op_type:
            node.op_type = 'iqSigmoid'
            node.name = node.name + '_iqSigmoid'
        elif 'QTanh' in node.op_type:
            node.op_type = 'iqTanh'
            node.name = node.name + '_iqTanh'
        elif 'QSoftmax' in node.op_type:
            node.op_type = 'SoftmaxInt'
            node.name = node.name + '_SoftmaxInt'
        elif 'QGLU' in node.op_type:
            node.op_type = 'GluInt'
            node.name = node.name + '_GluInt'
            for attr in node.attribute:
                if attr.name == 'axis':
                    attr.name = 'dim'

    # 保存修改后的模型
    dir_name = os.path.dirname(onnx_path)
    # 获取文件名和扩展名：aaa, .onnx
    base_name = os.path.basename(onnx_path)
    file_name, ext = os.path.splitext(base_name)
    # 生成新的文件名：aaa_2.0.onnx
    new_file_name = f"{file_name}_2.0{ext}"
    # 拼接成完整路径
    output_path = os.path.join(dir_name, new_file_name)
    onnx.save(onnx_model, output_path)
    print(f"转换完成：{output_path}")

def convert_parameter_from_float_to_int(onnx_model):
    # onnx_model = onnx.load(onnx_path)
    graph = onnx_model.graph
    # 构建 initializer 查找表
    initializer_map = {init.name: init for init in graph.initializer}

    for node in graph.node:
        if 'QGRU' in node.op_type or 'QLSTM' in node.op_type:
            scale_i = None
            scale_h = None
            scale_iw = None
            scale_hw = None
            for attr in node.attribute:
                if attr.name == 'scale_x':
                    scale_i = attr.f
                elif attr.name == 'scale_h':
                    scale_h = attr.f
                elif attr.name == 'scale_iw':
                    scale_iw = attr.f
                elif attr.name == 'scale_hw':
                    scale_hw = attr.f
            for i in range(1, 3):
                if i == 1:
                    scale_w = scale_iw
                else:
                    scale_w = scale_hw
                weight_name = node.input[i]
                weight_f = numpy_helper.to_array(initializer_map[weight_name])
                weight_q = quant_weight_bias(weight_f, 8, scale_w)
                weight_init_new = numpy_helper.from_array(weight_q, weight_name)
                graph.initializer.remove(initializer_map[weight_name])
                graph.initializer.append(weight_init_new)
            if len(node.input) > 3:
                for i in range(3, 5):
                    if i == 3:
                        scale_b = scale_i * scale_iw
                    else:
                        scale_b = scale_h * scale_hw
                    bias_name = node.input[i]
                    bias_f = numpy_helper.to_array(initializer_map[bias_name])
                    bias_q = quant_weight_bias(bias_f, 32, scale_b)
                    bias_init_new = numpy_helper.from_array(bias_q, bias_name)
                    graph.initializer.remove(initializer_map[bias_name])
                    graph.initializer.append(bias_init_new)
            continue

        scale_x = None
        scale_w = None
        for attr in node.attribute:
            if attr.name == 'scale_x':
                scale_x = attr.f
            elif attr.name == 'scale_w':
                scale_w = attr.f

        if len(node.input) > 1 and 'weight' in str(node.input[1]):
            weight_name = node.input[1]
            weight_f = numpy_helper.to_array(initializer_map[weight_name])
            weight_q = quant_weight_bias(weight_f, 8, scale_w)
            weight_init_new = numpy_helper.from_array(weight_q, weight_name)

            graph.initializer.remove(initializer_map[weight_name])
            graph.initializer.append(weight_init_new)

        if len(node.input) > 2 and 'bias' in str(node.input[2]):
            bias_name = node.input[2]
            bias_f = numpy_helper.to_array(initializer_map[bias_name])
            bias_q = quant_weight_bias(bias_f, 32, scale_x * scale_w)
            bias_init_new = numpy_helper.from_array(bias_q, bias_name)
            graph.initializer.remove(initializer_map[bias_name])
            graph.initializer.append(bias_init_new)

    return onnx_model

    # 保存修改后的模型
    # dir_name = os.path.dirname(onnx_path)
    # # 获取文件名和扩展名：aaa, .onnx
    # base_name = os.path.basename(onnx_path)
    # file_name, ext = os.path.splitext(base_name)
    # # 生成新的文件名：aaa_2.0.onnx
    # new_file_name = f"{file_name}_2.0{ext}"
    # # 拼接成完整路径
    # output_path = os.path.join(dir_name, new_file_name)
    # onnx.save(onnx_model, onnx_path)
    # print(f"转换完成：{onnx_path}")

def export(model, args, f, **kwargs):

    # 1. 正常导出 ONNX
    tmp = BytesIO()
    torch_onnx_export(model, args, tmp, **kwargs)
    tmp.seek(0)

    # 2. 自动加载并执行 shape inference
    onnx_model = onnx.load(tmp)

    onnx_model = convert_parameter_from_float_to_int(onnx_model)

    onnx_model = parser_dequant(onnx_model, False)

    onnx.save(onnx_model, f)

    # inferred_model = shape_inference.infer_shapes(onnx_model)

    # 3. 覆盖保存，让最终导出的文件带有 shape
    # onnx.save(inferred_model, f)

def generate_onnx_qparam_dict(cls, input_list = False):
    qparam_dict = {'platform_s': str(QUANT_CONFIGS.platform.name), 'quant_mode_s': str(cls.output_quantizer.round_mode.name)}
    if input_list:
        qparam_dict['x_bits_i'] = int(cls.input_quantizer[0].data_bits)
        qparam_dict['scale_x_f'] = float(cls.input_quantizer[0].scale)
        if len(cls.input_quantizer) > 1:
            qparam_dict['y_bits_i'] = int(cls.input_quantizer[1].data_bits)
            qparam_dict['scale_y_f'] = float(cls.input_quantizer[1].scale)
    elif hasattr(cls, "input_quantizer"):
        qparam_dict['x_bits_i'] = int(cls.input_quantizer.data_bits)
        qparam_dict['scale_x_f'] = float(cls.input_quantizer.scale)
    if hasattr(cls, 'output_quantizer'):
        qparam_dict['o_bits_i'] = int(cls.output_quantizer.data_bits)
        qparam_dict['scale_o_f'] = float(cls.output_quantizer.scale)
    if hasattr(cls, 'weight_quantizer') and cls.weight_quantizer is not None:
        # qparam_dict['weight'] = cls.weight
        qparam_dict['w_bits_i'] = int(cls.weight_quantizer.data_bits)
        qparam_dict['scale_w_f'] = float(cls.weight_quantizer.scale)
    # if hasattr(cls, 'weight_quantizer') and cls.weight_quantizer is not None:
    #     qparam_dict['bias'] = cls.bias
    qparam_dict['op_type'] = cls._get_name()

    if 'Softmax' in qparam_dict['op_type'] or 'GLU' in qparam_dict['op_type']:
        qparam_dict['axis_i'] = int(cls.dim)
        qparam_dict.pop('y_bits_i', None)
        qparam_dict.pop('scale_y_f', None)
    elif 'ConvTranspose' in qparam_dict['op_type']:
        qparam_dict['dilations_i'] = cls.dilation
        qparam_dict['kernel_shape_i'] = cls.kernel_size
        qparam_dict['pads_i'] = cls.padding * 2
        qparam_dict['strides_i'] = cls.stride
        qparam_dict['group_i'] = cls.groups
        qparam_dict['output_padding_i'] = cls.output_padding
    elif 'Conv' in qparam_dict['op_type']:
        qparam_dict['dilations_i'] = cls.dilation
        qparam_dict['kernel_shape_i'] = cls.kernel_size
        qparam_dict['pads_i'] = cls.padding * 2
        qparam_dict['strides_i'] = cls.stride
        qparam_dict['group_i'] = cls.groups
    elif 'AvgPool' in qparam_dict['op_type']:
        tuple_fn = _pair # for AvgPool2D
        qparam_dict['kernel_shape_i'] = tuple_fn(cls.kernel_size)
        qparam_dict['pads_i'] = tuple_fn(cls.padding) * 2
        qparam_dict['strides_i'] = tuple_fn(cls.stride)
        qparam_dict['ceil_mode_i'] = cls.ceil_mode
    elif 'GRU' in qparam_dict['op_type'] or 'LSTM' in qparam_dict['op_type']:
        qparam_dict['input_size_i'] = int(cls.input_size)
        qparam_dict['hidden_size_i'] = int(cls.hidden_size)
        qparam_dict['num_layers_i'] = int(cls.num_layers)
        qparam_dict['batch_first_i'] = int(cls.batch_first)
        qparam_dict['go_forward_i'] = True
        qparam_dict['scale_h_f'] = float(cls.hidden_quantizer.scale)
        qparam_dict['w_bits_i'] = int(cls.weightih_quantizer.data_bits)
        qparam_dict['scale_iw_f'] = float(cls.weightih_quantizer.scale)
        qparam_dict['scale_hw_f'] = float(cls.weighthh_quantizer.scale)
        qparam_dict['outputs'] = 2
        if cls.bidirectional:
            qparam_dict_r = {'platform_s': str(QUANT_CONFIGS.platform.name), 'quant_mode_s': str(cls.output_quantizer.round_mode.name)}
            qparam_dict_r['x_bits_i'] = int(cls.input_quantizer.data_bits)
            qparam_dict_r['scale_x_f'] = float(cls.input_quantizer.scale)
            qparam_dict_r['input_size_i'] = int(cls.input_size)
            qparam_dict_r['hidden_size_i'] = int(cls.hidden_size)
            qparam_dict_r['num_layers_i'] = int(cls.num_layers)
            qparam_dict_r['batch_first_i'] = int(cls.batch_first)
            qparam_dict_r['go_forward_i'] = False
            qparam_dict_r['scale_h_f'] = float(cls.hidden_reverse_quantizer.scale)
            qparam_dict_r['w_bits_i'] = int(cls.weightih_reverse_quantizer.data_bits)
            qparam_dict_r['scale_iw_f'] = float(cls.weightih_reverse_quantizer.scale)
            qparam_dict_r['scale_hw_f'] = float(cls.weighthh_reverse_quantizer.scale)
            qparam_dict_r['o_bits_i'] = int(cls.output_reverse_quantizer.data_bits)
            qparam_dict_r['scale_o_f'] = float(cls.output_reverse_quantizer.scale)
            qparam_dict_r['outputs'] = 2
            qparam_dict['qparam_dict_r'] = qparam_dict_r
    return qparam_dict

def quantlinear(g, input, scale_x, platform, data_bits, zero_point):
    return g.op("linger::Quant", input, data_bits_i=data_bits, scale_x_f = scale_x, platform_s = platform, zero_point_i = zero_point)

class QCustomOpSymbolic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, *args):
        # 占位符
        return input.clone()

    @staticmethod
    def symbolic(g, input, weight, bias, *args):
        # 获取算子类型
        qparams = {} if args[0] is None else args[0]
        
        op_type = qparams.get("op_type", "QGeneric")
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparams.pop('op_type', None)

        input_list = []
        if 'Linear' in op_type or 'Conv' in op_type or 'ConvTranspose' in op_type or 'BatchNorm' in op_type:
            is_input_qtensor = args[1] if len(args) > 1 else None
            if not is_input_qtensor:
                op_inner = quantlinear(g, input, qparams['scale_x_f'], qparams['platform_s'], qparams['x_bits_i'], 0)
                input_list = [op_inner, weight]
            else:
                input_list = [input, weight]
            # input_list = [input, weight]
            if bias is not None:
                input_list.append(bias)
        elif 'AvgPool' in op_type or 'Sigmoid' in op_type or 'Tanh' in op_type or 'Softmax' in op_type or 'GLU' in op_type:
            input_list = [input]
        elif 'Add' in op_type or 'Mul' in op_type or 'Matmul' in op_type or 'Bmm' in op_type:
            other = args[1]
            input_list = [input, other]
        elif 'Cat' in op_type:
            other = args[1]
            axis = args[2]
            input_list = [input, other]
            qparams['axis_i'] = int(axis)
            qparams['x_0_bits_i'] = qparams['x_bits_i']
            qparams['x_1_bits_i'] = qparams['y_bits_i']
            qparams['scale_x_0_f'] = qparams['scale_x_f']
            qparams['scale_x_1_f'] = qparams['scale_y_f']
            qparams.pop('x_bits_i', None)
            qparams.pop('y_bits_i', None)
            qparams.pop('scale_x_f', None)
            qparams.pop('scale_y_f', None)
        elif 'Embedding' in op_type:
            node_name = f"{QDOMAIN_NAME}::Gather"
        else:
            out = g.op("quant_domain::IdentityQ", input)
        
            # # shape 推导
            # input_shape = sym_help._get_tensor_sizes(input)
            # weight_shape = sym_help._get_tensor_sizes(weight)
            # if input_shape and weight_shape:
            #     out_shape = input_shape[:-1] + [weight_shape[0]]
            #     out_shape[0] = None
            #     out.setType(input.type().with_sizes(out_shape))
            # else:
            #     out.setType(input.type())

        out = g.op(
                node_name,
                *input_list,
                **qparams
            )
        return out

class QCustomRNNSymbolic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_i, weight_h, bias_i, bias_h, weight_i_r, weight_h_r, bias_i_r, bias_h_r, *args):
        # 占位符
        return input.clone(), input.clone()

    @staticmethod
    def symbolic(g, input, weight_i, weight_h, bias_i, bias_h, weight_i_r, weight_h_r, bias_i_r, bias_h_r, *args):
        # 获取算子类型
        qparams = {} if args[0] is None else args[0]
        
        op_type = qparams.get("op_type", "QGeneric")
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparams.pop('op_type', None)

        input_list = []
        if 'GRU' in op_type or 'LSTM' in op_type:
            qparam_dict_r = qparams.get("qparam_dict_r", None)
            qparams.pop('qparam_dict_r', None)

            is_input_qtensor = args[1] if len(args) > 1 else None
            if not is_input_qtensor:
                op_inner = quantlinear(g, input, qparams['scale_x_f'], qparams['platform_s'], qparams['x_bits_i'], 0)
                input_list = [op_inner, weight_i, weight_h]
            else:
                input_list = [input, weight_i, weight_h]

            # input_list = [input, weight_i, weight_h]
            if bias_i is not None:
                input_list.append(bias_i)
                input_list.append(bias_h)
            # To do: insert length and hidden

            out, hidden = g.op(node_name, *input_list, **qparams)
            if qparam_dict_r is not None:   # 双向RNN
                if not is_input_qtensor:
                    op_inner = quantlinear(g, input, qparam_dict_r['scale_x_f'], qparam_dict_r['platform_s'], qparam_dict_r['x_bits_i'], 0)
                    input_list_r = [op_inner, weight_i_r, weight_h_r]
                else:
                    input_list_r = [input, weight_i_r, weight_h_r]
                # input_list_r = [input, weight_i_r, weight_h_r]
                if bias_i_r is not None:
                    input_list_r.append(bias_i_r)
                    input_list_r.append(bias_h_r)
                out_r, hidden_r = g.op(node_name, *input_list_r, **qparam_dict_r)

                cat_node_name = f"{QDOMAIN_NAME}::QCat"
                cat_input_list = [out, out_r]
                cat_param = {}
                cat_param['platform_s'] = qparams.get("platform_s", None)
                cat_param['quant_mode_s'] = qparams.get("quant_mode_s", None)
                cat_param['axis_i'] = int(2)
                cat_param['x_0_bits_i'] = qparams.get("o_bits_i", 8)
                cat_param['scale_x_0_f'] = qparams.get("scale_o_f", 1.0)
                cat_param['x_1_bits_i'] = qparam_dict_r.get("o_bits_i", 8)
                cat_param['scale_x_1_f'] = qparam_dict_r.get("scale_o_f", 1.0)
                cat_param['o_bits_i'] = qparams.get("o_bits_i", 8)
                cat_param['scale_o_f'] = min(qparam_dict_r.get("scale_o_f", 1.0), qparams.get("scale_o_f", 1.0))
                out = g.op(cat_node_name, *cat_input_list, **cat_param)
                hidden = g.op("Concat", hidden, hidden_r, axis_i=0)
            return out, hidden
        else:
            return g.op("quant_domain::IdentityQ", input)


__all__ = ['export', 'QCustomOpSymbolic', 'QCustomRNNSymbolic', 'change_onnx_to_linger2_0']
