import torch
import torch.nn
import torch.onnx
import numpy as np
import onnx
from onnx import numpy_helper
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

def quant_weight_bias(f_data, data_bits, scale):
    min_val = -(2**(data_bits - 1))
    max_val = (2**(data_bits - 1)) - 1

    x_scaled = np.round(f_data * scale)
    x_clipped = np.clip(x_scaled, min_val, max_val)

    if data_bits <= 8:
        q_val = x_clipped.astype(np.int8)
    elif data_bits <= 16:
        q_val = x_clipped.astype(np.int16)
    else:
        q_val = x_clipped.astype(np.int32)
    return q_val

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
        if hasattr(cls.input_quantizer, 'is_qtensor'):
            qparam_dict['is_input_qtensor'] = cls.input_quantizer.is_qtensor
    if hasattr(cls, 'output_quantizer'):
        qparam_dict['o_bits_i'] = int(cls.output_quantizer.data_bits)
        qparam_dict['scale_o_f'] = float(cls.output_quantizer.scale)
    if hasattr(cls, 'weight_quantizer') and cls.weight_quantizer is not None:
        qparam_dict['w_bits_i'] = int(cls.weight_quantizer.data_bits)
        qparam_dict['scale_w_f'] = float(cls.weight_quantizer.scale)
    qparam_dict['op_type'] = cls._get_name()

    if 'Cat' in qparam_dict['op_type']:
        qparam_dict['x_0_bits_i'] = qparam_dict['x_bits_i']
        qparam_dict['x_1_bits_i'] = qparam_dict['y_bits_i']
        qparam_dict['scale_x_0_f'] = qparam_dict['scale_x_f']
        qparam_dict['scale_x_1_f'] = qparam_dict['scale_y_f']
        qparam_dict.pop('x_bits_i', None)
        qparam_dict.pop('y_bits_i', None)
        qparam_dict.pop('scale_x_f', None)
        qparam_dict.pop('scale_y_f', None)
    elif 'Softmax' in qparam_dict['op_type'] or 'GLU' in qparam_dict['op_type']:
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
        qparam_dict['outputs'] = 3 if 'LSTM' in qparam_dict['op_type'] else 2
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
            qparam_dict_r['outputs'] = qparam_dict['outputs']
            qparam_dict['qparam_dict_r'] = qparam_dict_r
    return qparam_dict

def quantlinear(g, input, scale_x, platform, data_bits, zero_point):
    return g.op("linger::Quant", input, data_bits_i=data_bits, scale_x_f = scale_x, platform_s = platform, zero_point_i = zero_point)

__all__ = ['export']
