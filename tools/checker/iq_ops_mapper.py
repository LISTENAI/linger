import torch.nn.functional as F
import torch
from linger.quant.ops import *
from linger.checker.utils import get_param,register_op
from linger.config import QUANT_CONFIGS
import numpy as np
import linger
from linger.utils import quant, dequant
from .utils import create_qmodule, create_qmodule_tensor, load_quantized_weights, StringToQuantMode, canonicalize_attrs
from .utils import create_rnn_module, load_rnn_quantized_weights

@register_op(op_type=['QAvgPool2d', 'AvgPool2dInt'])
def avgpool2dint(inputs, kwargs):
    input = inputs[0]
    kwargs = canonicalize_attrs(kwargs)

    kernel_size = tuple(kwargs['kernel_shape'])
    stride = tuple(kwargs['strides'])
    padding = tuple(kwargs['pads'][0:2])
    ceil_mode = bool(kwargs['ceil_mode'])
    device = input.device

    module = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode).to(device)

    instance = create_qmodule(QAvgPool2d, module, device, kwargs)

    return instance(input)

@register_op(op_type=['QConv1d', 'Conv1dInt'])
def conv1dInt(inputs, kwargs):
    inputs_len = len(inputs)
    assert inputs_len == 2 or inputs_len == 3, \
    f"Conv2dInt: invalid number of input tensors (expected 2 or 3, got {inputs_len})"
    if inputs_len == 2:
        input, weights= inputs
        bias = None
    else:
        input, weights, bias = inputs
    kwargs = canonicalize_attrs(kwargs)

    in_channels = input.shape[1]
    out_channels = weights.shape[0]
    kernel_shape = tuple(kwargs.get('kernel_shape', None))
    strides = kwargs.get('strides', 1)
    padding = kwargs.get('pads', (0, 0))[0]
    dilations = kwargs.get('dilations', 1)
    group = kwargs.get('group', 1)
    device = input.device

    module = nn.Conv1d(in_channels, out_channels, kernel_shape, strides, padding, dilations, group, bias=bias!=None).to(device)

    instance = create_qmodule(QConv1d, module, device, kwargs)
    if weights.dtype != torch.float32:
        instance = load_quantized_weights(instance, kwargs, weights, bias)
    
    res =  instance(input)
    if kwargs.get('act_type', 0) == 1:
        res = F.relu(res)
    return res

@register_op(op_type=['QConv2d', 'QConvBN2d', 'Conv2dInt'])
def conv2dint(inputs, kwargs):
    inputs_len = len(inputs)
    assert inputs_len == 2 or inputs_len == 3, \
    f"Conv2dInt: invalid number of input tensors (expected 2 or 3, got {inputs_len})"
    if inputs_len == 2:
        input, weights= inputs
        bias = None
    else:
        input, weights, bias = inputs
    kwargs = canonicalize_attrs(kwargs)

    in_channels = input.shape[1]
    out_channels = weights.shape[0]
    kernel_shape = tuple(kwargs.get('kernel_shape', None))
    strides = tuple(kwargs.get('strides', (1, 1)))
    pads = kwargs.get('pads', None)
    if pads is not None:
        pads = tuple(pads[0:2])
    else:
        pads = (0, 0)
    dilations = tuple(kwargs.get('dilations', (1, 1)))
    group = kwargs.get('group', 1)
    device = input.device

    module = torch.nn.Conv2d(in_channels, out_channels, kernel_shape, strides, pads, dilations, group, bias=bias!=None).to(device)
    
    instance = create_qmodule(QConv2d, module, device, kwargs)
    if weights.dtype != torch.float32:
        instance = load_quantized_weights(instance, kwargs, weights, bias)
    
    res = instance(input)

    if kwargs.get('act_type', 0) == 1:
        res = F.relu(res)
    return res

@register_op(op_type=['QLinear', 'LinearInt'])
def linearint(inputs, kwargs):
    inputs_len = len(inputs)
    assert inputs_len == 2 or inputs_len == 3, \
    f"LinearInt ops: the number of inputs is wrong, \
    expect 2 or 3, but {inputs_len}, the length of List must be 2([input,weight]) or 3\
    ([input,weight,bias])"
    if inputs_len == 2:
        input, weights= inputs
        bias = None
    else:
        input, weights, bias = inputs
    kwargs = canonicalize_attrs(kwargs)

    out_features, in_features = weights.shape
    has_bias = inputs_len == 3
    device = input.device

    module = nn.Linear(in_features, out_features, has_bias).to(device)

    instance = create_qmodule(QLinear, module, device, kwargs)
    if weights.dtype != torch.float32:
        instance = load_quantized_weights(instance, kwargs, weights, bias)

    return instance(input)

@register_op(op_type=['QCat', 'iqCat'])
def iqcat(inputs, kwargs):
    kwargs['is_cat'] = True
    dim = kwargs.get('dim', -1)
    num_input = len(inputs)

    if inputs[0].dtype in {torch.int32, torch.int64}:
        return torch.cat(inputs, dim)
    else:
        instance = create_qmodule_tensor(QCat, None, num_input, kwargs)
        return instance(inputs, dim)

@register_op(op_type='Quant')
def quant_(inputs, kwargs):
    bits = kwargs.get('data_bits', 8)
    scale = torch.tensor(kwargs.get('scale_x', 1.0), dtype=torch.float32)
    zp = torch.tensor(kwargs.get('zero_point', 0), dtype=torch.float32)
    quant_mode = StringToQuantMode(kwargs.get('quant_mode', 'floor_add'))
    input = inputs[0]
    qinput, _ = quant(input, bits, scale, zp, quant_mode)
    input = dequant(qinput, scale)
    input = from_tensor_to_qtensor(input, scale, bits, zp)
    return input

@register_op(op_type='Dequant')
def dequant_(inputs, kwargs):
    input = inputs[0]
    return input

@register_op(op_type=['QBmm', 'BmmInt'])
def bmmint(inputs, kwargs):
    num_input = len(inputs)
    assert num_input == 2, f'invalid input number, expeted 2, but got {num_input}'
    input_0, input_1 = inputs
    kwargs = canonicalize_attrs(kwargs)

    instance = create_qmodule_tensor(QBmm, None, 2, kwargs)
    return instance(input_0, input_1)

@register_op(op_type=['QLayerNorm', 'LayerNormInt'])
def layernormint(inputs, kwargs):
    inputs_len = len(inputs)
    assert inputs_len == 2 or inputs_len == 3, \
    f"LayerNormInt ops: the number of inputs is wrong, \
    expect 2 or 3, but {inputs_len}, the length of List must be 2([input,weight]) or 3\
    ([input,weight,bias])"
    if inputs_len == 2:
        input, weights= inputs
        bias = None
    else:
        input, weights, bias = inputs
    kwargs = canonicalize_attrs(kwargs)

    axis = kwargs.get('axis', -1)
    normalized_shape = list(input.shape)[axis:]
    device = input.device

    module = nn.LayerNorm(normalized_shape, device=device)

    instance = create_qmodule(QLayerNorm, module, device, kwargs)
    if weights.dtype != torch.float32:
        instance = load_quantized_weights(instance, kwargs, weights, bias)

    return instance(input)

@register_op(op_type=['QGLU', 'GluInt'])
def gluint(inputs, kwargs):
    input = inputs[0]
    kwargs = canonicalize_attrs(kwargs)

    dim = kwargs.get('dim', -1)
    module = nn.GLU(dim=dim)
    device = input.device

    instance = create_qmodule(QGLU, module, device, kwargs)
    return instance(input)

@register_op(op_type='MaxPool')
def maxpool2d(inputs, kwargs):
    input = inputs[0]

    kernel_size = tuple(kwargs.get('kernel_shape', None))
    pads = kwargs.get('pads', None)
    if pads is not None:
        pads = tuple(pads[0:2])
    else:
        pads = (0, 0)
    strides = get_param(kwargs,'strides')
    dilations = tuple(kwargs.get('dilations', (1, 1)))
    ceil_mode = bool(kwargs.get('ceil_mode',False))
    device = input.device

    module = nn.MaxPool2d(kernel_size, strides, pads, dilations, False, ceil_mode)

    instance = create_qmodule(QMaxPool2d, module, device, kwargs)

    return instance(input)

@register_op(op_type=['QAdd', 'iqAdd'])
def iqadd(inputs, kwargs):
    input_len = len(inputs)
    assert input_len == 2, 'The inputs number of iqAdd is wrong'
    input_0, input_1 = inputs
    kwargs = canonicalize_attrs(kwargs)

    instance = create_qmodule_tensor(QAdd, None, 2, kwargs)
    return instance(input_0, input_1)

# @register_op(op_type='iqDiv')
# def iqdiv(inputs, kwargs):
#     platform = kwargs.get("platform", "")
#     op_cls = get_op_class(platform, "iqDiv")
#     return op_cls.excute_base(inputs, kwargs)

@register_op(op_type=['QMul', 'iqMul'])
def iqmul(inputs, kwargs):
    input_len = len(inputs)
    assert input_len == 2, 'The inputs number of iqMul is wrong'
    x, y = inputs
    kwargs = canonicalize_attrs(kwargs)

    scale_x = kwargs.get('scale_x')
    scale_y = kwargs.get('scale_y')

    if isinstance(x, QTensor) and isinstance(y, QTensor):
        qx, qy = x, y
    elif isinstance(x, QTensor) and (not isinstance(y, QTensor)):
        qx = dequant(x, scale_x)
        qy = y
    elif (not isinstance(x, QTensor)) and isinstance(y, QTensor):
        qx = x
        qy = dequant(y, scale_y)
    else:
        qx = dequant(x, scale_x)
        qy = dequant(y, scale_y)
        
    instance = create_qmodule_tensor(QMul, None, 2, kwargs)
    return instance(qx, qy)

@register_op(op_type='Relu')
def relu(inputs, kwargs):
    input = inputs[0]

    module = nn.ReLU()
    device = input.device

    instance = create_qmodule(QRelu, module, device, kwargs)
    return instance(input)

@register_op(op_type='Split')
def split(inputs, kwargs):
    axis = get_param(kwargs,'axis')
    split = get_param(kwargs,'split')
    return inputs[0].split(split, axis)

@register_op(op_type='Cast')
def cast(inputs, kwargs):

    onnx_dtype = {
        0: 'UNDEFINED',    1: 'float32',    2: 'uint8',    3: 'int8',    4: 'uint16',
        5: 'int16',    6: 'int32',    7: 'int64',    8: 'str',    9: 'bool',    10:'float16',
        11:'double',    12:'uint32',    13:'uint64',    14:'complex64',    15:'complex128',
        16:'bfloat16'
    }

    onnx_numpy_type={
        1:np.float32, 2:np.uint8, 3:np.int8, 4:np.uint16,
        5:np.int16, 6:np.int32, 7:np.int64,  9:np.bool8, 10:np.float16,
        11:np.double ,12:np.uint32, 13:np.uint64, 14:np.complex64, 15:np.complex128,
    }

    onnx_tensor_type={
        1:torch.float, 2:torch.uint8, 3:torch.int8,
        5:torch.int16, 6:torch.int32 ,7:torch.int64, 9:torch.bool,10:torch.float16,
        11:torch.double, 14:torch.complex64, 15:torch.complex128
    }
    to = get_param(kwargs, 'to')
    output = None
    if isinstance(inputs[0], QTensor) or isinstance(inputs[0], torch.Tensor):
        if to in onnx_tensor_type:
            output = inputs[0].type(onnx_tensor_type[to])
        else:
            raise TypeError("Type Error!!Current Version don't support {}(type:{}) in cast node!!!".format(to,onnx_dtype(to)))
    else:
        if to in onnx_numpy_type:
            output = np.array(inputs[0]).astype(onnx_numpy_type[to])
        else:
            raise TypeError("Type Error!!Current Version don't support {}(type:{}) in cast node!!!".format(to,onnx_dtype(to)))
    return output

@register_op(op_type= ['QConvTranspose2d', 'ConvTranspose2dInt'])
def convTranspose2dInt(inputs, kwargs):
    inputs_len = len(inputs)
    assert inputs_len == 2 or inputs_len == 3, \
    f"Conv2dInt: invalid number of input tensors (expected 2 or 3, got {inputs_len})"
    if inputs_len == 2:
        input, weights= inputs
        bias = None
    else:
        input, weights, bias = inputs
    kwargs = canonicalize_attrs(kwargs)

    in_channels = input.shape[1]
    out_channels = weights.shape[0]
    kernel_shape = tuple(kwargs.get('kernel_shape', None))
    strides = tuple(kwargs.get('strides', (1, 1)))
    pads = kwargs.get('pads', None)
    if pads is not None:
        pads = tuple(pads[0:2])
    else:
        pads = (0, 0)
    dilations = tuple(kwargs.get('dilations', (1, 1)))
    group = kwargs.get('group', 1)
    device = input.device

    module = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_shape, strides, pads, dilations, group).to(device)
    
    instance = create_qmodule(QConvTranspose2d, module, device, kwargs)
    if weights.dtype != torch.float32:
        instance = load_quantized_weights(instance, kwargs, weights, bias)
    
    res = instance(input)

    if kwargs.get('act_type', 0) == 1:
        res = F.relu(res)
    return res

# @register_op(op_type="iqSum")
# def iqsum(inputs, kwargs):
#     platform = kwargs.get("platform", "")
#     op_cls = get_op_class(platform, "iqSum")
#     return op_cls.excute_base(inputs, kwargs)

@register_op(op_type=['QMatmul', 'MatMulInt'])
def matmulint(inputs, kwargs):
    num_input = len(inputs)
    assert num_input == 2, f'invalid input number, expeted 2, but got {num_input}'
    input_0, input_1 = inputs
    kwargs = canonicalize_attrs(kwargs)

    instance = create_qmodule_tensor(QMatmul, None, 2, kwargs)
    return instance(input_0, input_1)

@register_op(op_type=['QBatchNorm2d', 'BatchNorm2dInt'])
def batchnorm2dInt(inputs, kwargs):
    input, weights, bias = inputs
    kwargs = canonicalize_attrs(kwargs)

    num_features = input.shape[1]
    device = input.device

    module = nn.BatchNorm2d(num_features)

    instance = create_qmodule(QBatchNorm2d, module, device, kwargs)
    if weights.dtype != torch.float32:
        instance = load_quantized_weights(instance, kwargs, weights, bias)

    return instance(input)

@register_op(op_type="GRUInt")
def gruint(inputs, kwargs):
    inputs_len = len(inputs)
    assert inputs_len == 5, f'GRUInt/QGRU input numbuder {inputs_len} is invalid.'
    input, weight_ih, weight_hh, bias_ih, bias_hh = inputs
    
    device = input.device
    batch_first = kwargs.get('batch_first', 1)
    batch_first = True if batch_first else False
    kwargs = canonicalize_attrs(kwargs)

    module = nn.GRU(input_size=kwargs.get("input_size", None), hidden_size = kwargs.get("hidden_size", None),
                     num_layers=kwargs.get("num_layers", None), bias=True, batch_first=batch_first,
                     dropout=0, bidirectional=False)
    
    instance = create_rnn_module(QGRU, module, device, kwargs)
    instance = load_rnn_quantized_weights(instance, kwargs, weight_ih, weight_hh, bias_ih, bias_hh)

    if kwargs['go_forward'] == 1:  
        return instance(input)
    else:
        reversed_input = torch.flip(input, dims=[1])
        out = instance(reversed_input)
        out_0 = torch.flip(out[0], dims=[1])
        return tuple([out_0, out[1]])
    

@register_op(op_type=['QLSTM', "LSTMInt"])
def lstmint(inputs, kwargs):
    inputs_len = len(inputs)
    seq_lens, h0, c0 = None, None, None
    if inputs_len == 5:
        input, weight_ih, weight_hh, bias_ih, bias_hh = inputs
    elif inputs_len == 6:
        input, seq_lens, weight_ih, weight_hh, bias_ih, bias_hh = inputs
    elif inputs_len == 7:
        input, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh = inputs
    elif inputs_len == 8:
        input, seq_lens, h0, c0, weight_ih, weight_hh, bias_ih, bias_hh = inputs

    device = input.device
    batch_first = kwargs.get('batch_first', 1)
    batch_first = True if batch_first else False
    kwargs = canonicalize_attrs(kwargs)

    module = nn.LSTM(input_size=kwargs.get("input_size", None), hidden_size = kwargs.get("hidden_size", None),
                     num_layers=kwargs.get("num_layers", None), bias=True, batch_first=batch_first,
                     dropout=0, bidirectional=False, proj_size=kwargs.get('proj_size', 0))
    
    instance = create_rnn_module(QLSTM, module, device, kwargs)
    instance = load_rnn_quantized_weights(instance, kwargs, weight_ih, weight_hh, bias_ih, bias_hh)

    if kwargs['go_forward'] == 1:  
        return instance(input)
    else:
        reversed_input = torch.flip(input, dims=[1])
        out = instance(reversed_input)
        out_0 = torch.flip(out[0], dims=[1])
        return tuple([out_0, out[1]])
    

@register_op(op_type=['QSigmoid', 'iqSigmoid'])
def iqsigmoid(inputs, kwargs):
    input = inputs[0]
    kwargs = canonicalize_attrs(kwargs)

    instance = create_qmodule_tensor(QSigmoid, None, 1, kwargs)
    return instance(input)

@register_op(op_type=['QTanh', 'iqTanh'])
def iqtanh(inputs, kwargs):
    input = inputs[0]
    kwargs = canonicalize_attrs(kwargs)

    instance = create_qmodule_tensor(QTanh, None, 1, kwargs)
    return instance(input)

@register_op(op_type=['QSoftmax', 'SoftmaxInt'])
def softmaxInt(inputs,kwargs):
    input = inputs[0]
    kwargs = canonicalize_attrs(kwargs)

    dim = kwargs.get('axis', -1)
    kwargs['dim'] = dim

    instance = create_qmodule_tensor(QSoftmax, None, 1, kwargs)
    return instance(input)

# @register_op(op_type="LogSoftmaxInt")
# def softmaxInt(inputs,kwargs):
#     platform = kwargs.get("platform", "")
#     op_cls = get_op_class(platform, "LogSoftmaxInt")
#     return op_cls.excute_base(inputs, kwargs)

# @register_op(op_type="ReQuant")
# def onnxinferdequant(inputs,kwargs):
#     import math
#     src_bits = kwargs.get("bit_src")
#     dst_bits = kwargs.get('bit_dst')
#     scale_src = kwargs.get('scale_src')
#     s_rescale = (math.pow(2,dst_bits-1) -1.0)/(math.pow(2,src_bits-1) -1.0)
#     if kwargs.get('qmax') == 2: #qvalue
#         s_rescale = math.pow(2,round(math.log(s_rescale,2)))
#     scale = s_rescale*scale_src
#     zero_point = 0
#     if isinstance(input, QTensor):
#         zero_point = input.zero_point
#     if zero_point != 0:
#         zero_point = math.pow(2, dst_bits-1)

#     s = from_torch_tensor(inputs[0],scale,dst_bits, zero_point=zero_point)
#     s.requant_()
#     return s

@register_op(op_type='topN')
def topn(inputs, kwargs):
    input, idx_offset = inputs
    assert isinstance(input, linger.QTensor) == True, 'input of topN must be QTensor'
    dim = kwargs.get('dim', -1)
    assert dim == -1 or dim == input.ndim-1, f'only the last dim is supported, the current value is {dim}'
    max_num = kwargs.get('max_num', 1)
    assert max_num == 1, f'only max_num=1 is supported, the current value is {max_num}'
    assert input.shape[0] == 1, 'only input.shape[0] == 1 is supported.'

    scale = input.scale
    zp = input.zero_point
    data_bits = kwargs.get('data_bits', 8)
    quant_mode = StringToQuantMode(kwargs.get('quant_mode', 'floor_add'))
    
    q_input = (input * scale + 0.5).floor().to(torch.int32).clamp(-128, 127).cpu()
    q_input, _ = quant(input, scale, zp, quant_mode)
    val, idx = torch.topk(q_input, max_num, dim)
    idx = idx.to(torch.int32) + idx_offset
    res = torch.cat([val, idx], dim=0)
    return res

@register_op(op_type='topN2')
def topn2(inputs, kwargs):
    dim = kwargs.get('dim', -1)
    assert dim == -1 or dim == input.ndim-1, f'only the last dim is supported, the current value is {dim}'
    max_num = kwargs.get('max_num', 1)
    assert max_num == 1, f'only max_num=1 is supported, the current value is {max_num}'

    input = inputs[0]
    
    leading = torch.tensor(input.shape[:dim]).prod()
    val, ori_idx = torch.tensor_split(input, 2, 0)
    ori_idx = ori_idx.to(torch.long)
    max_val, fake_idx = torch.topk(val, max_num, dim)
    real_idx = torch.gather(ori_idx, dim=dim, index=fake_idx)
    res = torch.cat([max_val, real_idx], dim=0)
    return res