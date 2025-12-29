import torch
from torch.onnx import is_in_onnx_export
from typing import Dict, Any, Optional
from torch.types import _dtype as DType

from .qadd import QAdd
from .qmul import QMul
from .qbmm import QBmm
from .qmatmul import QMatmul
from .qcat import QCat
from .qsigmoid import QSigmoid
from .qtanh import QTanh
from .qsoftmax import QSoftmax
from .qsqueeze import QSqueezeOnnxFunction
from ..qconfig import *
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor, qfallback
from ...quantizer import AQuantizer
from ....config import QUANT_CONFIGS
from ....onnx import generate_onnx_qparam_dict, QDOMAIN_NAME

# @register_qtensor_op([torch.ops.aten.add, torch.ops.aten.add_])
# @register_qtensor_op([torch.add, torch._C.TensorBase.add, torch._C.TensorBase.add_])
@register_qtensor_op([torch.add, torch.Tensor.add, torch.Tensor.add_])
def add(op, input, other):
    module_self = get_current_module()
    if module_self is None:
        return qfallback(op, input, other)

    iname_index = getattr(module_self, LINGER_QTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_QTENSOR_LAYER_COUNTER, iname_index+1)
    var_name = LINGER_QTENSOR_LAYERS_PREIFX + '_qadd_' + str(iname_index)

    q_layer = None
    if hasattr(module_self, var_name):
        q_layer = getattr(module_self, var_name)
    else:
        q_layer = QAdd(activate_config = QUANT_CONFIGS.quant_info.to_dict(), num_input = 2)
        q_layer.training = module_self.training
        q_layer = q_layer.to(input.device)
        setattr(module_self, var_name, q_layer)
    output = q_layer(input, other)
    return output

# @register_qtensor_op([torch.ops.aten.mul, torch.ops.aten.mul_])
@register_qtensor_op([torch.mul, torch.Tensor.mul, torch.Tensor.mul_])
def mul(op, input, other):
    module_self = get_current_module()
    if module_self is None:
        return qfallback(op, input, other)

    iname_index = getattr(module_self, LINGER_QTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_QTENSOR_LAYER_COUNTER, iname_index+1)
    var_name = LINGER_QTENSOR_LAYERS_PREIFX + '_qmul_' + str(iname_index)

    q_layer = None
    if hasattr(module_self, var_name):
        q_layer = getattr(module_self, var_name)
    else:
        q_layer = QMul(activate_config = QUANT_CONFIGS.quant_info.to_dict(), num_input = 2)
        q_layer.training = module_self.training
        q_layer = q_layer.to(input.device)
        setattr(module_self, var_name, q_layer)
    output = q_layer(input, other)
    return output

@register_qtensor_op([torch.matmul])
def matmul(op, input, other):
    module_self = get_current_module()
    if module_self is None:
        return qfallback(op, input, other)

    iname_index = getattr(module_self, LINGER_QTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_QTENSOR_LAYER_COUNTER, iname_index+1)
    var_name = LINGER_QTENSOR_LAYERS_PREIFX + '_qmatmul_' + str(iname_index)

    q_layer = None
    if hasattr(module_self, var_name):
        q_layer = getattr(module_self, var_name)
    else:
        q_layer = QMatmul(activate_config = QUANT_CONFIGS.quant_info.to_dict(), num_input = 2)
        q_layer.training = module_self.training
        q_layer = q_layer.to(input.device)
        setattr(module_self, var_name, q_layer)
    output = q_layer(input, other)
    return output

@register_qtensor_op([torch.Tensor.contiguous])
def contiguous(op, input):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input)
        qtensor = from_tensor_to_qtensor(out, scale, data_bits)
        return qtensor
    return op(input, *size)

@register_qtensor_op([torch.bmm, torch.Tensor.bmm])
def bmm(op, input, other):
    module_self = get_current_module()
    if module_self is None:
        return qfallback(op, input, other)

    iname_index = getattr(module_self, LINGER_QTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_QTENSOR_LAYER_COUNTER, iname_index+1)
    var_name = LINGER_QTENSOR_LAYERS_PREIFX + '_qbmm_' + str(iname_index)

    q_layer = None
    if hasattr(module_self, var_name):
        q_layer = getattr(module_self, var_name)
    else:
        q_layer = QBmm(activate_config = QUANT_CONFIGS.quant_info.to_dict(), num_input = 2)
        q_layer.training = module_self.training
        q_layer = q_layer.to(input.device)
        setattr(module_self, var_name, q_layer)
    output = q_layer(input, other)
    return output

# @register_qtensor_op([torch.concat, torch.cat])
@register_qtensor_op([torch.cat])
def cat(op, input, dim = 0):
    module_self = get_current_module()
    if module_self is None:
        return qfallback(op, input, dim)

    iname_index = getattr(module_self, LINGER_QTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_QTENSOR_LAYER_COUNTER, iname_index+1)
    var_name = LINGER_QTENSOR_LAYERS_PREIFX + '_qcat_' + str(iname_index)

    q_layer = None
    if hasattr(module_self, var_name):
        q_layer = getattr(module_self, var_name)
    else:
        q_layer = QCat(activate_config = QUANT_CONFIGS.quant_info.to_dict(), num_input = 2, is_cat=True)
        q_layer.training = module_self.training
        q_layer = q_layer.to(input[0].device)
        setattr(module_self, var_name, q_layer)
    output = q_layer(input, dim)
    return output

@register_qtensor_op([torch.reshape, torch.Tensor.reshape])
def reshape(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.split, torch.Tensor.split])
def split(op, input, split_size_or_sections, dim: int = 0):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, split_size_or_sections, dim)
        qtensor_list = []
        for tensor in out:
            qtensor = from_tensor_to_qtensor(tensor, scale, data_bits)
            qtensor_list.append(qtensor)
        return qtensor_list
        # return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, split_size_or_sections, dim)

@register_qtensor_op([torch.flip, torch.Tensor.flip])
def flip(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.Tensor.view])
def view(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.transpose, torch.Tensor.transpose])
def transpose(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.permute, torch.Tensor.permute])
def permute(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.Tensor.reshape])
def reshape(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.ops.aten.slice])
def slice(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.ops.aten.select, torch.select, torch.Tensor.select])
def slice(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.squeeze, torch.Tensor.squeeze, torch.Tensor.squeeze_])
def squeeze(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        if torch.onnx.is_in_onnx_export():
            out = QSqueezeOnnxFunction.apply(input, *size)
        else:
            out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.unsqueeze, torch.Tensor.unsqueeze, torch.Tensor.unsqueeze_])
def unsqueeze(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.flatten, torch.Tensor.flatten])
def flatten(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.ops.aten.__getitem__])
def __getitem__(op, input, *size):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *size)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *size)

@register_qtensor_op([torch.nn.functional.pad])
def pad(op, input, pad, mode: str = ..., value: Optional[float] = None):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, pad, mode, value)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, pad, mode, value)

@register_qtensor_op([torch.sigmoid, torch.sigmoid_, torch.Tensor.sigmoid, torch.Tensor.sigmoid_])
def sigmoid(op, input):
    module_self = get_current_module()
    if module_self is None:
        return qfallback(op, input)

    iname_index = getattr(module_self, LINGER_QTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_QTENSOR_LAYER_COUNTER, iname_index+1)
    var_name = LINGER_QTENSOR_LAYERS_PREIFX + '_qsigmoid_' + str(iname_index)

    q_layer = None
    if hasattr(module_self, var_name):
        q_layer = getattr(module_self, var_name)
    else:
        q_layer = QSigmoid(activate_config = QUANT_CONFIGS.quant_info.to_dict(), num_input=1)
        q_layer.training = module_self.training
        q_layer = q_layer.to(input.device)
        setattr(module_self, var_name, q_layer)
    output = q_layer(input)
    return output

@register_qtensor_op([torch.tanh, torch.tanh_, torch.Tensor.tanh, torch.Tensor.tanh_])
def tanh(op, input):
    module_self = get_current_module()
    if module_self is None:
        return qfallback(op, input)

    iname_index = getattr(module_self, LINGER_QTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_QTENSOR_LAYER_COUNTER, iname_index+1)
    var_name = LINGER_QTENSOR_LAYERS_PREIFX + '_qtanh_' + str(iname_index)

    q_layer = None
    if hasattr(module_self, var_name):
        q_layer = getattr(module_self, var_name)
    else:
        q_layer = QTanh(activate_config = QUANT_CONFIGS.quant_info.to_dict(), num_input=1)
        q_layer.training = module_self.training
        q_layer = q_layer.to(input.device)
        setattr(module_self, var_name, q_layer)
    output = q_layer(input)
    return output

@register_qtensor_op([torch.softmax, torch._softmax, torch.Tensor.softmax, torch.nn.functional.softmax])
def softmax(op, input, dim, _stacklevel: int = 3, dtype: Optional[DType] = None):
    module_self = get_current_module()
    if module_self is None:
        return qfallback(op, input, dim)

    iname_index = getattr(module_self, LINGER_QTENSOR_LAYER_COUNTER)
    setattr(module_self, LINGER_QTENSOR_LAYER_COUNTER, iname_index+1)
    var_name = LINGER_QTENSOR_LAYERS_PREIFX + '_qsoftmax_' + str(iname_index)

    q_layer = None
    if hasattr(module_self, var_name):
        q_layer = getattr(module_self, var_name)
    else:
        q_layer = QSoftmax(activate_config = QUANT_CONFIGS.quant_info.to_dict(), num_input=2, dim = dim)
        q_layer.training = module_self.training
        q_layer = q_layer.to(input.device)
        setattr(module_self, var_name, q_layer)
    q_layer.dim = dim
    output = q_layer(input, dim)
    return output

@register_qtensor_op([torch.relu, torch.relu_, torch.nn.functional.relu, torch.Tensor.relu, torch.Tensor.relu_])
def relu(op, input, *args, **kwargs):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *args, **kwargs)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *args, **kwargs)

@register_qtensor_op([torch.topk, torch.Tensor.topk])
def topk(op, input, *args, **kwargs):
    assert isinstance(input, QTensor), 'input is not QTensor'
    if isinstance(input, QTensor):
        tmp_input = from_qtensor_to_tensor(input)
        scale = input.scale.detach()
        data_bits = input.data_bits
        out = op(tmp_input, *args, **kwargs)
        return from_tensor_to_qtensor(out, scale, data_bits)
    return op(input, *args, **kwargs)

