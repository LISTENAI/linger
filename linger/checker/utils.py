from linger.utils import PlatForm
import onnx
from onnx import numpy_helper
from typing import Dict, Any
import torch
import torch.nn as nn
from linger.quant.ops.qconfig import get_qmodule_op, get_qtensor_op_dispatch
from linger.quant.ops.qmodule import QModuleMixin
from linger.quant.ops.qtensor.qtensor_mod import QModuleTensor
from linger.utils import QuantMode, dequant

def get_param(kwargs, key):
    value = kwargs.get(key)
    if value is None:
        raise KeyError("OP must have attribute '{}' !".format(key))
    return value

MODE_TABLE = {
    "floor": QuantMode.floor,
    "floor_add": QuantMode.floor_add,
    "round": QuantMode.round,
    "ceil": QuantMode.ceil,
}

def StringToQuantMode(mode: str):
    try:
        return MODE_TABLE[mode]
    except KeyError:
        raise ValueError(f"Invalid quant mode '{mode}'. Supported: {list(MODE_TABLE.keys())}")
    
def create_qmodule(q_cls: QModuleMixin, torch_module: nn.Module, device: torch.device, attrs: Dict[str, Any]):
    if not issubclass(q_cls, QModuleMixin):
        raise TypeError(f"Expected subclass of QModuleMixin, got {q_cls.__name__}")
    act_cfg = {'activate_bits': None}
    w_cfg = {'weight_bits': None}
    b_cfg = {'bias_bits': None}
    instance = q_cls.qcreate(torch_module, act_cfg, w_cfg, b_cfg, device=device)

    quant_mode = StringToQuantMode(attrs.get("quant_mode", 'floor_add'))
    data_bits = attrs.get('data_bits', 8)
    parameter_bits = attrs.get("parameter_bits", 8)
    bias_bits = attrs.get('bias_bits', 32)
    o_bits = attrs.get('o_bits', 8)
    scale_x = attrs.get('scale_x', None)
    scale_w = attrs.get('scale_w', None)
    scale_o = attrs.get('scale_o', None)
    scale_b = None
    if scale_x is not None and scale_w is not None:
        scale_b = scale_x * scale_w

    if scale_x is not None:
        instance.input_quantizer.round_mode = quant_mode
        instance.input_quantizer.scale = torch.tensor(scale_x, dtype=torch.float32)
        instance.input_quantizer.data_bits = data_bits
        instance.input_quantizer.training = False
    
    if scale_w is not None:
        instance.weight_quantizer.round_mode = quant_mode
        instance.weight_quantizer.scale = torch.tensor(scale_w, dtype=torch.float32)
        instance.weight_quantizer.data_bits = parameter_bits
        instance.weight_quantizer.training = False

    if scale_b is not None:
        instance.bias_quantizer.round_mode = quant_mode
        instance.bias_quantizer.scale = torch.tensor(scale_b, dtype=torch.float32)
        instance.bias_quantizer.data_bits = bias_bits
        instance.bias_quantizer.training = False

    if scale_o is not None:
        instance.output_quantizer.round_mode = quant_mode
        instance.output_quantizer.scale = torch.tensor(scale_o, dtype=torch.float32)
        instance.output_quantizer.data_bits = o_bits
        instance.output_quantizer.training = False
    
    return instance

def create_qmodule_tensor(q_cls: QModuleTensor, module: nn.Module, num_input: int, attrs: Dict[str, Any]):
    if not issubclass(q_cls, QModuleTensor):
        raise TypeError(f"Expected subclass of QModuleTensor, got {q_cls.__name__}")
    act_cfg = {'activate_bits': None}
    is_cat = attrs.get("is_cat", False)
    instance = q_cls.qcreate(module, act_cfg, num_input, dim=attrs.get('dim', None))
    instance.is_cat = is_cat
    
    quant_mode = StringToQuantMode(attrs.get("quant_mode", None))
    data_bits = attrs.get('data_bits', 8)
    o_bits = attrs.get('o_bits', 8)
    scale_o = attrs.get("scale_o", 1.0)
    zp_o = attrs.get('output_zero_point', 0)

    if is_cat:
        for i in range(num_input):
            scale = attrs.get(f'scale_x_{i}', 1.0)
            zp = attrs.get(f'input_zero_point_{i}', 0)
            
            instance.input_quantizer[i].data_bits = o_bits
            instance.input_quantizer[i].round_mode = quant_mode
            instance.input_quantizer[i].scale = torch.tensor(scale, dtype=torch.float32)
            instance.input_quantizer[i].training = False
    else:
        if num_input == 2:
            scale_x = attrs.get('scale_x', 1.0)
            zp_x = attrs.get("input_x_zero_point", 0)
            scale_y = attrs.get('scale_y', 1.0)
            zp_y = attrs.get("input_y_zeropoint", 0)

            instance.input_quantizer[0].data_bits = o_bits
            instance.input_quantizer[0].round_mode = quant_mode
            instance.input_quantizer[0].scale = torch.tensor(scale_x, dtype=torch.float32)
            instance.input_quantizer[0].training = False

            instance.input_quantizer[1].data_bits = o_bits
            instance.input_quantizer[1].round_mode = quant_mode
            instance.input_quantizer[1].scale = torch.tensor(scale_y, dtype=torch.float32)
            instance.input_quantizer[1].training = False
        else:
            scale_x = attrs.get('scale_x', 1.0)
            zp_x = attrs.get("data_zero_point", 0)
            
            instance.input_quantizer[0].data_bits = data_bits
            instance.input_quantizer[0].round_mode = quant_mode
            instance.input_quantizer[0].scale = torch.tensor(scale_x, dtype=torch.float32)
            instance.input_quantizer[0].training = False

    instance.output_quantizer.data_bits = o_bits
    instance.output_quantizer.round_mode = quant_mode
    instance.output_quantizer.scale = torch.tensor(scale_o, dtype=torch.float32)
    instance.output_quantizer.training = False

    return instance

def load_quantized_weights(q_instance, attrs, weights = None, bias = None):
    scale_x = attrs.get('scale_x', None)
    scale_w = attrs.get('scale_w', None)
    scale_b = None
    if scale_x is not None and scale_w is not None:
        scale_b = scale_x * scale_w
    
    if weights is not None and scale_w is not None:
        w_data = dequant(weights, scale_w)
        q_instance.weight = nn.Parameter(w_data, requires_grad=False)
        
    if bias is not None and scale_b is not None:
        b_data = dequant(bias, scale_b)
        q_instance.bias = nn.Parameter(b_data, requires_grad=False)
        
    return q_instance

def onnx_topologically_sort(model) :
    node_degree_dict = {}
    for node in model.graph.node:
        node.name = node.op_type + '_' + node.output[0]
        node_degree_dict[node.name] = 0
    for node in model.graph.node:
        for in_node in model.graph.node:
            for output in in_node.output:
                if output in node.input:
                    node_degree_dict[node.name] += 1
    begin_node = []
    for node in model.graph.node:
        if node_degree_dict[node.name] == 0:
            begin_node.append(node)
    sorted = []
    while len(begin_node) > 0:
        child_node = begin_node.pop()
        sorted.append(child_node)
        for node in model.graph.node:
            for output in child_node.output:
                if output in node.input:
                    node_degree_dict[node.name] -= 1
                    if node_degree_dict[node.name] == 0:
                        begin_node.append(node)
    assert len(sorted) == len(model.graph.node)

    model.graph.ClearField("node")
    model.graph.node.extend(sorted)

    return model

def get_attribute_value(node, attr_name):
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.FLOAT: return attr.f
            elif attr.type == onnx.AttributeProto.INT: return attr.i
            elif attr.type == onnx.AttributeProto.FLOATS: return attr.floats
            elif attr.type == onnx.AttributeProto.INTS: return attr.ints
            elif attr.type == onnx.AttributeProto.STRING: return attr.s.decode('utf-8')
    return None

def parse_attribute_and_name(node):
        node_attribute = dict()
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.AttributeType.INTS:
                node_attribute[attr.name] = tuple(attr.ints)
            elif attr.type == onnx.AttributeProto.AttributeType.INT:
                node_attribute[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.AttributeType.FLOAT:
                node_attribute[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.AttributeType.FLOATS:
                node_attribute[attr.name] = tuple(attr.floats)
            elif attr.type == onnx.AttributeProto.AttributeType.STRING:
                node_attribute[attr.name] = attr.s.decode('utf-8')
            elif attr.type == onnx.AttributeProto.AttributeType.TENSOR:
                node_attribute[attr.name] = list(numpy_helper.to_array(node.attribute[0].t))
            elif attr.type == onnx.AttributeProto.AttributeType.GRAPH:
                node_attribute[attr.name] = attr.g
            else:
                raise KeyError(
                            "The current operator({}) attribute({}) type is not supported,only support [float,int,ints,string,tensor,graph]".format(node.name,attr.name)
                        )
        return node.name, node_attribute

_Method_MAP={}

def register_op(op_type:str = None):
    def decorator(func, op_type):
        if op_type in _Method_MAP:
            raise LookupError("Operator %s already registered!" %op_type)
        _Method_MAP[op_type] = func
        return func
    
    if type(op_type) != str:
        func = op_type
        decorator(func,func.__name__)
        return func
    return lambda func : decorator(func, op_type)

def single_node_run(node:onnx.NodeProto,inputs:list):
    func = _Method_MAP.get(node.op_type, None)
    if func is None:
        raise NotImplementedError("Current Version don't support the {} ops.".format(node.op_type))
    if node.op_type == 'Constant':
        return func(node)
    else:
        node_name, kwargs = parse_attribute_and_name(node)
        return func(inputs, kwargs)

def if_node_run(node:onnx.NodeProto, inputs:list, inputs_dict:dict):
    # If you want to know why I do this, you can customize the network to use squeeze, and then observe the onnx graph to understand
    _, kwargs = parse_attribute_and_name(node)
    if inputs[0] == True:
        children_graph = kwargs.get("then_branch")  #  equal to _Method_MAP.get("Squeeze",None)
    else:
        children_graph = kwargs.get("else_branch")  # # equal to _Method_MAP.get("Identity",None)
    
    children_inputs = [inputs_dict.get(children_input_name) for children_input_name in children_graph.node[0].input]
    return single_node_run(children_graph.node[0], children_inputs)

def print_method_map():
    print(_Method_MAP)
        

__all__=["get_param",'register_op','node_run','print_method_map','onnx_topologically_sort','get_attribute_value',
         'parse_attribute_and_name']