import  onnx
from onnx import numpy_helper
import numpy as np
from onnx import TensorProto
from .onnx_utils import get_node_id,remove_identity_node


def get_quant_conv_node(onnx_model, node, scale_x, scale_w, scale_o, platform_quant):
    # scale_x = 16.0
    # scale_w = 64.0
    # scale_o = 16.0
    domain = "thinker"
    activation_bits = 8
    weight_bits = 8

    # get weights of conv
    weight_data = None
    bias_data = None
    
    # for init in reversed(onnx_model.graph.initializer):
    for init in onnx_model.graph.initializer[::-1]:

        if init.name == node.input[1]:
            weight_data = numpy_helper.to_array(init)
            onnx_model.graph.initializer.remove(init)
        try:
            if init.name == node.input[2]:
                bias_data = numpy_helper.to_array(init)
                onnx_model.graph.initializer.remove(init)
        except:
            pass

    # initializer quantization
    # the definition of scale is different from the normal !!!
    # todo, how to handle data type
    if weight_data.dtype == "float32":
        weight_data = np.floor(weight_data * scale_w + 0.5).astype(np.int8)    
    weight_tensor = onnx.helper.make_tensor(node.input[1], TensorProto.INT8, weight_data.shape, \
        weight_data.tobytes(), raw=True)  # raw 
    try:
        if bias_data.dtype == "float32":
            bias_data = np.floor(bias_data * scale_w * scale_x + 0.5).astype(np.int32)
        bias_tensor = onnx.helper.make_tensor(node.input[2], TensorProto.INT32, bias_data.shape, bias_data.tobytes(), raw=True)     
    except:
        pass           
    if node.op_type == "Conv":
        if len(node.input) == 3:
            quant_node = onnx.helper.make_node(
                "Conv2dInt",
                name=node.name,
                inputs=[node.input[0], node.input[1], node.input[2]],
                outputs=[node.output[0]],
                domain=domain
            )
        else:
            quant_node = onnx.helper.make_node(
                "Conv2dInt",
                name=node.name,
                inputs=[node.input[0], node.input[1]],
                outputs=[node.output[0]],
                domain=domain
            )
    elif node.op_type == "ConvTranspose":
        if len(node.input) == 3:
            quant_node = onnx.helper.make_node(
                "ConvTranspose2dInt",
                name=node.name,
                inputs=[node.input[0], node.input[1], node.input[2]],
                outputs=[node.output[0]],
                domain=domain
            )
        else:
            quant_node = onnx.helper.make_node(
                "ConvTranspose2dInt",
                name=node.name,
                inputs=[node.input[0], node.input[1]],
                outputs=[node.output[0]],
                domain=domain
            )

    # add attributes of original node to the quant node
    quant_node.attribute.extend(
        onnx.helper.make_attribute(attr.name, onnx.helper.get_attribute_value(attr)) for attr in node.attribute
    )
    # add attributes that original node does not have to the quant node
    quant_node.attribute.extend(
        [onnx.helper.make_attribute("scale_x", scale_x), onnx.helper.make_attribute("scale_w", scale_w), onnx.helper.make_attribute("scale_o", scale_o), \
            onnx.helper.make_attribute("data_bits", activation_bits), onnx.helper.make_attribute("o_bits", activation_bits), onnx.helper.make_attribute("parameter_bits", weight_bits), \
            onnx.helper.make_attribute("platform_quant", platform_quant)]
    )
    if node.op_type == "ConvTranspose":
        quant_node.attribute.extend(
        [onnx.helper.make_attribute("output_padding", [0,0])])

    # insert the quant initializers
    try:
        onnx_model.graph.initializer.extend([weight_tensor, bias_tensor])
    except:
        onnx_model.graph.initializer.extend([weight_tensor])
    return quant_node

def insert_op_before(model, target_node_index, ori_node, scale_x, scale_w, scale_o, platform_quant ):
    '''
    op_name
    weight_dict
    attr_dict
    '''

    replace_node = get_quant_conv_node(model, ori_node, scale_x=scale_x, scale_w=scale_w, scale_o=scale_o, platform_quant=platform_quant)
    
    model.graph.node.insert(target_node_index, replace_node)

def replace_conv2dint(module, scale_x, scale_w, scale_o, platform_quant):
    nodes  = module.graph.node[::-1]
    remove_node_all=[]
    for i,node in enumerate(nodes):
        if node.op_type == 'Conv' or node.op_type == "ConvTranspose":
            remove_node_all.append((i,node))

    from onnx import numpy_helper
    input_tensors = {  t.name: numpy_helper.to_array(t) for t in module.graph.initializer }
            
    for i_node in remove_node_all:
        i    = i_node[0]
        node = i_node[1] 

        origin_layer_index = get_node_id(node,module.graph.node)
        module.graph.node.remove(node)        
        insert_op_before(
                    module,
                    target_node_index = origin_layer_index,
                    ori_node = node,
                    scale_x = scale_x, 
                    scale_w = scale_w, 
                    scale_o = scale_o, 
                    platform_quant = platform_quant
                     )
   
    return module