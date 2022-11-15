import onnx

from ..__version import __version__
from .infer_type import *

onnx_dtype = {
    0: 'UNDEFINED',    1: 'float',    2: 'uint8',    3: 'int8',    4: 'uint16',
    5: 'int16',    6: 'int32',    7: 'int64',    8: 'str',    9: 'bool',    10: 'float16',
    11: 'double',    12: 'uint32',    13: 'uint64',    14: 'complex64',    15: 'complex128',
    16: 'bfloat16'
}

only_need_input0_op_list = ['Reshape']


def get_node_id(node, nodes):
    for k in range(len(nodes)):
        if node == nodes[k]:
            return k


def get_index_by_name(self, name):
    for i in range(len(self.model.graph.node)):
        if self.model.graph.node[i].name == name:
            return i


def get_node_by_index(model, index):
    return model.graph.node[index]


def insert_op_before(model, node_input, node_output, target_node_index, input_i, *args, **kwargs):
    '''
    op_name
    weight_dict
    attr_dict
    '''

    model.graph.node[target_node_index].input[input_i] = "Dequant_" + \
        str(node_input)+"_"+str(node_output)
    new_dequant_node = onnx.helper.make_node(
        'Dequant',
        inputs=[node_input],
        outputs=["Dequant_"+str(node_input)+"_"+str(node_output)],
        domain="thinker",
        **kwargs['attr_dict']
    )

    # set target_node input to new node outputs

    model.graph.node.insert(target_node_index, new_dequant_node)


def add_dequant_layer(model, target_node_index, saved_node, params, input_i):

    saved_scale_o = float(params[0])

    weight_dict = {
    }
    attr_dict = {
        "scale_o": saved_scale_o,
    }

    insert_op_before(
        model,
        node_input=saved_node.input[input_i],
        node_output=saved_node.output[0],
        target_node_index=target_node_index,
        input_i=input_i,
        weight_dict=weight_dict,
        attr_dict=attr_dict
    )


def insert_op_after(model, node_input, node_output, target_node_index, output_i, fixed_input, *args, **kwargs):
    '''
    op_name
    weight_dict
    attr_dict
    '''
    for each_node in model.graph.node:
        for input_idx in range(len(each_node.input)):
            if each_node.input[input_idx] == fixed_input:
                each_node.input[input_idx] = "Dequant_" + \
                    str(node_input)+"_"+str(node_output)

    model.graph.node[target_node_index].output[output_i] = "Dequant_" + \
        str(node_input)+"_"+str(node_output)
    new_dequant_node = onnx.helper.make_node(
        'Dequant',
        inputs=["Dequant_"+str(node_input)+"_"+str(node_output)],
        outputs=[node_input],
        domain="thinker",
        **kwargs['attr_dict']
    )

    # set target_node input to new node outputs
    model.graph.node.insert(target_node_index+1, new_dequant_node)


def add_dequant_layer_after(model, target_node_index, saved_node, params, output_i, fixed_input):

    saved_scale_o = float(params[0])

    weight_dict = {
    }
    attr_dict = {
        "scale_o": saved_scale_o,
    }

    insert_op_after(
        model,
        node_input=saved_node.output[output_i],
        node_output=saved_node.input[0],
        target_node_index=target_node_index,
        output_i=output_i,
        fixed_input=fixed_input,
        weight_dict=weight_dict,
        attr_dict=attr_dict
    )


def update_dequant_out(module, saved_node, other_params, output_i, fixed_input):

    origin_layer_index = get_node_id(saved_node, module.graph.node)

    add_dequant_layer_after(module, origin_layer_index,
                            saved_node, other_params, output_i, fixed_input)


def update_dequant(module, saved_node, other_params, input_i):

    origin_layer_index = get_node_id(saved_node, module.graph.node)

    add_dequant_layer(module, origin_layer_index,
                      saved_node, other_params, input_i)


def analyse_attribute(node, saved_scale_o):

    other_params = [saved_scale_o]

    for i in range(len(node.attribute)):

        if node.attribute[i].name == 'scale_o':
            other_params[0] = node.attribute[i].f
        if node.op_type == "BatchNorm2dInt":
            if node.attribute[i].name == 'scale_add_o':
                other_params[0] = node.attribute[i].f

    return other_params


def remove_identity_node(model, op_type="Identity"):
    graph_output_name = []
    for ii in model.graph.output:
        graph_output_name.append(ii.name)
    nodes = model.graph.node[::-1]
    for i, node in enumerate(nodes):
        if node.op_type == op_type:
            model.graph.node.remove(node)
            if node.output[0] in graph_output_name:
                for each in model.graph.node:
                    for idx in range(len(each.output)):
                        if each.output[idx] == node.input[0]:
                            each.output[idx] = node.output[0]
                for each in model.graph.node:
                    for idx in range(len(each.input)):
                        if each.input[idx] == node.input[0]:
                            each.input[idx] = node.output[0]

                for each in model.graph.node:
                    if each.op_type == "If":
                        for gi in range(len(each.attribute)):
                            for x_node in each.attribute[gi].g.node:
                                for idx in range(len(x_node.output)):
                                    if x_node.output[idx] == node.input[0]:
                                        x_node.output[idx] = node.output[0]
                                for idx in range(len(x_node.input)):
                                    if x_node.input[idx] == node.input[0]:
                                        x_node.input[idx] = node.output[0]

            else:
                for each in model.graph.node:
                    for idx in range(len(each.input)):
                        if each.input[idx] == node.output[0]:
                            each.input[idx] = node.input[0]
                for each in model.graph.node:
                    if each.op_type == "If":
                        for gi in range(len(each.attribute)):
                            for x_node in each.attribute[gi].g.node:
                                for idx in range(len(x_node.input)):
                                    if x_node.input[idx] == node.output[0]:
                                        x_node.input[idx] = node.input[0]

    return model


def strategy_list_process(strategy_type_list):
    strategy_list_input = {}
    strategy_list_output = {}
    for k, v in enumerate(strategy_type_list):

        input_temp = []
        output_temp = []
        for ele in strategy_type_list[v]:
            input_temp.append(ele[0])
            output_temp.append(ele[1])
        strategy_list_input[v] = input_temp
        strategy_list_output[v] = output_temp
    return strategy_list_input, strategy_list_output


def add_node_name(model):
    for node in model.graph.node:
        node.name = node.name+str(node.op_type)+'_'+str(node.output[0])
    return model


def infer_type_linger(model, is_change_in_out_type):
    multi_input_non_quant_op_list = ['Concat', 'Mul', 'Add', 'Sub', 'Div', 'Max']
    
    initializer_names = []
    for x in model.graph.initializer:
        initializer_names.append(x.name)
    
    input_idxs = []
    for idx, y in enumerate(model.graph.input):
        if y.name not in initializer_names:
            input_idxs.append(y.name)

    nodes = model.graph.node
    tensor_type_map = {}
    remove_node_all = []
    out_change_nodes = []
    output_idxs = {}
    for idx, y in enumerate(model.graph.output):
        output_idxs[y.name] = onnx_dtype[y.type.tensor_type.elem_type]
    for inp in model.graph.input:
        tensor_type_map[inp.name] = inp.type.tensor_type.elem_type
    for inp in model.graph.initializer:
        tensor_type_map[inp.name] = inp.data_type

    for i, node in enumerate(nodes):
        in_type = []
        
        try:
            if node.op_type == 'LSTM':
                in_type = [tensor_type_map[node.input[0]]]
            elif node.op_type == 'GRU':
                in_type = [tensor_type_map[node.input[0]],
                           tensor_type_map[node.input[1]], tensor_type_map[node.input[2]]]
            else:
                in_type = [tensor_type_map[inp] for inp in node.input]
            
            if node.op_type in op_map.keys():
                it = op_map[node.op_type](node)
                # consider  slice/split  no-quant float situation ,   split don't support intop
                if node.op_type == "Slice" or node.op_type == "Split":
                    for quant_node in nodes:
                        for iii in range(len(node.output)):
                            if quant_node.input == [node.output[iii]] and quant_node.op_type == "OnnxInferQuant":
                                if in_type[0] == 3:
                                    in_type[0] = 0
            else:
                print("Warning: InferType OP ", node.op_type,
                      " is not supported,this may cause error !")
                it = op_map['Others'](node)

            if node.op_type == 'If':
                tensor_type_map.update(it.infer_type(in_type, tensor_type_map))
            else:
                tensor_type_map.update(it.infer_type(in_type))
        except:
            flag = 1
            for node_input_i in node.input:  # does it exist Quant op
                if node_input_i in input_idxs:
                    for iid in model.graph.input:
                        if iid.name == node_input_i and node.op_type not in multi_input_non_quant_op_list:
                            iid.type.tensor_type.elem_type = 3
                            tensor_type_map[node_input_i] = 3
                            flag = 0

            if flag:
                idx = 0
                if len(node.input) >= 2:
                    if node.op_type == "LSTM":
                        if tensor_type_map[node.input[0]] == 3:
                            remove_node_all.append((i, node, 0))
                    elif node.op_type == "GRU":
                        for x in range(3):
                            if tensor_type_map[node.input[x]] == 3:
                                remove_node_all.append((i, node, x))
                    elif node.op_type == "Clip":
                        if tensor_type_map[node.input[0]] == 3:
                            remove_node_all.append((i, node, 0))
                    else:
                        for x in range(len(node.input)):
                            if tensor_type_map[node.input[x]] == 3:
                                remove_node_all.append((i, node, x))
                else:
                    remove_node_all.append((i, node, idx))
                for output in node.output:
                    tensor_type_map[output] = 1
            else:
                for output in node.output:
                    tensor_type_map[output] = 3

        if not is_change_in_out_type:
            for output_i in range(len(node.output)):
                if node.output[output_i] in output_idxs.keys():
                    if tensor_type_map[node.output[output_i]] in [3, 6]:
                        out_change_nodes.append(
                            (i, node, output_i, node.output[output_i]))
        else:
            for output_i in range(len(node.output)):
                if node.output[output_i] in output_idxs.keys():
                    for oid in model.graph.output:
                        if oid.name == node.output[output_i]:
                            if tensor_type_map[node.output[output_i]] != oid.type.tensor_type.elem_type:
                                oid.type.tensor_type.elem_type = tensor_type_map[node.output[output_i]]

    return remove_node_all, out_change_nodes


def seach_scale_o(model, op_name):
    def seach_before(model, id_):
        for node in model.graph.node:
            if id_ in node.output:
                for num in range(len(node.attribute)):
                    if node.op_type != "Dequant" and node.attribute[num].name == "scale_o":
                        scale_o = node.attribute[num].f
                        return scale_o
                    if node.op_type != "Dequant" and node.attribute[num].name == "scale_add_o":
                        scale_o = node.attribute[num].f
                        return scale_o
                try:
                    return seach_before(model, node.input[0])
                except:
                    return 1.0

    for node in model.graph.node:
        if node.name == op_name:
            scale_o = seach_before(model, node.input[0])
            if scale_o is None:
                scale_o = 1.0
            if scale_o != node.attribute[-1].f:
                print("Scale_o Warning: The Dequant node (" + op_name +
                      ") attribute scale_o has wrong risk, Check the exported onnx model! ")

            node.attribute[-1].f = scale_o


def check_scale_o(model, op_name):
    def seach_before(model, id_):
        for node in model.graph.node:
            if id_ in node.output:
                for num in range(len(node.attribute)):
                    if node.op_type != "Dequant" and node.attribute[num].name == "scale_o":
                        scale_o = node.attribute[num].f
                        return scale_o
                    if node.op_type != "Dequant" and node.attribute[num].name == "scale_add_o":
                        scale_o = node.attribute[num].f
                        return scale_o
                try:
                    return seach_before(model, node.input[0])
                except:
                    return 1.0

    for node in model.graph.node:
        if node.name == op_name:
            scale_o = seach_before(model, node.input[0])
            if scale_o is None:
                scale_o = 1.0
            if scale_o != node.attribute[-1].f:
                print("Scale_o Error: The Dequant node (" + op_name +
                      ") attribute scale_o has wrong risk, Check the exported onnx model! ")


def parser_dequant(module, is_change_in_out_type):
    scale_o_list = []
    for i, node in enumerate(module.graph.node):
        attr_nums = len(node.attribute)
        attr_names = []
        saved_scale_o = 1.0000000000
        for num in range(attr_nums):
            attr_names.append(node.attribute[num].name)
            if node.attribute[num].name == "scale_o" or node.attribute[num].name == "scale_add_o":
                saved_scale_o = node.attribute[num].f

        if "scale_o" not in attr_names and "scale_add_o" not in attr_names:
            if len(scale_o_list) == 0:
                scale_o_list.append(1)
            else:
                scale_o_list.append(scale_o_list[-1])
        else:
            scale_o_list.append(saved_scale_o)

    all_op_infertyped_dict = {}

    initializer_names = []
    for x in module.graph.initializer:
        initializer_names.append(x.name)

    input_idxs = []
    for y in module.graph.input:
        if y.name not in initializer_names:
            input_idxs.append(y.name)

    output_idxs = {}
    for y in module.graph.output:
        output_idxs[y.name] = onnx_dtype[y.type.tensor_type.elem_type]

    for input_name in input_idxs:
        all_op_infertyped_dict[str(input_name)] = "float"

    remove_node_all = []
    out_change_nodes = []

    remove_node_all, out_change_nodes = infer_type_linger(
        module, is_change_in_out_type)

    for i_node in remove_node_all[::-1]:
        i = i_node[0]
        node = i_node[1]
        input_i = i_node[2]

        other_params = []

        saved_node = node
        saved_scale_o = scale_o_list[i]

        other_params = analyse_attribute(node, saved_scale_o)

        update_dequant(module, saved_node, other_params, input_i)

    for o_node in out_change_nodes[::-1]:
        i = o_node[0]
        node = o_node[1]
        output_i = o_node[2]
        fixed_input = o_node[3]
        other_params = []

        saved_node = node
        saved_scale_o = scale_o_list[i]

        other_params = analyse_attribute(node, saved_scale_o)

        update_dequant_out(module, saved_node, other_params,
                           output_i, fixed_input)

    module = remove_identity_node(module)

    module = add_node_name(module)

    module.producer_name = "linger_" + __version__ + '_' + module.producer_name

    for i, node in enumerate(module.graph.node):
        if node.op_type == "Dequant":
            seach_scale_o(module, node.name)

    # check scale_o
    for i, node in enumerate(module.graph.node):
        if node.op_type == "Constant" and '.weight' in node.output[0]:
            check_scale_o(module, node.name)

    for i, ini in enumerate(module.graph.initializer):
        if '.weight' in ini.name or '.bias' in ini.name:
            for node in module.graph.node:
                if len(node.input) > 0 and node.input[0] == ini.name and node.op_type == "Dequant":
                    print("Scale_o Error: The Dequant node (" + ini.name +
                          ") attribute scale_o is wrong ! ")

    return module


__all__ = ['parser_dequant']
