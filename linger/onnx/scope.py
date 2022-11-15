from typing import Dict, List, Set

import torch
import torch.nn

from ..ops.iqtensor import IQTensor, from_torch_tensor


class ScopedEnterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, scope):
        m = t.clone()
        if isinstance(t, IQTensor):
            m = from_torch_tensor(m, t.scale_data, t.bits, t.zero_point)
        return m

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None

    @staticmethod
    def symbolic(g, input, scope):
        return g.op("thinker::ScopedEnter", input, scope_s=scope)


class ScopedLeaveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, scope):
        m = t.clone()
        if isinstance(t, IQTensor):
            m = from_torch_tensor(m, t.scale_data, t.bits, t.zero_point)
        return m

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None

    @staticmethod
    def symbolic(g, input, scope):
        return g.op("thinker::ScopedLeave", input, scope_s=scope)


global_scoped = {}


def scope_forward(input, scope_name, func):
    list_r = []
    for t in input:
        if isinstance(t, torch.Tensor):
            list_r.append(func.apply(t, scope_name))
        elif isinstance(t, dict):
            dict_r = {}
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    dict_r[k] = func.apply(v, scope_name)
                else:
                    dict_r[k] = v
            list_r.append(dict_r)
        elif isinstance(t, (tuple, list)):
            list_r.append(scope_forward(t, scope_name, func))
        else:
            list_r.append(t)
    return type(input)(list_r)


def enter_forward(module, input):
    scope_name = global_scoped[module][0]
    if isinstance(input, torch.Tensor):
        return ScopedEnterFunction.apply(input, scope_name)
    list_r = None
    if isinstance(input, (tuple, list)):
        list_r = scope_forward(input, scope_name, ScopedEnterFunction)
    elif isinstance(input, dict):
        dict_r = {}
        for k, v in input.items():
            assert isinstance(
                v, torch.Tensor), 'dict input only support depth=1'
            dict_r[k] = ScopedEnterFunction.apply(v, scope_name)
        return dict_r
    else:
        assert 0, 'foward only support tensor, tuple or dict of tensor'
    return type(input)(list_r)


def leave_forward(module, input, output):
    scope_name = global_scoped[module][0]
    if isinstance(output, torch.Tensor):
        return ScopedLeaveFunction.apply(output, scope_name)
    list_r = None
    if isinstance(output, (tuple, list)):
        list_r = scope_forward(output, scope_name, ScopedLeaveFunction)
    elif isinstance(output, dict):
        dict_r = {}
        for k, v in input.items():
            if isinstance(v, torch.Tensor):
                dict_r[k] = ScopedLeaveFunction.apply(v, scope_name)
            else:
                dict_r[k] = v
        return dict_r
    else:
        assert 0, 'foward only support tensor or tuple of tensor'
    return type(output)(list_r)


def build_module_scope(model: torch.nn.Module):
    scopes = {}
    for name, child in model.named_children():
        hook_handle_foward_pre = child.register_forward_pre_hook(enter_forward)
        hook_handle_forward = child.register_forward_hook(leave_forward)
        scopes[child] = (name, hook_handle_foward_pre, hook_handle_forward)
        scopes.update(build_module_scope(child))
    return scopes


def build_global_scope(model: torch.nn.Module):
    global global_scoped
    global_scoped = build_module_scope(model)
    hook_handle_foward_pre = model.register_forward_pre_hook(
        enter_forward)  # not execute before forward
    hook_handle_forward = model.register_forward_hook(leave_forward)
    global_scoped[model] = (".", hook_handle_foward_pre, hook_handle_forward)
    return global_scoped


def remove_scoped_node(model, op_type):
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


def build_onnx_scope_info(onnx_model):

    nodes = onnx_model.graph.node
    node_son_dict: Dict[int, List[str]] = {}  # 保存所有的nodename 和对应的子节点，深度遍历使用
    node_scope_dict: Dict[int, List[str]] = {}  # 遍历后的名字保存 保存节点名字和对应的域名scope
    visited_node_set: Set[int] = set()  # 保存已经访问的节点
    # 先建立node之间的邻接关系
    nodeoutdict: Dict[int, str] = {}
    for node in nodes:
        for nodeout in node.output:
            nodeoutdict[nodeout] = node
        node_son_dict[id(node)] = []
    for node in nodes:
        for nodein in node.input:
            nodefather = nodeoutdict.get(nodein)
            if nodefather is not None:
                node_son_dict[id(nodefather)].append(node)
    # 根据邻接关系深度遍历所有节点
    nodetree = []  # 深度遍历栈
    # 加入图的所有输入节点  连接在graph.input上的第一个节点  原始只考虑了单一输入
    initializer_names = []
    for x in onnx_model.graph.initializer:
        initializer_names.append(x.name)
    input_idxs = []
    for idx, y in enumerate(onnx_model.graph.input):
        if y.name not in initializer_names:
            input_idxs.append(y.name)
    for node in onnx_model.graph.node[::-1]:
        for node_inp in node.input[::-1]:
            if node_inp in input_idxs:
                nodetree.append(node)
                visited_node_set.add(id(node))
                continue
    while len(nodetree) > 0:
        node = nodetree.pop()  # 获得栈中的节点
        node_sons = node_son_dict[id(node)]  # 获得当前节点的所有子节点
        if id(node) not in node_scope_dict:
            node_scope_dict[id(node)] = []
        for node_son in node_sons:
            if id(node_son) in visited_node_set:
                continue
            scopelist = node_scope_dict[id(node)].copy()  # 每个子节点都要记录一份当前的域
            if node.op_type == "ScopedEnter":
                scope = node.attribute[0].s.decode('utf-8')
                scopelist.append(scope)
            elif node.op_type == "ScopedLeave":
                scope = node.attribute[0].s.decode('utf-8')
                scopelist.pop()
            node_scope_dict[id(node_son)] = scopelist
            nodetree.append(node_son)  # 子节点入栈
            visited_node_set.add(id(node_son))  # 访问过子节点加入已访问
        # 遍历子节点结束后，保存修改的name
        if node.op_type != "ScopedEnter" and node.op_type != "ScopedLeave":
            newnamelist = node_scope_dict[id(node)]
            if len(newnamelist) > 1 and newnamelist[0] != '.':
                node.name = "." + ".".join(newnamelist) + "." + node.name
            elif len(newnamelist) > 1 and newnamelist[0] == '.':
                node.name = "." + ".".join(newnamelist[1:]) + "." + node.name
            elif len(newnamelist) == 1 and newnamelist[0] != '.':
                node.name = "." + newnamelist[0] + "." + node.name
            elif len(newnamelist) == 1 and newnamelist[0] == '.':
                node.name = "." + node.name
    # 删除多余节点
    onnx_model = remove_scoped_node(onnx_model, "ScopedEnter")
    onnx_model = remove_scoped_node(onnx_model, "ScopedLeave")

    return onnx_model


__all__ = ['build_global_scope', 'build_onnx_scope_info']
