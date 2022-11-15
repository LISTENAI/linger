import onnx



def get_node_id(node, nodes):
    for k in range(len(nodes)):
        if node==nodes[k]:
            return k


def remove_identity_node(model, op_type="Identity"):
    r"""实现onnx的identity节点移除
    Note:本功能针对的是导出onnx时未自动删除identity节点的onnx模型，linger最新版导出已自动支持此功能

    Args:
        model_path: 要修改的onnx路径
        op_type   : 要删去的无用节点类型（默认为Identity）
    
    Notes:
        本函数不返回数据    函数会将修改后的删除完相应类型节点后的模型  写入到要保存的新路径中，设为原始名称+“_remove_identity.onnx”

    """
    # model = onnx.load(model_path)
    graph_output_name = []
    for ii in model.graph.output:
        graph_output_name.append(ii.name)
    nodes  = model.graph.node[::-1]
    for i,node in enumerate(nodes):
        if node.op_type== op_type:
            model.graph.node.remove(node)
            if node.output[0] in graph_output_name:
                for each in model.graph.node:
                    for idx in range(len(each.output)):
                        if each.output[idx]==node.input[0]:
                            each.output[idx] = node.output[0]
                for each in model.graph.node:
                    for idx in range(len(each.input)):
                        if each.input[idx]==node.input[0]:
                            each.input[idx] = node.output[0]

                for each in model.graph.node:
                    if each.op_type == "If":
                        for gi in range(len(each.attribute)):
                            for x_node in each.attribute[gi].g.node:
                                for idx in range(len(x_node.output)):
                                    if x_node.output[idx]==node.input[0]:
                                        x_node.output[idx] = node.output[0]
                                for idx in range(len(x_node.input)):
                                    if x_node.input[idx]==node.input[0]:
                                        x_node.input[idx] = node.output[0]
                        
            else:
                for each in model.graph.node:
                    for idx in range(len(each.input)):
                        if each.input[idx]==node.output[0]:
                            each.input[idx] = node.input[0]
                # for each in model.graph.node:
                #     for idx in range(len(each.output)):
                #         if each.output[idx]==node.output[0]:
                #             each.output[idx] = node.input[0]
                #             print(each.output[idx])
                for each in model.graph.node:
                    if each.op_type == "If":
                        for gi in range(len(each.attribute)):
                            for x_node in each.attribute[gi].g.node:
                                for idx in range(len(x_node.input)):
                                    if x_node.input[idx]==node.output[0]:
                                        x_node.input[idx] = node.input[0]
                                # for idx in range(len(x_node.output)):
                                #     if x_node.output[idx]==node.output[0]:
                                #         x_node.output[idx] = node.input[0]
    # onnx.save(model,model_path[:-5]+"_remove_identity.onnx")
    return model