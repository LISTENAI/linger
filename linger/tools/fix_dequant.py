import linger
import numpy as np
import onnx

onnx_dtype = {
    'Undefined': 'UNDEFINED',    'Float': 'float32',    'UInt8': 'uint8',    'Int8': 'int8',    'UInt16': 'uint16',
    'Int16': 'int16',    'Int32': 'int32',    'Int64': 'int64',    'String': 'str',    'Bool': 'bool',    'Float16': 'float16',
    'Double': 'double',    'UInt32': 'uint32',    'UInt64': 'uint64',    'Complex64': 'complex64',    'Complex128': 'complex128',
    'BFloat16': 'bfloat16'
}


def remove_identity_node(model_path, op_type="Identity"):
    model = onnx.load(model_path)
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

    onnx.save(model, model_path[:-5]+"_remove_identity.onnx")


def check_model_run(fixed_onnx):  # 用于检测修复后的onnx是否能够正常前向运行
    import onnxinfer

    sessoption = onnxinfer.InferSessionOptions()

    # 此处fixed_onnx  即为上面成功运行的onnx模型
    sess = onnxinfer.InferSession(fixed_onnx, sessoption, is_fuse=0)

    data = {}
    for i in range(sess.GetInputCount()):
        ishape = sess.GetInputTypeInfo(i)
        inp = np.ones(ishape.GetShape(),
                      dtype=onnx_dtype[ishape.GetElementDataType().name])
        data[sess.GetInputNames()[i]] = inp

    option = onnxinfer.InferRunOptions()
    rlt = sess.Run(run_option=option, data_in=data)
    print(rlt[0].AsReadOnlyNumpy().shape)  # 能成功输出shape即可证明onnx图前向运行正常
    print("The Fixed model detection runs successfully !")


def fix_dequant(model_name, is_check):
    model_name = model_name[:-5]  # 原始出错的onnx模型名称

    ori_onnx = model_name + ".onnx"
    remove_dequant_onnx = model_name + "_remove_identity.onnx"
    fixed_onnx = model_name + "_fix.onnx"

    remove_identity_node(ori_onnx, "Dequant")

    model = onnx.load(remove_dequant_onnx)

    model = linger.parser_dequant(model, False)  # 此处的linger使用最新master版的环境
    # model.opset_import[0].version   =   10

    onnx.save(model, fixed_onnx)  # 最后将修复好的onnx保存为 后缀多了_fix.onnx
    print("ONNX fix over! Save model as onnx file \"", fixed_onnx, "\"")

    if is_check:
        check_model_run(fixed_onnx)