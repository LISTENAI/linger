import onnx


def replace_iqsigmoid(model, scale_x, scale_o, platform_quant):
    for node in model.graph.node[::-1]:
        if node.op_type == "Sigmoid":
            node.op_type = "iqSigmoid"
            node.domain = "thinker"
            node.attribute.extend(
            [onnx.helper.make_attribute("scale_x", scale_x), onnx.helper.make_attribute("scale_o", scale_o), \
                onnx.helper.make_attribute("data_bits", 8), onnx.helper.make_attribute("o_bits", 8), onnx.helper.make_attribute("parameter_bits", 8), \
                onnx.helper.make_attribute("platform_quant", platform_quant)]
        )
    return model