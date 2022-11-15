import onnx


def replace_add_to_iqadd(model, scale_x, scale_y, scale_o, platform_quant):
    for node in model.graph.node[::-1]:
        if node.op_type == "Add":
            node.op_type = "iqadd"
            node.domain = "thinker"
            node.attribute.extend(
            [onnx.helper.make_attribute("scale_x", scale_x), onnx.helper.make_attribute("scale_y", scale_y), onnx.helper.make_attribute("scale_o", scale_o), \
                onnx.helper.make_attribute("data_bits", 8), onnx.helper.make_attribute("o_bits", 8), onnx.helper.make_attribute("parameter_bits", 8), \
                onnx.helper.make_attribute("platform_quant", platform_quant)]
        )
    return model