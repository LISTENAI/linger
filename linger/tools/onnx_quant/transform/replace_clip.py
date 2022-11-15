import onnx
from onnx import numpy_helper

def replace_clip_attr(model,scale_o): #only support opset 9 and below
    if model.opset_import[0].version <12:
        assert False, "Clip quant only support opset 12 and above !"
    
    for index,node in enumerate(model.graph.node[::-1]):
        if node.op_type == "Clip":
            model.graph.node[::-1][index+1].attribute[0].i = 3
            model.graph.node[::-1][index+2].attribute[0].i = 3
            # max_value = int.from_bytes(model.graph.node[::-1][index+3].attribute[0].t.raw_data, byteorder='little', signed=True)
            # min_value = int.from_bytes(model.graph.node[::-1][index+4].attribute[0].t.raw_data, byteorder='little', signed=True)

            max_value = numpy_helper.to_array(model.graph.node[::-1][index+3].attribute[0].t)
            min_value = numpy_helper.to_array(model.graph.node[::-1][index+4].attribute[0].t)

            max_value = max_value * scale_o
            min_value = min_value * scale_o
            
            model.graph.node[::-1][index+3].attribute[0].t.data_type = 11  #double
            model.graph.node[::-1][index+4].attribute[0].t.data_type = 11

            model.graph.node[::-1][index+3].attribute[0].t.raw_data = max_value.tobytes()
            model.graph.node[::-1][index+4].attribute[0].t.raw_data= min_value.tobytes()

    return model