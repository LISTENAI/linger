import numpy as np

import onnx
import onnx.numpy_helper

onnx_dtype = {
    0: 'UNDEFINED',    1: 'float32',    2: 'uint8',    3: 'int8',    4: 'uint16',
    5: 'int16',    6: 'int32',    7: 'int64',    8: 'str',    9: 'bool',    10: 'float16',
    11: 'double',    12: 'uint32',    13: 'uint64',    14: 'complex64',    15: 'complex128',
    16: 'bfloat16'
}


def find_key(dtype):
    if dtype == 'float64':
        dtype = 'double'
    return list(onnx_dtype.keys())[list(onnx_dtype.values()).index(dtype)]


class OpBase():
    def __init__(self, node):
        self.node = node

    def infer_type(self, in_type):
        tensor_type_map = {}
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class QAdd(OpBase):
    def __init__(self, node):
        super(QAdd, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert (in_type[0] in [3, 6]) and (in_type[1] in [1, 3, 6])
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class QMul(OpBase):
    def __init__(self, node):
        super(QMul, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert (in_type[0] in [2, 3]) and (in_type[1] in [1, 2, 3])
        for output in self.node.output:
            tensor_type_map[output] = 3
        return tensor_type_map

class QCat(OpBase):
    def __init__(self, node):
        super(QCat, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] == 3
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class QSigmoid(OpBase):
    def __init__(self, node):
        super(QSigmoid, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] in [3, 5]
        for output in self.node.output:
            # 应当为uint8  (2)  但会导致dequant添加错误  所以暂时修改为int8输出
            tensor_type_map[output] = 3
        return tensor_type_map

class QTanh(OpBase):
    def __init__(self, node):
        super(QTanh, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] in [3, 5]
        for output in self.node.output:
            tensor_type_map[output] = 3
        return tensor_type_map

class QAvgPool(OpBase):
    def __init__(self, node):
        super(QAvgPool, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] == 3
        count = 0
        out_type = 3
        for attr in self.node.attribute:
            if attr.name == "o_bits":
                count = count + 1
                if attr.i == 8:
                    out_type = 3
                if attr.i == 32:
                    out_type = 6
        for output in self.node.output:
            if count == 0:
                tensor_type_map[self.node.output[0]] = find_key('float32')
            else:
                tensor_type_map[output] = out_type
        return tensor_type_map

class QBatchNorm(OpBase):
    def __init__(self, node):
        super(QBatchNorm, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 3
        assert in_type[1] == 3 or in_type[1] == 5
        assert in_type[2] == 6
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class QLayerNorm(OpBase):
    def __init__(self, node):
        super(QLayerNorm, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [3]
        out_type = 3
        flag = False
        for attr in self.node.attribute:
            if attr.name == "o_bits":
                flag = True
                if attr.i == 8:
                    out_type = 3
                if attr.i == 32:
                    out_type = 6
        for output in self.node.output:
            if not flag:
                tensor_type_map[self.node.output[0]] = find_key('float32')
            else:
                tensor_type_map[output] = out_type
        return tensor_type_map

class QConv(OpBase):
    def __init__(self, node):
        super(QConv, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 3  # and in_type[1] == 3
        count = 0
        out_type = 3
        for attr in self.node.attribute:
            if attr.name == "o_bits":
                count = count + 1
                if attr.i == 8:
                    out_type = 3
                if attr.i == 32:
                    out_type = 6
        for output in self.node.output:
            if count == 0:
                tensor_type_map[self.node.output[0]] = find_key('float32')
            else:
                tensor_type_map[output] = out_type
        return tensor_type_map

class QConvTranspose(OpBase):
    def __init__(self, node):
        super(QConvTranspose, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 3  # and in_type[1] == 3
        count = 0
        out_type = 3
        for attr in self.node.attribute:
            if attr.name == "o_bits":
                count = count + 1
            if attr.name == "o_bits" and attr.i == 32:
                out_type = 6
            if attr.name == "o_bits" and attr.i == 8:
                out_type = 3

        for output in self.node.output:
            if count == 0:
                tensor_type_map[self.node.output[0]] = find_key('float32')
            else:
                tensor_type_map[output] = out_type

        return tensor_type_map

class QLinear(OpBase):
    def __init__(self, node):
        super(QLinear, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 3  # and in_type[1] == 3
        count = 0
        out_type = 3
        for attr in self.node.attribute:
            if attr.name == "o_bits":
                count = count + 1
                if attr.i == 8:
                    out_type = 3
                if attr.i == 32:
                    out_type = 6
        for output in self.node.output:
            if count == 0:
                tensor_type_map[self.node.output[0]] = find_key('float32')
            else:
                tensor_type_map[output] = out_type
        return tensor_type_map

class QLSTM(OpBase):
    def __init__(self, node):
        super(QLSTM, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 3
        out_type = 3
        for attr in self.node.attribute:
            if attr.name == "o_bits" and attr.i == 32:
                out_type = 6
            if attr.name == "o_bits" and attr.i == 8:
                out_type = 3

        tensor_type_map[self.node.output[0]] = out_type

        tensor_type_map[self.node.output[1]] = find_key('float32')
        # tensor_type_map[self.node.output[2]] = find_key('float32')

        return tensor_type_map

class QGRU(OpBase):
    def __init__(self, node):
        super(QGRU, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 3] and in_type[1] == 3 and in_type[2] == 3
        # for output in self.node.output:
        tensor_type_map[self.node.output[0]] = 3
        tensor_type_map[self.node.output[1]] = 1

        return tensor_type_map

class QBmm(OpBase):
    def __init__(self, node):
        super(QBmm, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 3 and in_type[1] == 3
        count = 0
        out_type = 3
        for attr in self.node.attribute:
            if attr.name == "o_bits":
                count = count + 1
                if attr.i == 8:
                    out_type = 3
                if attr.i == 32:
                    out_type = 6
        for output in self.node.output:
            if count == 0:
                tensor_type_map[self.node.output[0]] = find_key('float32')
            else:
                tensor_type_map[output] = out_type
        return tensor_type_map
    
class QGLU(OpBase):
    def __init__(self, node):
        super(QGLU, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] in [3, 5]
        for output in self.node.output:
            # 应当为uint8  (2)  但会导致dequant添加错误  所以暂时修改为int8输出
            tensor_type_map[output] = 3
        return tensor_type_map

class QSoftmax(OpBase):
    def __init__(self, node):
        super(QSoftmax, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [3, 5, 6]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map

class Quant(OpBase):
    def __init__(self, node):
        super(Quant, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        for output in self.node.output:
            tensor_type_map[output] = find_key('int8')
        return tensor_type_map

class Dequant(OpBase):
    def __init__(self, node):
        super(Dequant, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for output in self.node.output:
            tensor_type_map[output] = find_key('float32')
        return tensor_type_map

class Flatten(OpBase):
    def __init__(self, node):
        super(Flatten, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class Gather(OpBase):
    def __init__(self, node):
        super(Gather, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [6, 7]  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class MaxPool(OpBase):
    def __init__(self, node):
        super(MaxPool, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        count = 1
        if len(self.node.attribute) == 0:
            tensor_type_map[self.node.output[0]] = in_type[0]
            return tensor_type_map
        for idx in range(len(self.node.attribute)):
            if self.node.attribute[idx].name == "strides":
                count = idx
                break
        if len(self.node.attribute[count].ints) == 2:
            for index in range(len(in_type)):
                # corresponding with 1.7 onnxruntime doc
                assert in_type[index] in [1, 2, 3, 10, 11]
            tensor_type_map[self.node.output[0]] = in_type[0]
            if len(self.node.output) == 2:
                tensor_type_map[self.node.output[1]] = find_key('int64')
            return tensor_type_map
        elif len(self.node.attribute[count].ints) == 1:
            for index in range(len(in_type)):
                # corresponding with 1.7 onnxruntime doc
                assert in_type[index] in [1, 2, 3, 10, 11]
            tensor_type_map[self.node.output[0]] = in_type[0]
            if len(self.node.output) == 2:
                tensor_type_map[self.node.output[1]] = find_key('int64')
            return tensor_type_map
        else:
            print("Maxpool node error infertype!")
            exit()

class Relu(OpBase):
    def __init__(self, node):
        super(Relu, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11, 3, 6]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class Reshape(OpBase):
    def __init__(self, node):
        super(Reshape, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        if len(in_type) > 1:
            assert in_type[1] == 7
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class Slice(OpBase):
    def __init__(self, node):
        super(Slice, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [1, 6, 7]
        # corresponding with 1.7 onnxruntime doc
        assert in_type[2] in [1, 6, 7]
        if len(in_type) > 3:
            assert in_type[3] in [1, 6, 7]
        if len(in_type) == 5:
            assert in_type[4] in [1, 6, 7]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class Split(OpBase):
    def __init__(self, node):
        super(Split, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        if len(in_type) == 2:
            # corresponding with 1.7 onnxruntime doc
            assert in_type[1] in [1, 2, 3, 4, 5,
                                  6, 7, 8, 9, 10, 11, 12, 13, 16]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class Transpose(OpBase):
    def __init__(self, node):
        super(Transpose, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class Concat(OpBase):
    def __init__(self, node):
        super(Concat, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # 去除支持int8  int8下接concat 会添加dequant
            assert in_type[index] != 0 and in_type[index] != 3
            assert in_type[index] == in_type[0]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class Shape(OpBase):
    def __init__(self, node):
        super(Shape, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16]
        for output in self.node.output:
            tensor_type_map[output] = find_key('int64')
        return tensor_type_map

class Constant(OpBase):
    def __init__(self, node):
        super(Constant, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for attr in self.node.attribute:
            data_type = str(onnx.numpy_helper.to_array(attr.t).dtype)
        for output in self.node.output:
            tensor_type_map[output] = find_key(data_type)
        return tensor_type_map

class Squeeze(OpBase):
    def __init__(self, node):
        super(Squeeze, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        if len(in_type) == 2:
            assert in_type[1] == 7
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map
    
class Unsqueeze(OpBase):
    def __init__(self, node):
        super(Unsqueeze, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class ArgMax(OpBase):
    def __init__(self, node):
        super(ArgMax, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 3, 6, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class ArgMin(OpBase):
    def __init__(self, node):
        super(ArgMin, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

op_map = {
          'Concat': Concat,
          'Shape': Shape,
          'Constant': Constant,
          'Squeeze': Squeeze,
          'Unsqueeze': Unsqueeze,
          'ArgMax': ArgMax,
          'ArgMin': ArgMin,

          'Quant': Quant,
          'Dequant': Dequant,

          'Flatten': Flatten,
          'Gather': Gather,
          'MaxPool': MaxPool,

          'QAdd': QAdd,
          'QMul': QMul,
          'QCat': QCat,
          'QAvgPool1d': QAvgPool,
          'QAvgPool2d': QAvgPool,
          'QConv1d': QConv,
          'QConv2d': QConv,
          'QConvBN1d': QConv,
          'QConvBN2d': QConv,
          'QConvTranspose1d': QConvTranspose,
          'QConvTranspose2d': QConvTranspose,
          'QBatchNorm1d': QBatchNorm,
          'QBatchNorm2d': QBatchNorm,
          'QLayerNorm': QLayerNorm,
          'QLinear': QLinear,
          'QLSTM': QLSTM,
          'QGRU': QGRU,
          'QBmm': QBmm,
          'QGLU': QGLU,
          'QSigmoid': QSigmoid,
          'QSoftmax': QSoftmax,
          'QTanh': QTanh,

          'Relu': Relu,
          'Reshape': Reshape,
          'Slice': Slice,
          'Split': Split,
          'Transpose': Transpose,
          'Others': OpBase
          }


def infer_type(model):
    nodes = model.graph.node
    tensor_type_map = {}
    for inp in model.graph.input:
        tensor_type_map[inp.name] = inp.type.tensor_type.elem_type
    for inp in model.graph.initializer:
        tensor_type_map[inp.name] = inp.data_type
    for node in nodes:
        try:
            if node.op_type == 'LSTM':
                in_type = [tensor_type_map[node.input[0]]]
            elif node.op_type == 'Clip':
                in_type = [tensor_type_map[node.input[0]]]
            elif node.op_type == 'GRU':
                in_type = [tensor_type_map[node.input[0]],
                           tensor_type_map[node.input[1]], tensor_type_map[node.input[2]]]
            else:
                in_type = [tensor_type_map[inp] for inp in node.input]
            it = None
            if node.op_type in op_map.keys():
                it = op_map[node.op_type](node)

            if it is not None:
                if node.op_type == 'If':
                    tensor_type_map.update(
                        it.infer_type(in_type, tensor_type_map))
                else:
                    tensor_type_map.update(it.infer_type(in_type))
            else:
                type_dict = {}
                for attr in node.attribute:
                    if attr.name == 'fusion_output_types':
                        assert len(attr.ints) == len(
                            node.output), 'fusion_output_types is not equal to outputs'
                        for name, type_i in zip(node.output, attr.ints):
                            type_dict[name] = type_i
                        break
                if type_dict:
                    tensor_type_map.update(type_dict)
                else:
                    it = op_map['Others'](node)
                    tensor_type_map.update(it.infer_type(in_type))
                    print("Warning: InferType OP ", node.op_type,
                          " is not supported,this may cause error !")

        except AssertionError as e:
            print("The "+node.op_type+"'s input_type has an error")
            raise

    return tensor_type_map
