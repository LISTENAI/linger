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


class ScopedEnter(OpBase):
    def __init__(self, node):
        super(ScopedEnter, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class ScopedLeave(OpBase):
    def __init__(self, node):
        super(ScopedLeave, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class TransposeMatMul(OpBase):
    def __init__(self, node):
        super(TransposeMatMul, self).__init__(node)

    def infer_type(self, in_type):
        assert in_type[0] == 1
        assert in_type[1] == 1
        tensor_type_map = {}
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ScaledTanh(OpBase):
    def __init__(self, node):
        super(ScaledTanh, self).__init__(node)

    def infer_type(self, in_type):
        assert in_type[0] == 1
        tensor_type_map = {}
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Scaler(OpBase):
    def __init__(self, node):
        super(Scaler, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 6, 7, 11]
        tensor_type_map[self.node.output[0]] = find_key('float32')
        return tensor_type_map


class Scale(OpBase):
    def __init__(self, node):
        super(Scale, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class SampleOp(OpBase):
    def __init__(self, node):
        super(SampleOp, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Rfft(OpBase):
    def __init__(self, node):
        super(Rfft, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Cast(OpBase):
    def __init__(self, node):
        super(Cast, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for attr in self.node.attribute:
            data_type = attr.i
        for output in self.node.output:
            tensor_type_map[output] = data_type
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


class IQAdd(OpBase):
    def __init__(self, node):
        super(IQAdd, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert (in_type[0] in [3, 6]) and (in_type[1] in [1, 3, 6])
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map

class iqVar(OpBase):
    def __init__(self, node):
        super(iqVar, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert (in_type[0] in [3])
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class IQSum(OpBase):
    def __init__(self, node):
        super(IQSum, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert (in_type[0] in [1, 3, 6])
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class IQDiv(OpBase):
    def __init__(self, node):
        super(IQDiv, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert (in_type[0] in [3]) and (in_type[1] in [1, 3])
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class IQMul(OpBase):
    def __init__(self, node):
        super(IQMul, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert (in_type[0] in [2, 3]) and (in_type[1] in [1, 2, 3])
        for output in self.node.output:
            tensor_type_map[output] = 3
        return tensor_type_map


class IQCat(OpBase):
    def __init__(self, node):
        super(IQCat, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] == 3
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class IQClamp(OpBase):
    def __init__(self, node):
        super(IQClamp, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] == 3
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class IQSigmoid(OpBase):
    def __init__(self, node):
        super(IQSigmoid, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] in [3, 5]
        for output in self.node.output:
            # 应当为uint8  (2)  但会导致dequant添加错误  所以暂时修改为int8输出
            tensor_type_map[output] = 3
        return tensor_type_map


class IQSigmoid_Is8_Os8(OpBase):
    def __init__(self, node):
        super(IQSigmoid_Is8_Os8, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] == 3
        for output in self.node.output:
            tensor_type_map[output] = 3
        return tensor_type_map


class IQTanh(OpBase):
    def __init__(self, node):
        super(IQTanh, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] in [3, 5]
        for output in self.node.output:
            tensor_type_map[output] = 3
        return tensor_type_map


class AvgPool2dInt(OpBase):
    def __init__(self, node):
        super(AvgPool2dInt, self).__init__(node)

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


class BatchNorm2dInt(OpBase):
    def __init__(self, node):
        super(BatchNorm2dInt, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 3
        assert in_type[1] == 3 or in_type[1] == 5
        assert in_type[2] == 6
        assert in_type[3] == 1
        assert in_type[4] == 1
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class LayerNormInt(OpBase):
    def __init__(self, node):
        super(LayerNormInt, self).__init__(node)

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


class ReLU(OpBase):
    def __init__(self, node):
        super(ReLU, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 3 or in_type[0] == 1 or in_type[0] == 6
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Conv2dInt(OpBase):
    def __init__(self, node):
        super(Conv2dInt, self).__init__(node)

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


class Conv1dInt(OpBase):
    def __init__(self, node):
        super(Conv1dInt, self).__init__(node)

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


class LSTMInt_Is8_Is64(OpBase):
    def __init__(self, node):
        super(LSTMInt_Is8_Is64, self).__init__(node)

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
        tensor_type_map[self.node.output[2]] = find_key('float32')

        return tensor_type_map


class LSTMInt_Is8_Is64_If32_If32(OpBase):
    def __init__(self, node):
        super(LSTMInt_Is8_Is64_If32_If32, self).__init__(node)

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
        tensor_type_map[self.node.output[2]] = find_key('float32')

        return tensor_type_map


class ConvTranspose2dInt(OpBase):
    def __init__(self, node):
        super(ConvTranspose2dInt, self).__init__(node)

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


class LinearInt(OpBase):
    def __init__(self, node):
        super(LinearInt, self).__init__(node)

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


class LSTMInt(OpBase):
    def __init__(self, node):
        super(LSTMInt, self).__init__(node)

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
        tensor_type_map[self.node.output[2]] = find_key('float32')

        return tensor_type_map


class GRUInt(OpBase):
    def __init__(self, node):
        super(GRUInt, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 3] and in_type[1] == 3 and in_type[2] == 3
        # for output in self.node.output:
        tensor_type_map[self.node.output[0]] = 3
        tensor_type_map[self.node.output[1]] = 1

        return tensor_type_map


class GRUInt_Is8_Is64(OpBase):
    def __init__(self, node):
        super(GRUInt_Is8_Is64, self).__init__(node)

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

        return tensor_type_map


class GRUInt_Is8_Is64_If32(OpBase):
    def __init__(self, node):
        super(GRUInt_Is8_Is64_If32, self).__init__(node)

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


class ReQuant(OpBase):
    def __init__(self, node):
        super(ReQuant, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for idx in range(len(self.node.attribute)):
            if self.node.attribute[idx].name == "o_bits":
                count = idx
                break
        out_type = 0
        if self.node.attribute[count].i == 8:
            out_type = 3
        elif self.node.attribute[count].i == 16:
            out_type = 5
        elif self.node.attribute[count].i == 32:
            out_type = 6
        for output in self.node.output:
            if out_type != 0:
                tensor_type_map[output] = out_type
            else:
                tensor_type_map[output] = find_key('int32')
        return tensor_type_map


class OnnxInferReQuant(OpBase):
    def __init__(self, node):
        super(OnnxInferReQuant, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for idx in range(len(self.node.attribute)):
            if self.node.attribute[idx].name == "bit_dst":
                count = idx
                break
        out_type = 0
        if self.node.attribute[count].i == 8:
            out_type = 3
        for output in self.node.output:
            if out_type != 0:
                tensor_type_map[output] = out_type
            else:
                tensor_type_map[output] = find_key('int32')
        return tensor_type_map


class IdentityInfer(OpBase):
    def __init__(self, node):
        super(IdentityInfer, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
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


class Abs(OpBase):
    def __init__(self, node):
        super(Abs, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 11, 12, 13]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Acos(OpBase):
    def __init__(self, node):
        super(Acos, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1   # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Acosh(OpBase):
    def __init__(self, node):
        super(Acosh, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1   # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Add(OpBase):
    def __init__(self, node):
        super(Add, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 7, 10, 11, 12, 13]
        # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [1, 6, 7, 10, 11, 12, 13]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class And(OpBase):
    def __init__(self, node):
        super(And, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 9
        assert in_type[1] == 9
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ArgMax(OpBase):
    def __init__(self, node):
        super(ArgMax, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 10, 11]
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


class Asin(OpBase):
    def __init__(self, node):
        super(Asin, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Asinh(OpBase):
    def __init__(self, node):
        super(Asinh, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Atan(OpBase):
    def __init__(self, node):
        super(Atan, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Atanh(OpBase):
    def __init__(self, node):
        super(Atanh, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class AveragePool(OpBase):
    def __init__(self, node):
        super(AveragePool, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class BatchNormalization(OpBase):
    def __init__(self, node):
        super(BatchNormalization, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Sub(OpBase):
    def __init__(self, node):
        super(Sub, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 6, 7, 10, 11, 12, 13]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Ceil(OpBase):
    def __init__(self, node):
        super(Ceil, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Celu(OpBase):
    def __init__(self, node):
        super(Celu, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] == 1
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Clip(OpBase):
    def __init__(self, node):
        super(Clip, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 2, 3, 7, 11, 13]
        # Determine whether the types are the same
        assert len(set(in_type)) == 1
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Compress(OpBase):
    def __init__(self, node):
        super(Compress, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        assert in_type[1] == 9
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


class Conv(OpBase):
    def __init__(self, node):
        super(Conv, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ConvTranspose(OpBase):
    def __init__(self, node):
        super(ConvTranspose, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Cos(OpBase):
    def __init__(self, node):
        super(Cos, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] == 1
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Cosh(OpBase):
    def __init__(self, node):
        super(Cosh, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] == 1
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class CumSum(OpBase):
    def __init__(self, node):
        super(CumSum, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 7, 10, 11, 12, 13]
        assert in_type[1] in [6, 7]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class DepthToSpace(OpBase):
    def __init__(self, node):
        super(DepthToSpace, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Det(OpBase):
    def __init__(self, node):
        super(Det, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Div(OpBase):
    def __init__(self, node):
        super(Div, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 7, 11, 10, 12, 13]
        # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [1, 6, 7, 11, 10, 12, 13]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Einsum(OpBase):
    def __init__(self, node):
        super(Einsum, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 7, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Elu(OpBase):
    def __init__(self, node):
        super(Elu, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Equal(OpBase):
    def __init__(self, node):
        super(Equal, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 7, 9]
        # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [1, 6, 7, 9]
        for output in self.node.output:
            tensor_type_map[output] = find_key('bool')
        return tensor_type_map


class Erf(OpBase):
    def __init__(self, node):
        super(Erf, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Exp(OpBase):
    def __init__(self, node):
        super(Exp, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Expand(OpBase):
    def __init__(self, node):
        super(Expand, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        assert in_type[1] == 7
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
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


class Floor(OpBase):
    def __init__(self, node):
        super(Floor, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class FixLength(OpBase):
    def __init__(self, node):
        super(FixLength, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] != 0 and in_type[1] != 0
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


class MaskedGather(OpBase):
    def __init__(self, node):
        super(MaskedGather, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] != 0
        if len(in_type) == 3:
            assert in_type[0] in [1, 3, 6, 10]
            assert in_type[1] in [6, 7]
            assert in_type[2] == 6
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class GatherElements(OpBase):
    def __init__(self, node):
        super(GatherElements, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [6, 7]  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class GatherND(OpBase):
    def __init__(self, node):
        super(GatherND, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        assert in_type[1] == 7  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class GenExcit(OpBase):
    def __init__(self, node):
        super(GenExcit, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] != 0
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Gemm(OpBase):
    def __init__(self, node):
        super(Gemm, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class GlobalAveragePool(OpBase):
    def __init__(self, node):
        super(GlobalAveragePool, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class GlobalAveragePoolMask(OpBase):
    def __init__(self, node):
        super(GlobalAveragePoolMask, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class GlobalLpPool(OpBase):
    def __init__(self, node):
        super(GlobalLpPool, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class GlobalMaxPool(OpBase):
    def __init__(self, node):
        super(GlobalMaxPool, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ConvInteger(OpBase):
    def __init__(self, node):
        super(ConvInteger, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 2  # corresponding with 1.7 onnxruntime doc
        assert in_type[1] == 2
        if len(in_type) > 2:
            assert in_type[2] == 2
        if len(in_type) == 4:
            assert in_type[3] == 2
        for output in self.node.output:
            tensor_type_map[output] = find_key('int32')
        return tensor_type_map


class ConvTranspose2dInteger(OpBase):
    def __init__(self, node):
        super(ConvTranspose2dInteger, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] == 3
        for output in self.node.output:
            tensor_type_map[output] = find_key('int32')
        return tensor_type_map


class Greater(OpBase):
    def __init__(self, node):
        super(Greater, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 6, 7, 10, 11, 12, 13]
        for output in self.node.output:
            tensor_type_map[output] = find_key('bool')
        return tensor_type_map


class GreaterOrEqual(OpBase):
    def __init__(self, node):
        super(GreaterOrEqual, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] in [2, 4, 12, 13, 3, 5, 6, 7, 1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = find_key('bool')
        return tensor_type_map


class Identity(OpBase):
    def __init__(self, node):
        super(Identity, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class InstanceNormalization(OpBase):
    def __init__(self, node):
        super(InstanceNormalization, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class IsInf(OpBase):
    def __init__(self, node):
        super(IsInf, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 11]
        for output in self.node.output:
            tensor_type_map[output] = find_key('bool')
        return tensor_type_map


class IsNaN(OpBase):
    def __init__(self, node):
        super(IsNaN, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10]  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = find_key('bool')
        return tensor_type_map


class LRN(OpBase):
    def __init__(self, node):
        super(LRN, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Less(OpBase):
    def __init__(self, node):
        super(Less, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 7, 10, 11, 12, 13]
        assert in_type[1] == in_type[0]
        for output in self.node.output:
            tensor_type_map[output] = find_key('bool')
        return tensor_type_map


class LessOrEqual(OpBase):
    def __init__(self, node):
        super(LessOrEqual, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [2, 4, 12, 13, 3, 5, 6, 7, 1, 10, 11]
        assert in_type[1] in [2, 4, 12, 13, 3, 5, 6, 7, 1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = find_key('bool')
        return tensor_type_map


class Log(OpBase):
    def __init__(self, node):
        super(Log, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class LogSoftmax(OpBase):
    def __init__(self, node):
        super(LogSoftmax, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 11]  # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class LpNormalization(OpBase):
    def __init__(self, node):
        super(LpNormalization, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1   # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class LpPool(OpBase):
    def __init__(self, node):
        super(LpPool, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1   # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class MatMul(OpBase):
    def __init__(self, node):
        super(MatMul, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11, 6, 7, 12, 13]
        # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [1, 10, 11, 6, 7, 12, 13]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class MatMulInteger(OpBase):
    def __init__(self, node):
        super(MatMulInteger, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [2, 3]  # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [2, 3]
        if len(in_type) > 2:
            assert in_type[2] in [2, 3]
        if len(in_type) == 4:
            assert in_type[3] in [2, 3]
        for output in self.node.output:
            tensor_type_map[output] = find_key('int32')
        return tensor_type_map


class Max(OpBase):
    def __init__(self, node):
        super(Max, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 11, 10, 6, 7, 12, 13]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class iqMax(OpBase):
    def __init__(self, node):
        super(iqMax, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc #add int8 input support
            assert in_type[index] in [1, 3, 11, 10, 6, 7, 12, 13]
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


class Mean(OpBase):
    def __init__(self, node):
        super(Mean, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1  # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class MeanVarianceNormalization(OpBase):
    def __init__(self, node):
        super(MeanVarianceNormalization, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1    # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class MaskToLength(OpBase):
    def __init__(self, node):
        super(MaskToLength, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 6, 7, 10]
        tensor_type_map[self.node.output[0]] = 6
        return tensor_type_map


class Min(OpBase):
    def __init__(self, node):
        super(Min, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 6, 7, 10, 11, 12, 13]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Mod(OpBase):
    def __init__(self, node):
        super(Mod, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [2, 4, 12, 13, 3, 5, 6, 7, 1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Mul(OpBase):
    def __init__(self, node):
        super(Mul, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 6, 7, 10, 11, 12, 13]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Neg(OpBase):
    def __init__(self, node):
        super(Neg, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 3, 5, 6, 7, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class NonMaxSuppression(OpBase):
    def __init__(self, node):
        super(NonMaxSuppression, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        assert in_type[1] == 1
        if len(in_type) > 2:   # corresponding with 1.7 onnxruntime doc
            assert in_type[2] == 7
            if len(in_type) > 3:
                assert in_type[3] == 1
                if len(in_type) == 5:
                    assert in_type[4] == 1
        tensor_type_map[self.node.output[0]] = find_key('int64')
        return tensor_type_map


class NonZero(OpBase):
    def __init__(self, node):
        super(NonZero, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 2, 6, 7, 9]
        tensor_type_map[self.node.output[0]] = find_key('int64')
        return tensor_type_map


class Not(OpBase):
    def __init__(self, node):
        super(Not, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 9
        tensor_type_map[self.node.output[0]] = find_key('bool')
        return tensor_type_map


class OneHot(OpBase):
    def __init__(self, node):
        super(OneHot, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 7]
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 7]
        # corresponding with 1.7 onnxruntime doc
        assert in_type[2] in [1, 6, 7, 8]
        tensor_type_map[self.node.output[0]] = in_type[2]
        return tensor_type_map


class ConstantOfShape(OpBase):
    def __init__(self, node):
        super(ConstantOfShape, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 7
        if not self.node.attribute:
            for output in self.node.output:
                tensor_type_map[output] = find_key('float32')
        else:
            for attr in self.node.attribute:
                data_type = str(onnx.numpy_helper.to_array(attr.t).dtype)
            for output in self.node.output:
                tensor_type_map[output] = find_key(data_type)
        return tensor_type_map


class DequantizeLinear(OpBase):
    def __init__(self, node):
        super(DequantizeLinear, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [2, 3]  # corresponding with 1.7 onnxruntime doc
        assert in_type[1] == 1
        if len(in_type) == 3:
            # corresponding with 1.7 onnxruntime doc
            assert in_type[2] in [2, 3]
        for output in self.node.output:
            tensor_type_map[output] = find_key('float32')
        return tensor_type_map


class Dropout(OpBase):
    def __init__(self, node):
        super(Dropout, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        if len(in_type) > 1:
            # corresponding with 1.7 onnxruntime doc
            assert in_type[1] in [1, 10, 11]
        if len(in_type) == 3:
            assert in_type[2] == 9
        tensor_type_map[self.node.output[0]] = in_type[0]
        if len(self.node.output) == 2:
            tensor_type_map[self.node.output[1]] = find_key('bool')
        return tensor_type_map


class GRU(OpBase):
    def __init__(self, node):
        super(GRU, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10, 11] and in_type[1] in [1, 10, 11] and in_type[2] in [
            1, 10, 11]    # corresponding with 1.7 onnxruntime doc
        if len(in_type) > 3:
            assert in_type[3] in [1, 10, 11]
        if len(in_type) > 4:
            assert in_type[4] == 6
        if len(in_type) == 6:
            assert in_type[5] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class If(OpBase):
    def __init__(self, node):
        super(If, self).__init__(node)

    def infer_type(self, in_type, tensor_type_map):
        assert in_type[0] == 9
        if self.node.attribute[0].g.output[0].type.tensor_type.elem_type == 0:
            for node in self.node.attribute[0].g.node:
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
                    else:
                        print("Warning: InferType OP ", node.op_type,
                              " is not supported,this may cause error !")
                        it = op_map['Others'](node)
                    tensor_type_map.update(it.infer_type(in_type))

                except AssertionError as e:
                    print("The "+node.op_type+"'s input_type has an error")
                    raise
            data_type = tensor_type_map[self.node.attribute[0].g.output[0].name]
        else:
            data_type = self.node.attribute[0].g.output[0].type.tensor_type.elem_type

        tensor_type_map = {}
        for output in self.node.output:
            tensor_type_map[output] = data_type
        return tensor_type_map


class LSTM(OpBase):
    def __init__(self, node):
        super(LSTM, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        assert in_type[1] in [1, 10, 11]
        assert in_type[2] in [1, 10, 11]
        if len(in_type) > 3:
            assert in_type[3] == in_type[0]
        if len(in_type) > 4:
            assert in_type[4] == in_type[0]
        if len(in_type) > 5:
            assert in_type[5] == 6
        if len(in_type) > 6:
            assert in_type[6] == in_type[0]
        if len(in_type) == 8:
            assert in_type[7] == in_type[0]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Loop(OpBase):
    def __init__(self, node):
        super(Loop, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        if len(in_type) == 1:  # corresponding with 1.7 onnxruntime doc
            assert in_type[0] == 7
        if len(in_type) > 1:
            assert in_type[1] == 9
        if len(in_type) > 2:
            assert in_type[2] in [1, 2, 3, 4, 5,
                                  6, 7, 8, 9, 10, 11, 12, 13, 16]
        for output in self.node.output:
            tensor_type_map[output] = in_type[2]
        return tensor_type_map


class Multinomial(OpBase):
    def __init__(self, node):
        super(Multinomial, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1  # corresponding with 1.7 onnxruntime doc
        if not self.node.attribute:
            data_type = find_key('int32')
        else:
            for attr in self.node.attribute:
                data_type = attr.i
                break
        for output in self.node.output:
            tensor_type_map[output] = data_type
        return tensor_type_map


class Or(OpBase):
    def __init__(self, node):
        super(Or, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 9
        assert in_type[1] == 9
        for output in self.node.output:
            tensor_type_map[output] = find_key('bool')
        return tensor_type_map


class PRelu(OpBase):
    def __init__(self, node):
        super(PRelu, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Pad(OpBase):
    def __init__(self, node):
        super(Pad, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 2, 3, 6, 7, 10, 11, 12, 13]
        if len(in_type) == 3:
            assert in_type[1] == 7
            assert in_type[2] == in_type[0]  # input[2] :optional 与intype[0]须一致

        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Pow(OpBase):
    def __init__(self, node):
        super(Pow, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 7, 11]
        assert in_type[1] in [1, 6, 7, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class QLinearConv(OpBase):
    def __init__(self, node):
        super(QLinearConv, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            if index in [1, 4, 6]:
                assert in_type[index] == 1
            else:
                assert in_type[index] == 2
                if len(in_type) == 9:   # corresponding with 1.7 onnxruntime doc
                    assert in_type[8] == 6
        for output in self.node.output:
            tensor_type_map[output] = in_type[7]
        return tensor_type_map


class QLinearMatMul(OpBase):
    def __init__(self, node):
        super(QLinearMatMul, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            if index in [1, 4, 6]:
                assert in_type[index] == 1
            else:
                # corresponding with 1.7 onnxruntime doc
                assert in_type[index] == 2
        for output in self.node.output:
            tensor_type_map[output] = in_type[7]
        return tensor_type_map


class QuantizeLinear(OpBase):
    def __init__(self, node):
        super(QuantizeLinear, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10]
        assert in_type[1] in [1, 10]
        if len(in_type) == 3:  # corresponding with 1.7 onnxruntime doc
            assert in_type[2] in [2, 3]
        for output in self.node.output:
            tensor_type_map[output] = in_type[2]
        return tensor_type_map


class RNN(OpBase):
    def __init__(self, node):
        super(RNN, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            if len(in_type) >= 5 and index == 4:
                assert in_type[index] == 6
            else:
                # corresponding with 1.7 onnxruntime doc
                assert in_type[index] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class RandomNormal(OpBase):
    def __init__(self, node):
        super(RandomNormal, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        if not self.node.attribute:
            data_type = find_key('float32')
        else:
            for attr in self.node.attribute:
                data_type = attr.i
                break
        for output in self.node.output:
            tensor_type_map[output] = data_type
        return tensor_type_map


class RandomNormalLike(OpBase):
    def __init__(self, node):
        super(RandomNormalLike, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        if not self.node.attribute:
            data_type = in_type[0]
        else:
            for attr in self.node.attribute:
                if attr.name == 'dtype':
                    data_type = attr.i
        for output in self.node.output:
            tensor_type_map[output] = data_type
        return tensor_type_map


class RandomUniform(OpBase):
    def __init__(self, node):
        super(RandomUniform, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        if not self.node.attribute:
            data_type = find_key('float32')
        else:
            for attr in self.node.attribute:
                data_type = attr.i
                break
        for output in self.node.output:
            tensor_type_map[output] = data_type
        return tensor_type_map


class RandomUniformLike(OpBase):
    def __init__(self, node):
        super(RandomUniformLike, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5,
                              6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        if not self.node.attribute:
            data_type = in_type[0]
        else:
            for attr in self.node.attribute:
                data_type = attr.i
                break
        for output in self.node.output:
            tensor_type_map[output] = data_type
        return tensor_type_map


class Range(OpBase):
    def __init__(self, node):
        super(Range, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] in [1, 5, 6, 7, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Reciprocal(OpBase):
    def __init__(self, node):
        super(Reciprocal, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ReduceL1(OpBase):
    def __init__(self, node):
        super(ReduceL1, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 6, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ReduceL2(OpBase):
    def __init__(self, node):
        super(ReduceL2, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 6, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ReduceLogSum(OpBase):
    def __init__(self, node):
        super(ReduceLogSum, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 6, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ReduceLogSumExp(OpBase):
    def __init__(self, node):
        super(ReduceLogSumExp, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 6, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ReduceMax(OpBase):
    def __init__(self, node):
        super(ReduceMax, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 2, 3, 6, 7, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ReduceMean(OpBase):
    def __init__(self, node):
        super(ReduceMean, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 6, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class MaskedReduceMean(OpBase):
    def __init__(self, node):
        super(MaskedReduceMean, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            assert in_type[index] in [1, 10, 11, 16, 6, 7, 12, 13]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ReduceMin(OpBase):
    def __init__(self, node):
        super(ReduceMin, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 2, 3, 6, 7, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ReduceProd(OpBase):
    def __init__(self, node):
        super(ReduceProd, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for index in range(len(in_type)):
            # corresponding with 1.7 onnxruntime doc
            assert in_type[index] in [1, 6, 7, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ReduceSum(OpBase):
    def __init__(self, node):
        super(ReduceSum, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 7, 10, 11]
        if len(in_type) > 1:
            assert in_type[1] == 7
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ReduceSumSquare(OpBase):
    def __init__(self, node):
        super(ReduceSumSquare, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


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


class ShuffleChannel(OpBase):
    def __init__(self, node):
        super(ShuffleChannel, self).__init__(node)

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


class Resize(OpBase):
    def __init__(self, node):
        super(Resize, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        if self.node.domain == "thinker":
            assert in_type[0] in [1, 6, 2, 3]
        else:
            # remove int8 input  对齐onnxruntime 1.7.0 版本
            assert in_type[0] in [1, 6, 2]
        if len(in_type) > 1:     # corresponding with 1.7 onnxruntime doc
            assert in_type[1] in [1, 10, 11]
        if len(in_type) > 2:
            assert in_type[2] == 1
        if len(in_type) == 4:
            assert in_type[3] == 7
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ReverseSequence(OpBase):
    def __init__(self, node):
        super(ReverseSequence, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16]
        assert in_type[1] == 7
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class DurationToAlignment(OpBase):
    def __init__(self, node):
        super(DurationToAlignment, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in onnx_dtype
        for output in self.node.output:
            tensor_type_map[output] = 1
        return tensor_type_map


class Round(OpBase):
    def __init__(self, node):
        super(Round, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class MeanVarianceNormalization(OpBase):
    def __init__(self, node):
        super(MeanVarianceNormalization, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1   # corresponding with 1.7 onnxruntime doc
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class NegativeLogLikelihoodLoss(OpBase):
    def __init__(self, node):
        super(NegativeLogLikelihoodLoss, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10, 11]
        assert in_type[1] in [6, 7]
        if len(in_type) == 3:
            assert in_type[2] in [1, 10, 11]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Scan(OpBase):
    def __init__(self, node):
        super(Scan, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        if len(in_type) == 1:   # corresponding with 1.7 onnxruntime doc
            assert in_type[0] in [1, 2, 3, 4, 5,
                                  6, 7, 8, 9, 10, 11, 12, 13, 16]
        if len(in_type) == 2:
            assert in_type[0] == 7
            assert in_type[1] in [1, 2, 3, 4, 5,
                                  6, 7, 8, 9, 10, 11, 12, 13, 16]
        for output in self.node.output:
            tensor_type_map[output] in [1, 2, 3, 4,
                                        5, 6, 7, 8, 9, 10, 11, 12, 13, 16]
        return tensor_type_map


class ScatterElements(OpBase):
    def __init__(self, node):
        super(ScatterElements, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [6, 7]   # corresponding with 1.7 onnxruntime doc
        assert in_type[2] in [6, 7]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class ScatterND(OpBase):
    def __init__(self, node):
        super(ScatterND, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        assert in_type[1] == 7
        assert in_type[2] == in_type[0]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Selu(OpBase):
    def __init__(self, node):
        super(Selu, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Shrink(OpBase):
    def __init__(self, node):
        super(Shrink, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12,
                              13, 16]  # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Sigmoid(OpBase):
    def __init__(self, node):
        super(Sigmoid, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Sign(OpBase):
    def __init__(self, node):
        super(Sign, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [2, 4, 12, 13, 3, 5, 6, 7, 1, 10, 11, 16]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Sin(OpBase):
    def __init__(self, node):
        super(Sin, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 11]  # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Sinh(OpBase):
    def __init__(self, node):
        super(Sinh, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1   # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Size(OpBase):
    def __init__(self, node):
        super(Size, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
        tensor_type_map[self.node.output[0]] = find_key('int64')
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


class Softmax(OpBase):
    def __init__(self, node):
        super(Softmax, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map

class SoftmaxInt(OpBase):
    def __init__(self, node):
        super(SoftmaxInt, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [3, 5, 6]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map

class LogSoftmaxInt(OpBase):
    def __init__(self, node):
        super(LogSoftmaxInt, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [3, 5, 6]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map

class SoftmaxCrossEntropyLoss(OpBase):
    def __init__(self, node):
        super(SoftmaxCrossEntropyLoss, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10, 11, 16]
        assert in_type[1] in [6, 7]
        if len(in_type) == 3:
            assert in_type[2] in [1, 10, 11, 16]
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Softplus(OpBase):
    def __init__(self, node):
        super(Softplus, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Softsign(OpBase):
    def __init__(self, node):
        super(Softsign, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class SpaceToDepth(OpBase):
    def __init__(self, node):
        super(SpaceToDepth, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1  # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
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


class Sqrt(OpBase):
    def __init__(self, node):
        super(Sqrt, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
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


class StringNormalizer(OpBase):
    def __init__(self, node):
        super(StringNormalizer, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 8
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Sum(OpBase):
    def __init__(self, node):
        super(Sum, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 6, 7, 10, 11, 12, 13]
        for input in in_type:
            assert input == in_type[0]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Tan(OpBase):
    def __init__(self, node):
        super(Tan, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1  # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Tanh(OpBase):
    def __init__(self, node):
        super(Tanh, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class TfIdfVectorizer(OpBase):
    def __init__(self, node):
        super(TfIdfVectorizer, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [6, 7, 8]
        tensor_type_map[self.node.output[0]] = find_key('float32')
        return tensor_type_map


class ThresholdedRelu(OpBase):
    def __init__(self, node):
        super(ThresholdedRelu, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Tile(OpBase):
    def __init__(self, node):
        super(Tile, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11,
                              12, 13]  # corresponding with 1.7 onnxruntime doc
        if len(in_type) == 2:
            assert in_type[1] == 7
        if len(in_type) == 3:
            assert in_type[1] in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
            assert in_type[2] in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class TopK(OpBase):
    def __init__(self, node):
        super(TopK, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16]
        if len(in_type) == 2:
            assert in_type[1] == 7
        tensor_type_map[self.node.output[0]] = in_type[0]
        # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[1]] = 7
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


class Unique(OpBase):
    def __init__(self, node):
        super(Unique, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 16]  # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
        if len(self.node.output) >= 2:
            tensor_type_map[self.node.output[1]] = find_key('int64')
        if len(self.node.output) >= 3:
            tensor_type_map[self.node.output[2]] = find_key('int64')
        if len(self.node.output) >= 4:
            tensor_type_map[self.node.output[3]] = find_key('int64')
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


class Where(OpBase):
    def __init__(self, node):
        super(Where, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 9
        # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [1, 2, 6, 7, 8]
        assert in_type[2] == in_type[1]
        for output in self.node.output:
            tensor_type_map[output] = in_type[1]
        return tensor_type_map


class Xor(OpBase):
    def __init__(self, node):
        super(Xor, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 9
        assert in_type[1] == 9
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Affine(OpBase):
    def __init__(self, node):
        super(Affine, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class CDist(OpBase):
    def __init__(self, node):
        super(CDist, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 11]
        assert in_type[1] in [1, 11]  # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class ComplexMul(OpBase):
    def __init__(self, node):
        super(ComplexMul, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10]
        assert in_type[1] in [1, 10]  # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class ComplexMulConj(OpBase):
    def __init__(self, node):
        super(ComplexMulConj, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10]
        assert in_type[1] in [1, 10]  # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class ConvTransposeWithDynamicPads(OpBase):
    def __init__(self, node):
        super(ConvTransposeWithDynamicPads, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        assert in_type[1] == 1
        assert in_type[2] == 7
        assert in_type[3] == 1
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Crop(OpBase):
    def __init__(self, node):
        super(Crop, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class DynamicQuantizeMatMul(OpBase):
    def __init__(self, node):
        super(DynamicQuantizeMatMul, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        assert in_type[1] in [2, 3]
        assert in_type[2] == 1
        assert in_type[3] == in_type[1]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class DynamicSlice(OpBase):
    def __init__(self, node):
        super(DynamicSlice, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [2, 4, 12, 13, 3, 5, 6, 7, 1, 10, 11, 9, 16]
        assert in_type[1] in [6, 7]
        assert in_type[2] == in_type[1]
        assert in_type[3] == in_type[1]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class ExpandDims(OpBase):
    def __init__(self, node):
        super(ExpandDims, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [2, 4, 12, 13, 3, 5, 6, 7, 1, 10, 11, 8, 9, 16]
        assert in_type[1] == 6
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class FastGelu(OpBase):
    def __init__(self, node):
        super(FastGelu, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10]   # corresponding with 1.7 onnxruntime doc
        assert in_type[1] in [1, 10]   # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class WindowAttentionChunk(OpBase):
    def __init__(self, node):
        super(WindowAttentionChunk, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10]
        assert in_type[1] in [1, 10]
        assert in_type[2] in [1, 10]
        for out in self.node.output:
            tensor_type_map[out] = in_type[0]

        return tensor_type_map


class FeatureVectorizer(OpBase):
    def __init__(self, node):
        super(FeatureVectorizer, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 6, 7, 11]
        tensor_type_map[self.node.output[0]] = list(
            onnx_dtype.keys())[list(onnx_dtype.values()).index('float32')]
        return tensor_type_map


class FusedConv(OpBase):
    def __init__(self, node):
        super(FusedConv, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        assert in_type[1] == 1
        assert in_type[2] == 1
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class FusedGemm(OpBase):
    def __init__(self, node):
        super(FusedGemm, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        assert in_type[1] == 1
        assert in_type[2] == 1
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Gelu(OpBase):
    def __init__(self, node):
        super(Gelu, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class ImageScaler(OpBase):
    def __init__(self, node):
        super(ImageScaler, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Imputer(OpBase):
    def __init__(self, node):
        super(Imputer, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 7]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Inverse(OpBase):
    def __init__(self, node):
        super(Inverse, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Irfft(OpBase):
    def __init__(self, node):
        super(Irfft, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 10, 11]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class LinearClassifier(OpBase):
    def __init__(self, node):
        super(LinearClassifier, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 6, 7, 11]
        assert in_type[1] in [7, 8]
        tensor_type_map[self.node.output[0]] = list(
            onnx_dtype.keys())[list(onnx_dtype.values()).index('float32')]
        return tensor_type_map


class LinearRegressor(OpBase):
    def __init__(self, node):
        super(LinearRegressor, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class MatMulInteger16(OpBase):
    def __init__(self, node):
        super(MatMulInteger16, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 5
        assert in_type[1] == 5
        tensor_type_map[self.node.output[0]] = list(
            onnx_dtype.keys())[list(onnx_dtype.values()).index('int32')]
        return tensor_type_map


class MaxpoolWithMask(OpBase):
    def __init__(self, node):
        super(MaxpoolWithMask, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        assert in_type[1] == 6
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class Normalizer(OpBase):
    def __init__(self, node):
        super(Normalizer, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 6, 7, 11]
        tensor_type_map[self.node.output[0]] = list(
            onnx_dtype.keys())[list(onnx_dtype.values()).index('float32')]
        return tensor_type_map


class LayerNormalization(OpBase):
    def __init__(self, node):
        super(LayerNormalization, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        # corresponding with 1.7 onnxruntime doc
        assert in_type[0] in [1, 10, 11]
        assert in_type[1] == in_type[0]
        assert in_type[2] == in_type[0]
        tensor_type_map[self.node.output[0]] = in_type[0]
        # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[1]] = 1
        # corresponding with 1.7 onnxruntime doc
        tensor_type_map[self.node.output[2]] = 1
        return tensor_type_map


class MaskedLayerNorm(OpBase):
    def __init__(self, node):
        super(MaskedLayerNorm, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 11, 10, 16]
        assert in_type[1] == in_type[0]
        assert in_type[2] == in_type[0]
        assert in_type[3] == in_type[0]
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class ATen(OpBase):
    def __init__(self, node):
        super(ATen, self).__init__(node)
        # print(node.attribute)

    def infer_type(self, in_type):
        tensor_type_map = {}
        op_type = "ATen"
        for attr in self.node.attribute:
            if attr.name == "operator":
                if attr.s == b'var':  # jgtian  aten-var export.07.08
                    assert in_type[0] != 3
                elif attr.s == b'layer_norm':
                    assert in_type[0] != 3
                elif attr.s == b'index':  # support int8 quant input
                    assert in_type[0] in [1, 2, 3, 4, 5, 6,
                                          7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                else:
                    assert in_type[0] in [1, 2, 4, 5, 6, 7, 8, 9, 10,
                                          11, 12, 13, 14, 15, 16]  # remove int8 support
                op_type = attr.s
                break
        tensor_type_map[self.node.output[0]] = in_type[0]
        print('ATen({}): The reasoning may be incorrect'.format(op_type.decode()))
        return tensor_type_map


class HistoryPadding(OpBase):
    def __init__(self, node):
        super(HistoryPadding, self).__init__(node)
        # print(node.attribute)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6,
                              7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        tensor_type_map[self.node.output[0]] = in_type[0]
        tensor_type_map[self.node.output[1]] = in_type[0]
        return tensor_type_map


class StreamPadding(OpBase):
    def __init__(self, node):
        super(StreamPadding, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 2, 3, 4, 5, 6, 7]
        assert in_type[1] in [1, 2, 3, 4, 5, 6, 7]
        assert in_type[2] in [6]
        assert in_type[0] == in_type[1]

        tensor_type_map[self.node.output[0]] = in_type[0]
        tensor_type_map[self.node.output[1]] = in_type[0]
        return tensor_type_map


class AdaptiveAvgPool2d(OpBase):
    def __init__(self, node):
        super(AdaptiveAvgPool2d, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 1
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class ToUint8(OpBase):
    def __init__(self, node):
        super(ToUint8, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] == 3
        tensor_type_map[self.node.output[0]] = 2
        return tensor_type_map


class RNNJoin(OpBase):
    def __init__(self, node):
        super(RNNJoin, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        assert in_type[0] in [1, 3]
        assert in_type[1] == 7
        assert in_type[2] == 7
        assert in_type[3] == 7
        tensor_type_map[self.node.output[0]] = in_type[0]
        return tensor_type_map


class BitwiseOP(OpBase):
    def __init__(self, node):
        super(BitwiseOP, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class Triu(OpBase):
    def __init__(self, node):
        super(Triu, self).__init__(node)

    def infer_type(self, in_type):
        tensor_type_map = {}
        for output in self.node.output:
            tensor_type_map[output] = in_type[0]
        return tensor_type_map


class OnnxInferCalcShape(OpBase):
    def __init__(self, node):
        super(OnnxInferCalcShape, self).__init__(node)

    def infer_type(self, in_type):
        assert in_type[0] == 7
        assert in_type[1] == 7
        assert in_type[2] == 7
        assert in_type[3] == 7

        tensor_type_map = {}
        for output in self.node.output:
            tensor_type_map[output] = 7
        return tensor_type_map


class BmmInt(OpBase):
    def __init__(self, node):
        super(BmmInt, self).__init__(node)

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


op_map = {'Cast': Cast,
          'Shape': Shape,
          'Constant': Constant,
          'Quant': Quant,
          'OnnxInferQuant': Quant,
          'Dequant': Dequant,
          'TransposeMatMul': TransposeMatMul,
          'ScaledTanh': ScaledTanh,
          'Scaler': Scaler,
          'Scale': Scale,
          'SampleOp': SampleOp,
          'Rfft': Rfft,
          'iqAdd': IQAdd,
          'iqMul': IQMul,
          'iqDiv': IQDiv,
          'iqSum': IQSum,
          'iqCat': IQCat,
          'iqClamp': IQClamp,
          'iqSigmoid': IQSigmoid,
          'iqSigmoid_Is8_Os8': IQSigmoid_Is8_Os8,
          'iqTanh': IQTanh,
          'ReLU': ReLU,
          'AvgPool2dInt': AvgPool2dInt,
          'Conv2dInt': Conv2dInt,
          'ConvTranspose2dInt': ConvTranspose2dInt,
          'BatchNorm2dInt': BatchNorm2dInt,
          'LayerNormInt': LayerNormInt,
          'LinearInt': LinearInt,
          'LSTMInt': LSTMInt,
          'GRUInt': GRUInt,
          'Abs': Abs,
          'Acos': Acos,
          'Acosh': Acosh,
          'Add': Add,
          'Affine': Affine,
          'And': And,
          'ArgMax': ArgMax,
          'ArgMin': ArgMin,
          'Asin': Asin,
          'Asinh': Asinh,
          'Atan': Atan,
          'Atanh': Atanh,
          'AveragePool': AveragePool,
          'BatchNormalization': BatchNormalization,
          'CDist': CDist,
          'Ceil': Ceil,
          'Clip': Clip,
          'ComplexMul': ComplexMul,
          'ComplexMulConj': ComplexMulConj,
          'Compress': Compress,
          'Concat': Concat,
          'Conv': Conv,
          'ConvTransposeWithDynamicPads': ConvTransposeWithDynamicPads,
          'ConstantOfShape': ConstantOfShape,
          'ConvInteger': ConvInteger,
          'ConvTranspose': ConvTranspose,
          'Cos': Cos,
          'Cosh': Cosh,
          'CumSum': CumSum,
          'Crop': Crop,
          'DynamicQuantizeMatMul': DynamicQuantizeMatMul,
          'DynamicSlice': DynamicSlice,
          'DepthToSpace': DepthToSpace,
          'DequantizeLinear': DequantizeLinear,
          'Det': Det,
          'Div': Div,
          'Dropout': Dropout,
          'Einsum': Einsum,
          'Elu': Elu,
          'Equal': Equal,
          'Erf': Erf,
          'Exp': Exp,
          'Expand': Expand,
          'Flatten': Flatten,
          'Floor': Floor,
          'FixLength': FixLength,
          'Identity': Identity,
          'GRU': GRU,
          'If': If,
          'Gather': Gather,
          'MaskedGather': MaskedGather,
          'GatherElements': GatherElements,
          'GatherND': GatherND,
          'Gemm': Gemm,
          'GenExcit': GenExcit,
          'GlobalAveragePool': GlobalAveragePool,
          'GlobalLpPool': GlobalLpPool,
          'GlobalMaxPool': GlobalMaxPool,
          'GlobalAveragePoolMask': GlobalAveragePoolMask,
          'MaskedReduceMean': MaskedReduceMean,
          'InstanceNormalization': InstanceNormalization,
          'IsInf': IsInf,
          'IsNaN': IsNaN,
          'LRN': LRN,
          'Less': Less,
          'Log': Log,
          'LSTM': LSTM,
          'Loop': Loop,
          'LogSoftmax': LogSoftmax,
          'LessOrEqual': LessOrEqual,
          'GreaterOrEqual': GreaterOrEqual,
          'Celu': Celu,
          'MaskToLength': MaskToLength,
          'MeanVarianceNormalization': MeanVarianceNormalization,
          'NegativeLogLikelihoodLoss': NegativeLogLikelihoodLoss,
          'LpNormalization': LpNormalization,
          'LpNormalization': LpNormalization,
          'LpPool': LpPool,
          'MatMul': MatMul,
          'MatMulInteger': MatMulInteger,
          'Max': Max,
          'MaxPool': MaxPool,
          'Mean': Mean,
          'MeanVarianceNormalization': MeanVarianceNormalization,
          'Multinomial': Multinomial,
          'Min': Min,
          'Mod': Mod,
          'Mul': Mul,
          'Neg': Neg,
          'NonMaxSuppression': NonMaxSuppression,
          'NonZero': NonZero,
          'Not': Not,
          'OneHot': OneHot,
          'Or': Or,
          'PRelu': PRelu,
          'Pad': Pad,
          'Pow': Pow,
          'QLinearConv': QLinearConv,
          'QLinearMatMul': QLinearMatMul,
          'QuantizeLinear': QuantizeLinear,
          'RNN': RNN,
          'RandomNormal': RandomNormal,
          'RandomNormalLike': RandomNormalLike,
          'RandomUniform': RandomUniform,
          'RandomUniformLike': RandomUniformLike,
          'Range': Range,
          'Reciprocal': Reciprocal,
          'ReduceL1': ReduceL1,
          'ReduceL2': ReduceL2,
          'ReduceLogSum': ReduceLogSum,
          'ReduceLogSumExp': ReduceLogSumExp,
          'ReduceMax': ReduceMax,
          'ReduceMean': ReduceMean,
          'ReduceMin': ReduceMin,
          'ReduceProd': ReduceProd,
          'ReduceSum': ReduceSum,
          'ReduceSumSquare': ReduceSumSquare,
          'Relu': Relu,
          'Reshape': Reshape,
          'Resize': Resize,
          'ReverseSequence': ReverseSequence,
          'Round': Round,
          'Scan': Scan,
          'ScatterElements': ScatterElements,
          'ScatterND': ScatterND,
          'Selu': Selu,
          'Shrink': Shrink,
          'Sigmoid': Sigmoid,
          'Sign': Sign,
          'Sin': Sin,
          'Sinh': Sinh,
          'Size': Size,
          'Slice': Slice,
          'Softmax': Softmax,
          'SoftmaxInt': SoftmaxInt,
          'LogSoftmaxInt': LogSoftmaxInt,
          'SoftmaxCrossEntropyLoss': SoftmaxCrossEntropyLoss,
          'Softplus': Softplus,
          'Softsign': Softsign,
          'SpaceToDepth': SpaceToDepth,
          'Split': Split,
          'Sqrt': Sqrt,
          'Squeeze': Squeeze,
          'StringNormalizer': StringNormalizer,
          'Sub': Sub,
          'Sum': Sum,
          'Tan': Tan,
          'Tanh': Tanh,
          'TfIdfVectorizer': TfIdfVectorizer,
          'ThresholdedRelu': ThresholdedRelu,
          'Tile': Tile,
          'TopK': TopK,
          'Transpose': Transpose,
          'Unique': Unique,
          'Unsqueeze': Unsqueeze,
          'Where': Where,
          'Xor': Xor,
          'ExpandDims': ExpandDims,
          'FastGelu': FastGelu,
          'FeatureVectorizer': FeatureVectorizer,
          'FusedConv': FusedConv,
          'FusedGemm': FusedGemm,
          'Gelu': Gelu,
          'Greater': Greater,
          'ImageScaler': ImageScaler,
          'Imputer': Imputer,
          'Inverse': Inverse,
          'Irfft': Irfft,
          'LinearClassifier': LinearClassifier,
          'LinearRegressor': LinearRegressor,
          'MatMulInteger16': MatMulInteger16,
          'MaxpoolWithMask': MaxpoolWithMask,
          'Normalizer': Normalizer,
          'LayerNormalization': LayerNormalization,
          'MaskedLayerNorm': MaskedLayerNorm,
          'ATen': ATen,
          'HistoryPadding': HistoryPadding,
          'Requant': ReQuant,
          'OnnxInferReQuant': OnnxInferReQuant,
          'ConvTranspose2dInteger': ConvTranspose2dInteger,
          'IdentityInfer': IdentityInfer,
          'ToUint8': ToUint8,
          'RNNJoin': RNNJoin,
          'bitwise_and': BitwiseOP,
          'bitwise_or': BitwiseOP,
          'bitwise_not': BitwiseOP,
          'Triu': Triu,
          'OnnxInferCalcShape': OnnxInferCalcShape,
          'ScopedEnter': ScopedEnter,
          'ScopedLeave': ScopedLeave,
          'Conv1dInt': Conv1dInt,
          'LSTMInt_Is8_Is64_If32_If32': LSTMInt_Is8_Is64_If32_If32,
          'LSTMInt_Is8_Is64': LSTMInt_Is8_Is64,
          'GRUInt_Is8_Is64': GRUInt_Is8_Is64,
          'GRUInt_Is8_Is64_If32': GRUInt_Is8_Is64_If32,
          'BmmInt': BmmInt,
          'StreamPadding': StreamPadding,
          'DurationToAlignment': DurationToAlignment,
          'iqMax': iqMax,
          'iqVar': iqVar,
          'ShuffleChannel': ShuffleChannel,
          'Others': OpBase
          }


def make_node():
    input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

    input_size = 2
    hidden_size = 3
    weight_scale = 0.1
    number_of_gates = 4

    node = onnx.helper.make_node(
        'LSTM',
        inputs=['X', 'W', 'R'],
        outputs=['', 'Y'],
        hidden_size=hidden_size
    )
    shape = LSTM(node)
    out_type = shape.infer_type([1, 1, 1])


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
