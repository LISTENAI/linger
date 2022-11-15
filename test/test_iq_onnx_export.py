import hashlib
import math
import os
from logging import PlaceHolder

import linger
import linger.onnx
import onnx
import torch
import torch.nn
import torch.nn.functional as F
import torch.onnx
from linger.config import config
from linger.ops import (IQTensor, from_torch_tensor, iqadd, iqAddLayer, iqmul,
                        iqMulLayer)
from linger.utils import PlatFormQuant


def get_file_topolo_sort_type_list(f):
    model = onnx.load(f)
    return [n.op_type for n in model.graph.node]


def get_file_md5(fname):
    m = hashlib.md5()
    with open(fname, 'rb') as fobj:
        while True:
            data = fobj.read(4096)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


if not os.path.exists("data.ignore"):
    os.mkdir("data.ignore")


def test_view_export():
    is_tuple = False

    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, *args):
            if is_tuple:
                return args[0].view((1, 4, -1))
            else:
                return args[0].view(1, 4, -1)
    net = TestModel()
    dummy_input = torch.randn(1, 2, 3, 4)
    with torch.no_grad():
        linger.onnx.export(net, dummy_input, "data.ignore/torch_view.onnx", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #original_model = onnx.load("data.ignore/torch_view.onnx.ignore")
    aa_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/torch_view.onnx")
    aa_iq = from_torch_tensor(dummy_input, 12, 3)
    with torch.no_grad():
        linger.onnx.export(net, aa_iq, "data.ignore/iq_view.onnx", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #iq1_model = onnx.load("data.ignore/iq_view.onnx.ignore")
    aa_iq_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/iq_view.onnx")
    assert aa_model_md5 == aa_iq_model_md5

    is_tuple = True
    with torch.no_grad():
        linger.onnx.export(net, aa_iq, "data.ignore/iq_view_tuple.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    aa_iq_tuple_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/iq_view_tuple.onnx.ignore")
    assert aa_model_md5 == aa_iq_tuple_model_md5


def test_view_as_export():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, *args):
            return args[0].view_as(args[1])

    net = TestModel()
    dummy_input = torch.randn(1, 2, 3, 4)
    aa_copy = dummy_input.detach().data

    aa_iq = from_torch_tensor(dummy_input, 12, 3)
    with torch.no_grad():
        linger.onnx.export(net, (aa_iq, aa_copy), "data.ignore/iq_view_as.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', 'y'], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    iq_model = onnx.load("data.ignore/iq_view_as.onnx.ignore")
    assert iq_model.graph.node[1].op_type == 'Reshape'


def test_relu_export():
    inplace = False

    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, *args):
            if inplace:
                return torch.relu(*args)
            else:
                return torch.relu_(*args)
    net = TestModel()
    dummy_input = torch.randn(9, 8, 7, 6)
    with torch.no_grad():
        linger.onnx.export(net, dummy_input, "data.ignore/torch_relu.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #original_model = onnx.load("data.ignore/torch_relu.onnx.ignore")
    base_md5 = get_file_topolo_sort_type_list(
        "data.ignore/torch_relu.onnx.ignore")
    aa_iq = from_torch_tensor(dummy_input, 12, 3)
    with torch.no_grad():
        linger.onnx.export(net, aa_iq, "data.ignore/torch_relu_iq1.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #iq1_model = onnx.load("data.ignore/torch_relu_iq1.onnx.ignore")
    iq1_md5 = get_file_topolo_sort_type_list(
        "data.ignore/torch_relu_iq1.onnx.ignore")
    assert iq1_md5 == base_md5
    inplace = True
    with torch.no_grad():
        linger.onnx.export(net, aa_iq, "data.ignore/torch_relu_iq2.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #iq2_model = onnx.load("data.ignore/torch_relu_iq2.onnx.ignore")
    iq2_md5 = get_file_topolo_sort_type_list(
        "data.ignore/torch_relu_iq2.onnx.ignore")
    assert iq2_md5 == base_md5


def test_maxpool_2d_export():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, *args):
            return torch.max_pool2d(args[0], kernel_size=(2, 2), stride=2, padding=0)
    net = TestModel()
    dummy_input = torch.randn(1, 8, 16)
    with torch.no_grad():
        linger.onnx.export(net, dummy_input, "data.ignore/torch_maxpool_2d.onnx.ignore", export_params=True, opset_version=9,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    mode_base = onnx.load("data.ignore/torch_maxpool_2d.onnx.ignore")

    iq_1 = from_torch_tensor(dummy_input, 11, 8)
    with torch.no_grad():
        linger.onnx.export(net, iq_1, "data.ignore/qi1_maxpool_2d.onnx.ignore", export_params=True, opset_version=9,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    mode_iq = onnx.load("data.ignore/qi1_maxpool_2d.onnx.ignore")
    assert mode_iq.graph.node[0].op_type == mode_base.graph.node[0].op_type
    assert mode_iq.graph.node[0].attribute == mode_base.graph.node[0].attribute


def test_iq_mul_export_onnx():
    x = torch.tensor([[6.0, 4.0]], requires_grad=True).cuda()
    y = torch.tensor([[3.0, 8.0]], requires_grad=True).cuda()
    iq_layer = iqMulLayer()
    a0 = from_torch_tensor(x-1, 127.0/(6.0-1), 8)
    b0 = from_torch_tensor(y-1, 127.0/(8-1), 8)
    oscale = 127.0/(12-2)
    iq_layer(a0, b0, oscale)
    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)
    scale = 127.0 / 12
    with torch.no_grad():
        linger.onnx.export(iq_layer, (a, b, scale), "data.ignore/iq_mul1.onnx", export_params=True, keep_initializers_as_inputs=False,
                           opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    m_float = onnx.load("data.ignore/iq_mul1.onnx")
    assert m_float.graph.node[0].op_type == 'iqMul'
    assert len(m_float.graph.node[0].attribute) == 4
    for m in m_float.graph.node[0].attribute:
        if m.name == "scale_o":
            assert m.f == 128
        if m.name == "scale_x":
            assert abs(m.f - 127.0/6) < 0.01
        if m.name == "scale_y":
            assert abs(m.f - 127.0/8) < 0.01


def test_iq_mul_module_export_onnx():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, x, y):
            return iqmul(self, x, y, 'testname')
    x = torch.tensor([[6.0, 4.0]], requires_grad=True).cuda()
    y = torch.tensor([[3.0, 8.0]], requires_grad=True).cuda()
    net = TestModel()
    a0 = from_torch_tensor(x-1, 127.0/(6.0-1), 8)
    b0 = from_torch_tensor(y-1, 127.0/(8-1), 8)

    net(a0, b0)
    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)

    with torch.no_grad():
        linger.onnx.export(net, (a, b), "data.ignore/iq_mul2.onnx", export_params=True, keep_initializers_as_inputs=False,
                           opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    m_float = onnx.load("data.ignore/iq_mul2.onnx")
    assert m_float.graph.node[0].op_type == 'iqMul'
    assert len(m_float.graph.node[0].attribute) == 4
    assert len(m_float.graph.node) == 2
    for m in m_float.graph.node[0].attribute:
        if m.name == "scale_o":
            max_value = round(math.log(127/2.1, 2))
            scale_local = math.pow(2, max_value)
            assert abs(m.f - scale_local) < 0.1
        if m.name == "scale_x":
            assert abs(m.f - 127.0/6) < 0.01
        if m.name == "scale_y":
            assert abs(m.f - 127.0/8) < 0.01


def test_iq_mul_u8i8_module_export_onnx():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, x, y):
            return iqmul(self, x, y, 'testname')
    x = torch.tensor([[6.0, 4.0]], requires_grad=True).cuda()
    y = torch.tensor([[3.0, 8.0]], requires_grad=True).cuda()
    net = TestModel()
    a0 = from_torch_tensor(x-1, 127.0/(6.0-1), 8, 128)
    b0 = from_torch_tensor(y-1, 127.0/(8-1), 8)

    net(a0, b0)
    a = from_torch_tensor(x, 127.0/6.0, 8, 128)
    b = from_torch_tensor(y, 127.0/8, 8)

    with torch.no_grad():
        linger.onnx.export(net, (a, b), "data.ignore/iq_mul3.onnx", export_params=True, keep_initializers_as_inputs=False,
                           opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    m_float = onnx.load("data.ignore/iq_mul3.onnx")
    assert m_float.graph.node[0].op_type == 'iqMul'


def test_iq_add_export_onnx():
    linger.SetPlatFormQuant(platform_quant=PlatFormQuant.luna_quant)
    x = torch.tensor([[6.0, 4.0]], requires_grad=True).cuda()
    y = torch.tensor([[3.0, 8.0]], requires_grad=True).cuda()
    iq_layer = iqAddLayer()
    a0 = from_torch_tensor(x-1, 127.0/(6.0-1), 8)
    b0 = from_torch_tensor(y-1, 127.0/(8-1), 8)
    oscale = 127.0/(12-2)
    iq_layer(a0, b0, oscale)
    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)
    scale = torch.tensor([127.0/12])
    with torch.no_grad():
        linger.onnx.export(iq_layer, (a, b, scale), "data.ignore/iq_add1.onnx.ignore", export_params=True,
                           keep_initializers_as_inputs=False, opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    m_float = onnx.load("data.ignore/iq_add1.onnx.ignore")
    assert m_float.graph.node[0].op_type == 'iqAdd'
    assert len(m_float.graph.node[0].attribute) == 5
    for m in m_float.graph.node[0].attribute:
        if m.name == "scale_o":
            assert m.f == 128
        if m.name == "scale_x":
            assert abs(m.f - 127.0/6) < 0.01
        if m.name == "scale_y":
            assert abs(m.f - 127.0/8) < 0.01


def test_iq_add_module_export_onnx():
    linger.SetPlatFormQuant(platform_quant=PlatFormQuant.luna_quant)

    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, x, y):
            return iqadd(self, x, y, 'testname')
    x = torch.tensor([[6.0, 4.0]], requires_grad=True).cuda()
    y = torch.tensor([[3.0, 8.0]], requires_grad=True).cuda()
    net = TestModel()
    a0 = from_torch_tensor(x-1, 127.0/(6.0-1), 8)
    b0 = from_torch_tensor(y-1, 127.0/(8-1), 8)
    oscale = 127.0/(12-2)
    net(a0, b0)
    a = from_torch_tensor(x, 127.0/6.0, 8)
    b = from_torch_tensor(y, 127.0/8, 8)
    scale = torch.tensor([127.0/12])
    with torch.no_grad():
        linger.onnx.export(net, (a, b), "data.ignore/iq_add2.onnx.ignore", export_params=True, keep_initializers_as_inputs=False,
                           opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    m_float = onnx.load("data.ignore/iq_add2.onnx.ignore")
    assert m_float.graph.node[0].op_type == 'iqAdd'
    assert len(m_float.graph.node[0].attribute) == 5
    assert len(m_float.graph.node) == 2
    for m in m_float.graph.node[0].attribute:
        if m.name == "scale_o":
            assert m.f == 128.0
        if m.name == "scale_x":
            assert abs(m.f - 127.0/6) < 0.01
        if m.name == "scale_y":
            assert abs(m.f - 127.0/8) < 0.01


def test_reshape_export():
    is_tuple = False

    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, *args):
            if is_tuple:
                return args[0].reshape((1, 4, -1))
            else:
                return args[0].reshape(1, 4, -1)
    net = TestModel()
    dummy_input = torch.randn(1, 2, 3, 4)
    with torch.no_grad():
        linger.onnx.export(net, dummy_input, "data.ignore/torch_reshape.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #original_model = onnx.load("data.ignore/torch_view.onnx.ignore")
    aa_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/torch_reshape.onnx.ignore")
    aa_iq = from_torch_tensor(dummy_input, 12, 3)
    with torch.no_grad():
        linger.onnx.export(net, aa_iq, "data.ignore/iq_reshape.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #iq1_model = onnx.load("data.ignore/iq_view.onnx.ignore")
    aa_iq_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/iq_reshape.onnx.ignore")
    assert aa_model_md5 == aa_iq_model_md5

    is_tuple = True
    with torch.no_grad():
        linger.onnx.export(net, aa_iq, "data.ignore/iq_reshape_tuple.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    aa_iq_tuple_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/iq_reshape_tuple.onnx.ignore")
    assert aa_model_md5 == aa_iq_tuple_model_md5


def test_reshape_as_export():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, *args):
            return args[0].reshape_as(args[1])

    net = TestModel()
    dummy_input = torch.randn(1, 2, 3, 4)
    aa_copy = dummy_input.detach().data

    aa_iq = from_torch_tensor(dummy_input, 12, 3)
    with torch.no_grad():
        linger.onnx.export(net, (aa_iq, aa_copy), "data.ignore/iq_reshape_as.onnx.ignore", export_params=True,
                           opset_version=11, input_names=['x', 'y'], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    iq_model = onnx.load("data.ignore/iq_reshape_as.onnx.ignore")
    assert iq_model.graph.node[1].op_type == 'Reshape'


def test_squeeze_export():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, *args):
            return args[0].squeeze()
    net = TestModel()
    dummy_input = torch.randn(1, 2, 3, 4)
    with torch.no_grad():
        linger.onnx.export(net, dummy_input, "data.ignore/torch_squeeze.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #original_model = onnx.load("data.ignore/torch_view.onnx.ignore")
    aa_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/torch_squeeze.onnx.ignore")
    aa_iq = from_torch_tensor(dummy_input, 12, 3)
    with torch.no_grad():
        linger.onnx.export(net, aa_iq, "data.ignore/iq_squeeze.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #iq1_model = onnx.load("data.ignore/iq_view.onnx.ignore")
    aa_iq_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/iq_squeeze.onnx.ignore")
    assert aa_model_md5 == aa_iq_model_md5


def test_unsqueeze_export():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, *args):
            return args[0].unsqueeze(dim=3)
    net = TestModel()
    dummy_input = torch.randn(1, 2, 3, 5)
    with torch.no_grad():
        linger.onnx.export(net, dummy_input, "data.ignore/torch_unsqueeze.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #original_model = onnx.load("data.ignore/torch_view.onnx.ignore")
    aa_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/torch_unsqueeze.onnx.ignore")
    aa_iq = from_torch_tensor(dummy_input, 12, 3)
    with torch.no_grad():
        linger.onnx.export(net, aa_iq, "data.ignore/iq_unsqueeze.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #iq1_model = onnx.load("data.ignore/iq_view.onnx.ignore")
    aa_iq_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/iq_unsqueeze.onnx.ignore")
    assert aa_model_md5 == aa_iq_model_md5


def test_transpose_export():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, *args):
            x = args[0].squeeze()
            return x.transpose(1, 2)
    net = TestModel()
    dummy_input = torch.randn(1, 2, 2, 2)
    with torch.no_grad():
        linger.onnx.export(net, dummy_input, "data.ignore/torch_transpose.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #original_model = onnx.load("data.ignore/torch_view.onnx.ignore")
    aa_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/torch_transpose.onnx.ignore")
    aa_iq = from_torch_tensor(dummy_input, 5, 3)
    with torch.no_grad():
        linger.onnx.export(net, aa_iq, "data.ignore/iq_transpose.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #iq1_model = onnx.load("data.ignore/iq_view.onnx.ignore")
    aa_iq_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/iq_transpose.onnx.ignore")
    assert aa_model_md5 == aa_iq_model_md5


def test_flatten_export():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, *args):
            return args[0].flatten(2, 4)
    net = TestModel()
    dummy_input = torch.randn(1, 2, 3, 7, 5)
    with torch.no_grad():
        linger.onnx.export(net, dummy_input, "data.ignore/torch_flatten.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #original_model = onnx.load("data.ignore/torch_view.onnx.ignore")
    aa_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/torch_flatten.onnx.ignore")
    aa_iq = from_torch_tensor(dummy_input, 12, 3)
    with torch.no_grad():
        linger.onnx.export(net, aa_iq, "data.ignore/iq_flatten.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    #iq1_model = onnx.load("data.ignore/iq_view.onnx.ignore")
    aa_iq_model_md5 = get_file_topolo_sort_type_list(
        "data.ignore/iq_flatten.onnx.ignore")
    assert aa_model_md5 == aa_iq_model_md5


def test_getitem_export():
    class TestModel(torch.nn.Module):
        def __init__(self):
            super(TestModel, self).__init__()

        def forward(self, *args):
            return args[0][:, 1, :, 2, :]
    net = TestModel()
    dummy_input = torch.randn(1, 2, 3, 7, 5)
    with torch.no_grad():
        linger.onnx.export(net, dummy_input, "data.ignore/torch_getitem.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    original_model = onnx.load("data.ignore/torch_getitem.onnx.ignore")

    aa_iq = from_torch_tensor(dummy_input, 12, 3)
    with torch.no_grad():
        linger.onnx.export(net, aa_iq, "data.ignore/iq_getitem.onnx.ignore", export_params=True, opset_version=11,
                           input_names=['x', ], operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    iq_model = onnx.load("data.ignore/iq_getitem.onnx.ignore")
    assert original_model.graph.node[0].op_type == iq_model.graph.node[0].op_type
    assert original_model.graph.node[1].op_type == iq_model.graph.node[1].op_type
    assert original_model.graph.node[2].op_type == iq_model.graph.node[2].op_type
    assert original_model.graph.node[3].op_type == iq_model.graph.node[3].op_type
