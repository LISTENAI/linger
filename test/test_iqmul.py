import torch
import torch.nn as nn
import linger
from linger.ops import from_torch_tensor
from linger.ops import (IQTensor, from_torch_tensor, iqadd, iqAddLayer,
                             iqmul, iqMulLayer, torch_cat)

def test_iqmul_iqtensor_scalar():
    class iqTestLayer(torch.nn.Module):
        def __init__(self):
            super(iqTestLayer, self).__init__()
            self.fc = nn.Linear(2, 1)

        def forward(self, x):
            x = self.fc(x)
            x = x * 0.125
            return x

    model = iqTestLayer().cuda()
    model = linger.init(model, quant_modules=(nn.Linear), parameter_bits=8)

    x = torch.tensor([[0.6, 0.4]], requires_grad=True).cuda()
    # import pdb; pdb.set_trace()
    a = from_torch_tensor(x, 127.0/8, 8)
    
    out = model(x)
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, (x),"./data.ignore/iqmul.onnx",export_params=True,opset_version=11,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

def test_iqmul_scale_x_y_o():
    
    class iqTestLayer(torch.nn.Module):
        def __init__(self):
            super(iqTestLayer, self).__init__()
            self.fc = nn.Linear(2, 1)

        def forward(self, x):
            x = iqmul(self, x, 0.125)
            return x

    x = torch.tensor([[0.6, 0.4]], requires_grad=True).cuda()
    a = from_torch_tensor(x, 16, 8)

    model = iqTestLayer().cuda()
    # model = linger.init(model, quant_modules=(), parameter_bits=8)

    out = model(a)
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, (a),"./data.ignore/iqmul.onnx",export_params=True,opset_version=11,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
