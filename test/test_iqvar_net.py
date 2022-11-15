import numpy as np
import torch
import torch.nn as nn

import linger


def test_var():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                                  padding=1, bias=True)
            self.fc = nn.Linear(1, 100)

        def forward(self, x):
            x = self.conv(x)
            x = torch.var(x, 1, False).reshape(-1)
            x = x.unsqueeze(-1)
            x = self.fc(x)
            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    dummy_input = torch.randn(1, 10, 10, 10).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()

    replace_tuple = (nn.Conv2d, nn.Linear, nn.AvgPool2d)
    model = Model().cuda()

    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    linger.SetIQTensorVar(True)
    model = linger.init(model, quant_modules=replace_tuple,
                        mode=linger.QuantMode.QValue)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = None
    for i in range(20):
        optimizer.zero_grad()
        out = model(dummy_input)
        print(out)
        loss = criterion(out, target)
        if i % 1 == 0:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        torch.onnx.export(model, dummy_input, "./data.ignore/var.onnx", export_params=True, opset_version=11,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
