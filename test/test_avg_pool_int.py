import os

import linger
import numpy as np
import torch
import torch.nn as nn

if not os.path.exists("data.ignore"):
    os.makedirs("data.ignore")


def test_avgpool2dint():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                                  padding=1, bias=True)
            self.bn = nn.BatchNorm2d(10)
            self.relu = nn.ReLU()
            self.pool = nn.AvgPool2d((2, 2), (2, 2), (0, 0), False)
            self.fc = nn.Linear(250, 100)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            n, c, h, w = x.shape
            x = self.pool(x)
            n, c, h, w = x.shape
            x = x.view((n, c*h*w))
            x = self.fc(x)
            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    dummy_input = torch.randn(1, 10, 10, 10).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()
    replace_tuple = (nn.Conv2d, nn.Linear, nn.AvgPool2d)
    net.train()
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    # linger.disable_quant(net.fc)
    net = linger.init(net, quant_modules=replace_tuple,
                      mode=linger.QuantMode.QValue)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss = None
    for i in range(200):
        optimizer.zero_grad()
        out = net(dummy_input)
        loss = criterion(out, target)
        if i % 20 == 0:
            print('loss: ', loss)
        # if i == 190:
        #     import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()
    net.eval()
    torch.save(net.state_dict(), 'data.ignore/aa.pt')

    out1 = net(dummy_input)
    with torch.no_grad():
        torch.onnx.export(net, dummy_input, "data.ignore/avg_pool.onnx", export_params=True, opset_version=11,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    print("out1.mean(): ", out1.mean())
    assert abs(out1.mean() - 1.0) < 0.15
