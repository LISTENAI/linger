import os

import linger
import numpy as np
import torch
import torch.nn as nn

if not os.path.exists('data.ignore'):
    os.mkdir('data.ignore')


def test_softmaxint():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(64, 256)
            self.fc2 = nn.Linear(64, 256)
            self.fc3 = nn.Linear(64, 256)
            self.fc4 = nn.Linear(480, 1000)

        def forward(self, input):
            x = self.fc1(input)
            y = self.fc2(input)
            z = self.fc3(input)

            x = x.view(8, 15, 32)
            y = y.view(8, 32, 15)
            z = z.view(8, 15, 32)
            x = torch.bmm(x, y)
            x = torch.softmax(x, dim=-1)
            x = torch.bmm(x, z)
            x = x.view(8, 480)
            x = self.fc4(x)
            x = torch.log_softmax(x, dim=-1)

            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    #gpu code current has bugs to fix
    torch.cuda.set_device(0)
    net = Net().cuda()
    dummy_input = torch.randn(15, 64, requires_grad=True).cuda()
    target = torch.ones(8, dtype=torch.int64).cuda()
    # net = Net()
    # dummy_input = torch.randn(15, 64, requires_grad=True)
    # target = torch.ones(1, dtype=torch.int64)
    criterion = nn.CrossEntropyLoss()
    replace_tuple = (nn.Linear)
    net.train()
    linger.SetFunctionBmmQuant(True)
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    net = linger.init(net, quant_modules=replace_tuple,
                      mode=linger.QuantMode.QValue)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss = None
    for i in range(10):
        optimizer.zero_grad()
        out = net(dummy_input)
        loss = criterion(out, target)
        if i % 1 == 0:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()
    net.eval()
    torch.save(net.state_dict(), 'data.ignore/softmax.pt')
    out1 = net(dummy_input)
    with torch.no_grad():
        torch.onnx.export(net, dummy_input, "data.ignore/softmax_net.onnx", export_params=True,
                          opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    out2 = net(dummy_input)
    assert out1.abs().sum() == out2.abs().sum()
