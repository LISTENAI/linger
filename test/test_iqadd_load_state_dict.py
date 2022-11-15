import os

import linger
import numpy as np
import torch
import torch.nn as nn
from linger.ops import IQTensor, from_torch_tensor

if not os.path.exists('data.ignore'):
    os.mkdir('data.ignore')


def test_iqadd_load_state_dict_1():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.transpose = nn.ConvTranspose2d(2, 2, 5, 5, 2, 4, 2, True, 2)
            self.conv = nn.Conv2d(2, 2, kernel_size=3, stride=1,
                                  padding=1, bias=True)
            self.fc = nn.Linear(392, 100, bias=False)

        def forward(self, x):
            trans = self.transpose(x)
            assert isinstance(trans, IQTensor)
            conv = self.conv(trans)
            assert isinstance(conv, IQTensor)
            x = trans + conv
            assert isinstance(x, IQTensor)
            n, c, h, w = x.shape
            x = x.view((n, c*h*w))
            x = self.fc(x)
            return x
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    torch.cuda.set_device(0)
    net1 = Net().cuda()
    dummy_input = torch.randn(1, 2, 2, 2).cuda()
    target = torch.ones(1, 100).cuda()
    criterion = nn.MSELoss()

    replace_tuple = (nn.Linear, nn.ConvTranspose2d, nn.Conv2d)
    net1.train()
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    net1 = linger.init(net1, quant_modules=replace_tuple,
                       mode=linger.QuantMode.QValue)
    optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.01)
    for i in range(150):
        optimizer1.zero_grad()
        out1 = net1(dummy_input)
        loss1 = criterion(out1, target)
        loss1.backward()
        optimizer1.step()
        if i % 30 == 29:
            print('loss1 {}'.format(loss1))
    net1.eval()
    torch.save(net1.state_dict(), 'data.ignore/model.pt.ignore')
    out1 = net1(dummy_input)

    net2 = Net().cuda()
    net2 = linger.init(net2, quant_modules=replace_tuple,
                       mode=linger.QuantMode.QValue)
    net2.load_state_dict(torch.load(
        'data.ignore/model.pt.ignore', map_location='cpu'))
    net2.eval()
    out2 = net2(dummy_input)

    with torch.no_grad():
        torch.onnx.export(net2, dummy_input, "data.ignore/iqadd_t710.onnx", export_params=True,
                          opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    assert out1.sum() == out2.sum()