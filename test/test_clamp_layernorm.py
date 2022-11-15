import numpy as np
import torch
import torch.nn as nn
from linger import NormalizeLayerNorm


def test_normalize_batchnorm2d_foward():
    module = NormalizeLayerNorm(
        [4,4], normalize_data=4, normalize_weight=4, normalize_bias=4)
    module.weight.data.fill_(8)
    module.bias.data.fill_(8)
    input = 8 * torch.randn(1, 64, 4, 4)
    assert (input < 8).any()
    assert (input > 4).any()
    m = module(input)
    assert (m < 4.0001).all()

def test_normalize_batchnorm2d():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 3, kernel_size=2,
                                  stride=1, padding=1, bias=False, groups=1)
            self.ln = NormalizeLayerNorm([4,4], normalize_data=4, normalize_weight=4, normalize_bias=4)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(48, 100)

        def forward(self, x):
            x = self.conv(x)
            x = self.ln(x)
            x = self.relu(x)
            n, c, h, w = x.shape
            x = x.view((n, c*h*w))
            x = self.fc(x)
            return x
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)
    # random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    aa = torch.randn(1, 1, 3, 3).cuda()
    target = torch.ones(1, 100).cuda()

    criterion = nn.MSELoss()
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    loss = None
    for i in range(200):
        optimizer.zero_grad()
        out = net(aa)
        loss = criterion(out, target)
        if i % 20 == 0:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()
    # assert loss < 1e-12, 'training loss error'
    net.eval()
    torch.save(net.state_dict(), 'data.ignore/layernorm_normalize.pt')
    out1 = net(aa)
    # print(out1)
    with torch.no_grad():
        torch.onnx.export(net, aa, "data.ignore/conv_layernorm_normalize.onnx", export_params=True,
                          opset_version=9, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
