import linger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def test_normalize_shuffle_channel():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = nn.Conv2d(10, 10, kernel_size=3, stride=1,
                                  padding=1, bias=True)
            # self.linger_shuffle_channel = linger.NormalizeShuffleChannel(8)
            self.fc = nn.Linear(1000, 100)

        def forward(self, x):
            x = self.conv(x)
            # x = self.linger_shuffle_channel(x, 1)
            x = linger.channel_shuffle(x, 2)
            x = x.view(1, -1)
            x = torch.relu(x)
            x = self.fc(x)
            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    dummy_input = torch.randn(1, 10, 10, 10).cuda()
    target = torch.ones(100).cuda()
    criterion = nn.MSELoss()

    replace_tuple = (nn.Conv2d, nn.Linear)
    model = Model().cuda()

    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    linger.SetFunctionChannelShuffleQuant(True)
    model = linger.init(model, quant_modules=replace_tuple,
                        mode=linger.QuantMode.QValue)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = None
    for i in range(20):
        optimizer.zero_grad()
        out = model(dummy_input)
        loss = criterion(out, target)
        if i % 1 == 0:
            print('loss: ', loss)
        loss.backward()
        optimizer.step()


    with linger.Dumper() as dumper :
        model.eval()
        dumper.enable_dump_quanted(model, path="./data.ignore/dump_shuffle_net_10_09_01")
        out = model(dummy_input)
    dummy_input1 = dummy_input.detach().cpu().numpy()
    dummy_input1.tofile("./data.ignore/dump_shuffle_net_10_09_01/input.bin")
    out1 = out.detach().cpu().numpy()
    out1.tofile("./data.ignore/dump_shuffle_net_10_09_01/output.bin")


    with torch.no_grad():
        torch.onnx.export(model, dummy_input, "./data.ignore/shuffle_net_10_09_01.onnx", export_params=True, opset_version=11,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
