import linger
import numpy as np
import torch
import torch.nn as nn

channel_size = 3


def try_dump():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(channel_size, channel_size,
                                  kernel_size=1, stride=1, padding=0, bias=True)
            self.bn = nn.BatchNorm2d(channel_size)

        def forward(self, x):

            x = self.conv(x)
            z = self.bn(x)

            add_rlt = x + z
            mul_rlt = x * z
            div_rlt = x / 0.5
            sum_rlt = x.sum(axis=[2, 3], keepdim=False)
            return add_rlt, mul_rlt, div_rlt, sum_rlt

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    torch.cuda.set_device(0)
    net = Net().cuda()
    replace_tuple = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                     nn.BatchNorm2d, linger.NormalizeConvBN2d)

    net = linger.init(net,   quant_modules=replace_tuple, mode=linger.QuantMode.QValue)

    bb = torch.randn(1, channel_size, 32, 32).cuda()

    net.train()
    for _ in range(22):
        net(bb)
    net.eval()

    # print(net(bb))

    with linger.Dumper() as dumper:
        net.eval()
        # dumper.enable_dump_quanted(net,path="dump_iqadd")
        dumper.enable_dump_quanted(net, path="dump_iqtensor")
        net(bb)

    # print("dump: ",net(bb))

    # export_path  = "dump_iqadd.onnx"
    # with torch.no_grad(): #training = torch.onnx.TrainingMode.TRAINING,
    #     torch.onnx.export(net, (bb),export_path, export_params=True,opset_version=12,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
