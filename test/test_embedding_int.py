#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import linger
import numpy
import torch
import torch.nn as nn


def test_embeddingint_net():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.gather = nn.Embedding(10, 3)

        def forward(self, input):

            x = self.gather(input)
            x = x * 1

            return x
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    numpy.random.seed(1)
    aa = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]).cuda()
    replace_tuple = (nn.Conv2d, nn.LSTM, nn.Embedding)

    net = Net().cuda()
    net = linger.init(net, quant_modules=replace_tuple, mode=linger.QuantMode.QValue)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    loss = None
    label = torch.randint(1, (2,  3)).cuda()
    for i in range(10):
        optimizer.zero_grad()
        out = net(aa)
        loss = criterion(out, label)
        if i % 1 == 0:
            print('loss: ', loss)

        loss.backward()
        optimizer.step()
    with torch.no_grad():
        net.eval()
        net(aa)
        torch.onnx.export(net, (aa), "data.ignore/embedding_int.onnx", export_params=True,
                          opset_version=12, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)