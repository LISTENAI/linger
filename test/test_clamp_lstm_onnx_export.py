#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import linger
import numpy
import torch
import torch.nn as nn


def test_NormalizeLSTM_onnx_export():

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.lstm = linger.NormalizeFastLSTM(
                100, 50, num_layers=1, batch_first=True, bidirectional=True)
            # self.lstm = nn.LSTM(
            #     100, 50, num_layers=1, batch_first=True, bidirectional=True)

        def forward(self, input, batch_lengths=None, initial_state=None):
            # input  (b t d)
            # x = nn.utils.rnn.pack_padded_sequence(x, batch_lengths, batch_first=True, enforce_sorted=False)

            # normalize
            x = (input, batch_lengths, True, False)
            x, hc = self.lstm(x, initial_state)
            x, _ = x

            # torch
            # x, hc = self.lstm(input, initial_state)

            # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

            return x
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    numpy.random.seed(1)
    aa = torch.randn(10, 10, 100).cuda()
    label = torch.randint(10, (10, 10)).cuda()  # class=10
    mask = torch.ones(10, 10)
    for i in range(9):
        index = numpy.random.randint(5, 10)
        mask[i, index:] = 0
        label[i, index:] = -1

    input_lengths = mask.long().sum(1).cpu().numpy()
    input_lengths = torch.tensor(input_lengths)  # .cuda()

    batch_size = 10
    hidden_size = 50
    size = 2
    initial_state = (torch.zeros(size, batch_size, hidden_size).cuda(),
                     torch.zeros(size, batch_size, hidden_size).cuda())
    net = Net().cuda()

    net.eval()

    out1 = net(aa, input_lengths, initial_state)

    with torch.no_grad():
        net.eval()
        torch.onnx.export(net, (aa, input_lengths, initial_state), "data.ignore/normalize_torch_lstm.onnx", export_params=True,
                          opset_version=12, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
