import linger
import pytest
import torch
import torch.nn as nn
from linger.ops import *


def test_linear():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    origin_fc = nn.Linear(50, 100, True).cuda().train()
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    my_fc = LinearInt(50, 100, True, data_bits=8,
                      parameter_bits=8).cuda().train()
    weight = torch.randn(100, 50)
    bias = torch.randn(100)
    origin_fc.weight.data.copy_(weight)
    origin_fc.bias.data.copy_(bias)
    my_fc.weight.data.copy_(weight)
    my_fc.bias.data.copy_(bias)
    input1 = torch.ones(50, 50, requires_grad=True).cuda()
    for epoch in range(10):
        output1 = origin_fc(input1)
        output2 = my_fc(input1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(origin_fc.parameters(), lr=0.1)
        target = torch.ones(50, 100).cuda()
        loss = criterion(output1, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer1 = torch.optim.SGD(my_fc.parameters(), lr=0.1)
        loss1 = criterion(output2, target)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        print('loss1, loss2: ', loss, loss1)
        assert criterion(origin_fc.weight, my_fc.weight) < 0.02


def test_conv():
    torch.manual_seed(1)
    origin_conv = nn.Conv2d(10, 10, 1, 1, 0, 1, 1).cuda()
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    my_conv = Conv2dInt(10, 10, 1, 1, 0, 1, 1, data_bits=8,
                        parameter_bits=8, o_bits=8).cuda()
    weight = torch.randn(10, 10, 1, 1)
    bias = torch.randn(10)
    origin_conv.weight.data.copy_(weight)
    origin_conv.bias.data.copy_(bias)
    my_conv.weight.data.copy_(weight)
    my_conv.bias.data.copy_(bias)
    input1 = torch.ones(10, 10, 50, 50, requires_grad=True).cuda()
    for epoch in range(10):
        output1 = origin_conv(input1)
        output2 = my_conv(input1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(origin_conv.parameters(), lr=0.01)
        target = torch.ones(10, 10, 50, 50, requires_grad=True).cuda()
        loss = criterion(output1, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer1 = torch.optim.SGD(my_conv.parameters(), lr=0.01)
        loss1 = criterion(output2, target)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        assert criterion(origin_conv.weight, my_conv.weight) < 0.02


class bn(nn.Module):
    def __init__(self):
        super(bn, self).__init__()

        self.out = nn.BatchNorm2d(100)

    def forward(self, input):
        return self.out(input)


class bn_(nn.Module):
    def __init__(self):
        super(bn_, self).__init__()

        self.out = BatchNormInt(100)

    def forward(self, input):
        return self.out(input)


def test_convtranspose():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    origin_convtranspose = nn.ConvTranspose2d(
        10, 10, 5, 5, 2, 4, 2, True, 2).cuda().train()
    my_convtranspose = ConvTranspose2dInt(
        10, 10, 5, 5, 2, 4, 2, True, 2, data_bits=16, parameter_bits=16, o_bits=8).cuda().train()
    weight = torch.randn(10, 5, 5, 5)
    bias = torch.randn(10)
    origin_convtranspose.weight.data.copy_(weight)
    origin_convtranspose.bias.data.copy_(bias)
    my_convtranspose.weight.data.copy_(weight)
    my_convtranspose.bias.data.copy_(bias)
    input1 = torch.randn(10, 10, 50, 50, requires_grad=True).cuda()
    for epoch in range(10):
        output1 = origin_convtranspose(input1)
        output2 = my_convtranspose(input1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(origin_convtranspose.parameters(), lr=0.01)
        target = torch.ones(10, 10, 254, 254, requires_grad=True).cuda()
        loss = criterion(output1, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer1 = torch.optim.SGD(my_convtranspose.parameters(), lr=0.01)
        loss1 = criterion(output2, target)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        print("loss1: {}, loss2: {}".format(loss, loss1))
        assert criterion(origin_convtranspose.weight,
                         my_convtranspose.weight) < 0.02


def test_batchnorm():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    origin_bn = nn.BatchNorm2d(10, eps=1e-5, momentum=0.1).cuda().train()
    my_bn = BatchNormInt(10, eps=1e-5, momentum=0.1, o_bits=8,
                         data_bits=8, parameter_bits=8).cuda().train()
    weight = torch.randn(10)
    bias = torch.randn(10)
    running_mean = torch.randn(10)
    running_var = torch.randn(10)
    origin_bn.weight.data.copy_(weight)
    origin_bn.bias.data.copy_(bias)
    origin_bn.running_mean.data.copy_(running_mean)
    origin_bn.running_var.data.copy_(running_var)

    my_bn.weight.data.copy_(weight)
    my_bn.bias.data.copy_(bias)
    my_bn.running_mean.data.copy_(running_mean)
    my_bn.running_var.data.copy_(running_var)
    # target = torch.ones(10, 10, 50, 50, requires_grad=True).cuda()
    grad = torch.randn(10, 10, 100, 100).cuda()
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    input1 = torch.randn(10, 10, 100, 100, requires_grad=True).cuda()
    input2 = input1.detach().clone()

    for i in range(100):
        output1 = origin_bn(input1)
        output2 = my_bn(input2)
    output1 = origin_bn.eval()(input1)
    output2 = my_bn.eval()(input2)
    assert (output1.abs().mean() - output2.abs().mean()).abs() < 1/127


def test_gru():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    origin_gru = nn.GRU(10, 20, batch_first=True,
                        bidirectional=True).cuda().train()
    my_gru = GRUInt(10, 20, batch_first=True,
                    bidirectional=True, o_bits=8).cuda().train()
    weight_ih = torch.randn(60, 10)
    weight_hh = torch.randn(60, 20)
    bias_ih = torch.randn(60)
    bias_hh = torch.randn(60)
    origin_gru.weight_ih_l0.data.copy_(weight_ih)
    origin_gru.weight_hh_l0.data.copy_(weight_hh)
    origin_gru.bias_ih_l0.data.copy_(bias_ih)
    origin_gru.bias_hh_l0.data.copy_(bias_hh)
    my_gru.weight_ih_l0.data.copy_(weight_ih)
    my_gru.weight_hh_l0.data.copy_(weight_hh)
    my_gru.bias_ih_l0.data.copy_(bias_ih)
    my_gru.bias_hh_l0.data.copy_(bias_hh)
    input1 = torch.ones(10, 5, 10, requires_grad=True).cuda()
    for epoch in range(10):
        output1, hy = origin_gru(input1)
        output2, hy = my_gru(input1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(origin_gru.parameters(), lr=0.1)
        target = torch.ones(10, 5, 40).cuda()
        loss = criterion(output1, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer1 = torch.optim.SGD(my_gru.parameters(), lr=0.1)
        loss1 = criterion(output2, target)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        print('loss1, loss2: ', loss, loss1)
        assert criterion(origin_gru.weight_ih_l0, my_gru.weight_ih_l0) < 0.02
        assert criterion(origin_gru.weight_hh_l0, my_gru.weight_hh_l0) < 0.02
        assert criterion(origin_gru.bias_ih_l0, my_gru.bias_ih_l0) < 0.02
        assert criterion(origin_gru.bias_hh_l0, my_gru.bias_hh_l0) < 0.02


def test_lstm():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    origin_lstm = nn.LSTM(10, 20, batch_first=True,
                          bidirectional=True).cuda().train()
    my_lstm = LSTMInt(10, 20, batch_first=True,
                      bidirectional=True, o_bits=8).cuda().train()
    weight_ih = torch.randn(80, 10)
    weight_hh = torch.randn(80, 20)
    bias_ih = torch.randn(80)
    bias_hh = torch.randn(80)
    origin_lstm.weight_ih_l0.data.copy_(weight_ih)
    origin_lstm.weight_hh_l0.data.copy_(weight_hh)
    origin_lstm.bias_ih_l0.data.copy_(bias_ih)
    origin_lstm.bias_hh_l0.data.copy_(bias_hh)
    my_lstm.weight_ih_l0.data.copy_(weight_ih)
    my_lstm.weight_hh_l0.data.copy_(weight_hh)
    my_lstm.bias_ih_l0.data.copy_(bias_ih)
    my_lstm.bias_hh_l0.data.copy_(bias_hh)
    input1 = torch.ones(10, 5, 10, requires_grad=True).cuda()
    for epoch in range(10):
        output1, (hy, cy) = origin_lstm(input1)
        output2, (hy, cy) = my_lstm(input1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(origin_lstm.parameters(), lr=0.1)
        target = torch.ones(10, 5, 40).cuda()
        loss = criterion(output1, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer1 = torch.optim.SGD(my_lstm.parameters(), lr=0.1)
        loss1 = criterion(output2, target)
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        print('loss1, loss2: ', loss, loss1)
        assert criterion(origin_lstm.weight_ih_l0, my_lstm.weight_ih_l0) < 0.02
        assert criterion(origin_lstm.weight_hh_l0, my_lstm.weight_hh_l0) < 0.02
        assert criterion(origin_lstm.bias_ih_l0, my_lstm.bias_ih_l0) < 0.02
        assert criterion(origin_lstm.bias_hh_l0, my_lstm.bias_hh_l0) < 0.02
