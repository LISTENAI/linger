#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import linger
import torch
import torch.nn as nn
import numpy

def test_lstmpint_net():

    def getacc(lprob, target):
        num_class = lprob.size()[1]
        _, new_target = torch.broadcast_tensors(lprob, target)

        remove_pad_mask = new_target.ne(-1)
        lprob = lprob[remove_pad_mask]

        target = target[target!=-1]
        target = target.unsqueeze(-1)


        lprob = lprob.reshape((-1, num_class))

        preds = torch.argmax(lprob, dim=1)
        
        correct_holder = torch.eq(preds.squeeze(), target.squeeze()).float()

        num_corr = correct_holder.sum()
        num_sample = torch.numel(correct_holder)
        acc = num_corr/num_sample
        return acc


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv0 = nn.Conv2d(1, 100, kernel_size=(1,3), padding=(0,1), groups=1, bias=True)
            self.bn0 = nn.BatchNorm2d(100)
            self.relu0 = nn.ReLU()
            self.conv1 = nn.Conv2d(100, 100, kernel_size=(1,3), padding=(0,1), groups=1, bias=True)
            self.bn1 = nn.BatchNorm2d(100)
            self.relu1 = nn.ReLU()
            self.lstmp = nn.LSTM(100, 100, num_layers=1, batch_first=True, bidirectional=False)
            # self.lstmp = nn.LSTM(100, 50, num_layers=1, batch_first=True, bidirectional=True)
            self.final_conv = nn.Conv2d(100, 10, 1, 1, 0)
        def forward(self, input, batch_lengths=None, initial_state=None):
            x = self.conv0(input)
            x = self.bn0(x)
            x = self.relu0(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            n, c, h, w = x.shape
            x = x.reshape(n, -1, 1, w).squeeze(2)
            x = x.permute(0, 2, 1)  #b t d
            # !! !!! !! !!! !! !! !! !!!! !! !! ! !!!! !!! !
            # 此处之前和之后的pack_padded_sequence和pad_packed_sequence 需要写全，不能用from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence 使用
            # 直接写全torch.nn.utils.rnn.pack_padded_sequence(_,_,_,_)   torch.nn.utils.rnn.pad_packed_sequence(_,_,_,_)
            # 不然linger替换不了函数指针    会导致运行出错
            x = nn.utils.rnn.pack_padded_sequence(x, batch_lengths, batch_first=True, enforce_sorted=False)
            x, hidden = self.lstmp(x, initial_state) #output b, t, h (10, 10, 100)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x = x.permute(2, 0, 1)
            d, b, t = x.shape
            x = x.reshape((1, d, 1, b*t)) # (1, 100, 1, 100)
            x = self.final_conv(x)  #(1, 10, 1, 100) (d, b*t)
            return x
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    numpy.random.seed(1)
    aa = torch.randn(10, 1, 1, 10).cuda()
    label = torch.randint(10, (10, 10)).cuda()   #class=10
    mask = torch.ones(10, 10)
    for i in range(9):
        index = numpy.random.randint(5, 10)
        mask[i, index:] = 0
        label[i, index:] = -1

    input_lengths = mask.long().sum(1).cpu().numpy()
    input_lengths = torch.tensor(input_lengths)#.cuda()
    print('input_lengths: ', input_lengths)
    # input_lengths = None
    # label = label.permute((1, 0))
    # batch_size = 10; hidden_size=50; size=2
    batch_size = 10; hidden_size=100; size=1
    initial_state = (torch.zeros(size, batch_size, hidden_size).cuda(), 
                                torch.zeros(size, batch_size, hidden_size).cuda())
    # initial_state = None
    net = Net().cuda()
    replace_modules = (nn.Conv2d, nn.LSTM, nn.BatchNorm2d)
    # replace_modules = (nn.LSTM,)
    linger.SetPlatFormQuant(platform_quant=linger.PlatFormQuant.luna_quant)
    net = linger.init(net, quant_modules=replace_modules)
    criterion = nn.CrossEntropyLoss(ignore_index = -1)

    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    loss = None
    for i in range(50):
        optimizer.zero_grad()
        out = net(aa, input_lengths, initial_state)
        out = out.squeeze().permute((1,0)) #(b*t, d)
        loss = criterion(out, label.reshape(-1))
        if i % 50 == 0:
            print('loss: ', loss)
            acc = getacc(out, label.reshape(-1, 1))
            print('train acc: ', acc)
        loss.backward()
        optimizer.step()

    net.eval()
    out1 = net(aa, input_lengths, initial_state)
    out1 = out1.squeeze().permute((1,0))
    acc = getacc(out1, label.reshape(-1, 1))
    print('test acc: ', acc)
    assert acc > 0.4
    input_lengths = torch.tensor(input_lengths)#.cuda()
    # net = torch.jit.trace(net, (aa, input_lengths, initial_state))
    print(net)
    with torch.no_grad():
        net.eval()
        torch.onnx.export(net, (aa, input_lengths, initial_state), "data.ignore/single_lstm_4.onnx",export_params=True,opset_version=12, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    out2 = net(aa, input_lengths, initial_state)
    out2 = out2.squeeze().permute((1,0))
    assert (out1 == out2).all()