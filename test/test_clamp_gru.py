import linger
import numpy
import torch
import torch.nn as nn
from linger import NormalizeFastGRU


def test_lstmpint_net():

    def getacc(lprob, target):
        num_class = lprob.size()[1]
        _, new_target = torch.broadcast_tensors(lprob, target)

        remove_pad_mask = new_target.ne(-1)
        lprob = lprob[remove_pad_mask]

        target = target[target != -1]
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
            self.conv0 = nn.Conv2d(1, 100, kernel_size=(
                1, 3), padding=(0, 1), groups=1, bias=True)
            self.bn0 = nn.BatchNorm2d(100)
            self.relu0 = nn.ReLU()
            self.conv1 = nn.Conv2d(100, 100, kernel_size=(
                1, 3), padding=(0, 1), groups=1, bias=True)
            self.bn1 = nn.BatchNorm2d(100)
            self.relu1 = nn.ReLU()
            self.lstmp = nn.GRU(100, 50, num_layers=1,
                                batch_first=True, bidirectional=True)
            self.final_conv = nn.Conv2d(100, 10, 1, 1, 0)

        def forward(self, input, batch_lengths=None, initial_state=None):
            x = self.conv0(input)
            x = self.bn0(x)
            x = self.relu0(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            n, c, h, w = x.shape
            x = x.reshape((n, -1, 1, w)).squeeze(2)
            x = x.permute((0, 2, 1))  # b t d
            x = nn.utils.rnn.pack_padded_sequence(
                x, batch_lengths, batch_first=True, enforce_sorted=False)
            x, _ = self.lstmp(x, initial_state)  # output b, t, h (10, 10, 100)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            x = x.permute((2, 0, 1))
            d, b, t = x.shape
            x = x.reshape((1, d, 1, b*t))  # (1, 100, 1, 100)
            x = self.final_conv(x)  # (1, 10, 1, 100) (d, b*t)
            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    numpy.random.seed(1)
    dummy_input = torch.randn(10, 1, 1, 10).cuda()
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
    initial_state = torch.zeros(size, batch_size, hidden_size).cuda()
    net = Net().cuda()
    replace_modules = (nn.Conv2d, nn.GRU, nn.BatchNorm2d)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # net = linger.normalize_layers(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    print('net: ', net)
    # net.load_state_dict(torch.load('data.ignore/gru.pt'))
    loss = None
    for i in range(5):
        optimizer.zero_grad()
        out = net(dummy_input, input_lengths, initial_state)
        out = out.squeeze().permute((1, 0))  # (b*t, d)
        loss = criterion(out, label.reshape(-1))
        if i % 50 == 0:
            print('loss: ', loss)
            acc = getacc(out, label.reshape(-1, 1))
            print('train acc: ', acc)
        loss.backward()
        optimizer.step()

    net.eval()
    out1 = net(dummy_input, input_lengths, initial_state)
    out1 = out1.squeeze().permute((1, 0))
    acc = getacc(out1, label.reshape(-1, 1))
    print('test acc: ', acc)
    torch.save(net.state_dict(), 'data.ignore/gru.pt')
    # out3 = net(dummy_input, input_lengths, initial_state)
    # out3 = out3.squeeze().permute((1,0))
    # input_lengths = torch.tensor(input_lengths)#.cuda()
    # with torch.no_grad():
    #     net.eval()
    #     torch.onnx.export(net, (dummy_input, input_lengths, initial_state), "data.ignore/gru.onnx", export_params=True,
    #                       opset_version=12, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
    out2 = net(dummy_input, input_lengths, initial_state)
    out2 = out2.squeeze().permute((1, 0))
    # print('out: ', out1)
    assert (out1 == out2).all()


def test_normalize_gru():

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.rnn = nn.GRU(10, 20, 1, batch_first=True, bidirectional=True)
            # self.rnn = NormalizeFastGRU(10, 20, 1, batch_first=True, bidirectional=True)

        def forward(self, input, h0):
            x, hn = self.rnn(input, h0)

            return x

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    numpy.random.seed(1)

    dummy_input = torch.randn(3, 5, 10).cuda()
    h0 = torch.randn(2, 3, 20).cuda()

    model = Net().cuda()

    model = linger.normalize_layers(model)

    # torch.save(model.state_dict(), "data.ignore/torch_gru.pt")
    # model.load_state_dict(torch.load("data.ignore/torch_gru.pt"))
    model.eval()

    out1 = model(dummy_input, h0)

    print('model: ', model)

    with torch.no_grad():
        model.eval()
        torch.onnx.export(model, (dummy_input, h0), "data.ignore/normalize_torch_gru.onnx", export_params=True,
                          opset_version=12, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
