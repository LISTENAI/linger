import copy
import math
from collections import OrderedDict

import lingerext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.onnx import is_in_onnx_export

from ..config import config
from ..quant import (normalize_bias_with_config, normalize_data_with_config,
                     normalize_weight_with_config)
from ..utils import (Dump, PlatFormQuant, QuantMode, ScalerBuffer, _slice,
                     _unbind, _unbind_packed, get_max_value, hx_slice)
from .iqtensor import (IQTensor, Quant2IQTensor, from_torch_tensor,
                       platform_to_string, quantlinear)
from .linger_functional import (iqCat, torch_pack_padded_sequence,
                                torch_pad_packed_sequence)
from .ops import ModuleIntConfig
from .requant import Requant

iqcat_sym = iqCat.symbolic


def castor_luna_sigmoid(x_int, scale_x):
    l_scale = 11 - int(math.log2(scale_x))

    if l_scale > 0:
        x_int = x_int * pow(2, l_scale)
    else:
        x_int = (x_int * pow(2, l_scale) + 0.5).floor().int()

    x_int.clamp_(-2**15, 2**15-1)
    y_int = lingerext.luna_iqsigmoid(x_int.contiguous(), float(scale_x))
    y_int.clamp_(0, 2**7-1)

    return y_int


def castor_luna_tanh(x_int, scale_x):
    l_scale = 11 - int(math.log2(scale_x))

    if l_scale > 0:
        x_int = x_int * pow(2, l_scale)
    else:
        x_int = (x_int * pow(2, l_scale) + 0.5).floor().int()

    x_int.clamp_(-2**15, 2**15-1)
    y_int = lingerext.luna_iqtanh(x_int.contiguous(), float(scale_x))
    y_int.clamp_(-2**7, 2**7-1)

    return y_int


class LSTMCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hidden, cx, weight_ih, weight_hh, bias_ih, bias_hh, data_bits, parameter_bits, o_bits,
                running_i, running_h, running_iw, running_hw, running_io, running_ho, running_o,
                scale_i, scale_h, scale_iw, scale_hw, scale_io, scale_ho, scale_o,
                momentum, training, prefix, dump, path, mode, quant, is_not_from_iqtensor,
                clamp_data):
        if training:
            ctx.clamp_data = clamp_data
            ctx.o_bits = o_bits
            save_tensors = [input, hidden, weight_ih,
                            weight_hh, bias_ih, bias_hh]
            q_input, scale_i, max_value_ix = quant.quant(
                input, data_bits, mode=mode, quant_data='input')
            q_iweight, scale_iw, max_value_iw = quant.quant(
                weight_ih, parameter_bits, mode=mode, quant_data='weight')
            running_iw.mul_(1-momentum).add_(momentum*max_value_iw)

            q_hidden, scale_h, max_value_hx = quant.quant(
                hidden, data_bits, mode=mode, quant_data='input')
            q_hweight, scale_hw, max_value_hw = quant.quant(
                weight_hh, parameter_bits, mode=mode, quant_data='weight')
            running_hw.mul_(1-momentum).add_(momentum*max_value_hw)

            q_input = q_input.float() if data_bits + parameter_bits <= 16 else q_input.double()
            q_iweight = q_iweight.float() if data_bits + \
                parameter_bits <= 16 else q_iweight.double()
            q_hidden = q_hidden.float() if data_bits + \
                parameter_bits <= 16 else q_hidden.double()
            q_hweight = q_hweight.float() if data_bits + \
                parameter_bits <= 16 else q_hweight.double()

            q_gi_outputs = F.linear(q_input, q_iweight)
            q_gh_outputs = F.linear(q_hidden, q_hweight)

            if bias_ih is not None:
                if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                    q_ibias = (bias_ih * scale_iw *
                               scale_i + 0.5).floor().int()
                    if data_bits + parameter_bits <= 16:
                        q_ibias = q_ibias.float()
                    else:
                        q_ibias = q_ibias.double()
                else:
                    assert False, "linger only support luna quant."
                q_gi_outputs += q_ibias.view(-1)

                if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                    q_hbias = (bias_hh * scale_hw *
                               scale_h + 0.5).floor().int()
                    if data_bits + parameter_bits <= 16:
                        q_hbias = q_hbias.float()
                    else:
                        q_hbias = q_hbias.double()
                else:
                    assert False, "linger only support luna quant."
                q_gh_outputs += q_hbias.view(-1)

            if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:  # QX+QW -> Q11
                l_scale_gi = 11 - int(math.log2(scale_i()*scale_iw()))
                if l_scale_gi > 0:
                    gi = q_gi_outputs * pow(2, l_scale_gi)
                else:
                    gi = (q_gi_outputs * pow(2, l_scale_gi) + 0.5).floor().int()

                l_scale_gh = 11 - int(math.log2(scale_h()*scale_hw()))
                if l_scale_gh > 0:
                    gh = q_gh_outputs * pow(2, l_scale_gh)
                else:
                    gh = (q_gh_outputs * pow(2, l_scale_gh) + 0.5).floor().int()
            else:  # QX+QW -> Q10
                assert False, "linger only support luna quant."

            for_backward_gi = quant.dequant(q_gi_outputs, scale_i*scale_iw)
            for_backward_gh = quant.dequant(q_gh_outputs, scale_h*scale_hw)
            save_tensors += [for_backward_gi, for_backward_gh]

            gates = gi + gh
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                ingate = castor_luna_sigmoid(ingate, 2048)  # Q11->Q7
                forgetgate = castor_luna_sigmoid(forgetgate, 2048)  # Q11->Q7
                cellgate = castor_luna_tanh(cellgate, 2048)  # Q11->Q7
                outgate = castor_luna_sigmoid(outgate, 2048)  # Q11->Q7
                new_cx = (cx * pow(2, 7) + 0.5).floor().int()  # float->Q7
                # Q7*Q7 + Q7*Q7 -> Q14+Q14 ->Q14
                cy = (forgetgate * new_cx) + (ingate * cellgate)
                # Q7*tanh(Q14->Q11)->Q7*Q7->Q14
                hy = outgate * castor_luna_tanh(cy, 2**14)
                cy = quant.dequant(cy, float(2**14))  # Q14->float
                hy = quant.dequant(hy, float(2**14))  # Q14->float
            else:
                assert False, "linger only support luna quant."

            save_tensors += [cx, cy, hy]

            if o_bits is not None:
                hy = normalize_data_with_config(hy, clamp_data)
                q_hy_outputs, scale_o, max_value_o = quant.quant(
                    hy, o_bits, mode=mode, quant_data='output')
                running_o.mul_(1-momentum).add_(momentum*max_value_o)
                running_h.mul_(1-momentum).add_(momentum*max_value_o)
                hy = quant.dequant(q_hy_outputs, scale_o)
            else:  # 为None 时  也得保证  running_h的实际值为running_o
                _, _, fake_max_value_o = quant.quant(
                    hy, 8, mode=mode, quant_data='output')
                running_h.mul_(1-momentum).add_(momentum*fake_max_value_o)

            ctx.save_for_backward(*save_tensors)

        else:
            assert running_i > 0, 'invalid running_i <= 0, please finetune training'
            if weight_ih.dtype == torch.float32:
                if is_not_from_iqtensor:
                    scale_i = ScalerBuffer(quant.running_to_scale(
                        running_i, parameter_bits, mode=mode))
                scale_h = ScalerBuffer(quant.running_to_scale(
                    running_h, parameter_bits, mode=mode))
                scale_iw = ScalerBuffer(quant.running_to_scale(
                    running_iw, parameter_bits, mode=mode))
                scale_hw = ScalerBuffer(quant.running_to_scale(
                    running_hw, parameter_bits, mode=mode))

            q_input, scale_i, _ = quant.quant(
                input, data_bits, scale_i, mode=mode, quant_data='input')
            q_hidden, scale_h, _ = quant.quant(
                hidden, data_bits, scale_h, mode=mode, quant_data='input')
            q_iweight = None
            q_hweight = None
            if weight_ih.dtype == torch.float32:
                q_iweight, _, _ = quant.quant(
                    weight_ih, parameter_bits, scale_iw, mode=mode, quant_data='weight')
                q_hweight, _, _ = quant.quant(
                    weight_hh, parameter_bits, scale_hw, mode=mode, quant_data='weight')
            else:
                q_iweight = weight_ih.double()
                q_hweight = weight_hh.double()
            q_input = q_input.double()
            q_iweight = q_iweight.double()
            q_hidden = q_hidden.double()
            q_hweight = q_hweight.double()
            q_gi_outputs = F.linear(q_input, q_iweight)
            q_gh_outputs = F.linear(q_hidden, q_hweight)
            if bias_ih is not None:
                if bias_ih.dtype == torch.float32:
                    if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                        q_ibias = (bias_ih * scale_iw *
                                   scale_i + 0.5).floor().int()
                        q_hbias = (bias_hh * scale_hw *
                                   scale_h + 0.5).floor().int()
                        if data_bits + parameter_bits <= 16:
                            q_ibias = q_ibias.float().double()
                            q_hbias = q_hbias.float().double()
                        else:
                            q_ibias = q_ibias.double()
                            q_hbias = q_hbias.double()
                    else:
                        assert False, "linger only support luna quant."
                else:
                    q_ibias = bias_ih.double()
                    q_hbias = bias_hh.double()
                q_gi_outputs += q_ibias.view(-1)
                q_gh_outputs += q_hbias.view(-1)
            if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:  # QX+QW -> Q11
                l_scale_gi = 11 - int(math.log2(scale_i()*scale_iw()))
                if l_scale_gi > 0:
                    gi = q_gi_outputs * pow(2, l_scale_gi)
                else:
                    gi = (q_gi_outputs * pow(2, l_scale_gi) + 0.5).floor().int()

                l_scale_gh = 11 - int(math.log2(scale_h()*scale_hw()))
                if l_scale_gh > 0:
                    gh = q_gh_outputs * pow(2, l_scale_gh)
                else:
                    gh = (q_gh_outputs * pow(2, l_scale_gh) + 0.5).floor().int()
            else:  # QX+QW -> Q10
                assert False, "linger only support luna quant."

            gates = gi + gh
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                ingate = castor_luna_sigmoid(ingate, 2048)  # Q11->Q7
                forgetgate = castor_luna_sigmoid(forgetgate, 2048)  # Q11->Q7
                cellgate = castor_luna_tanh(cellgate, 2048)  # Q11->Q7
                outgate = castor_luna_sigmoid(outgate, 2048)  # Q11->Q7
                new_cx = (cx * pow(2, 7) + 0.5).floor().int()  # float->Q7
                # Q7*Q7 + Q7*Q7 -> Q14+Q14 ->Q14
                cy = (forgetgate * new_cx) + (ingate * cellgate)
                # Q7*tanh(Q14->Q11)->Q7*Q7->Q14
                hy = outgate * castor_luna_tanh(cy, 2**14)
                cy = quant.dequant(cy, float(2**14))  # Q14->float
                hy = quant.dequant(hy, float(2**14))  # Q14->float
            else:
                assert False, "linger only support luna quant."

            if o_bits is not None:
                assert running_o > 0, 'invalid running_o<=0, please finetune training'
                if weight_ih.dtype == torch.float32:
                    scale_o = ScalerBuffer(quant.running_to_scale(
                        running_o, o_bits, mode=mode))
                q_hy_outputs, _, _ = quant.quant(
                    hy, o_bits, scale_o, mode=mode, quant_data='output')
                hy = quant.dequant(q_hy_outputs, scale_o)
            if dump:
                if bias_ih is not None and bias_hh is not None and o_bits is not None:
                    name_list = ["input", "hidden", "q_input", "q_hidden", 'q_iweight', 'q_hweight', "gi", "gh", "q_gi_outputs", "q_gh_outputs", "q_ibias",
                                 "q_hbias", "scale_i", "scale_iw", "scale_h", "scale_hw", "ingate", "forgetgate", "cellgate", "outgate", "output", "q_hy_outputs", "gates"]
                    attr_list = [input, hidden, q_input, q_hidden, q_iweight, q_hweight, gi, gh, q_gi_outputs, q_gh_outputs, q_ibias,
                                 q_hbias, scale_i, scale_iw, scale_h, scale_hw, ingate, forgetgate, cellgate, outgate,  hy, q_hy_outputs, gates]
                    Dump.dump_file(prefix, ".LstmInt.", zip(
                        name_list, attr_list), path)
                else:
                    name_list = ["input", "hidden", "q_input", "q_hidden", 'q_iweight', 'q_hweight',  "gi", "gh", "q_gi_outputs",
                                 "q_gh_outputs", "scale_i", "scale_iw", "scale_h", "scale_hw", "ingate", "forgetgate", "cellgate", "outgate", "output"]
                    attr_list = [input, hidden, q_input, q_hidden, q_iweight, q_hweight, gi, gh, q_gi_outputs,
                                 q_gh_outputs, scale_i, scale_iw, scale_h, scale_hw, ingate, forgetgate, cellgate, outgate, hy]
                    Dump.dump_file(prefix, ".LstmInt.", zip(
                        name_list, attr_list), path)
        return hy, cy

    @staticmethod
    def backward(ctx, grad_hy, grad_hc):
        input, hidden, weight_ih, weight_hh, input_bias, hidden_bias, input_gates, hidden_gates, cx, cy, hy = ctx.saved_tensors
        clamp_data = ctx.clamp_data
        o_bits = ctx.o_bits
        hy.requires_grad_(True)
        hidden.requires_grad_(True)
        with torch.enable_grad():
            if o_bits is not None:
                clamp_hy = normalize_data_with_config(hy, clamp_data)
            else:
                clamp_hy = hy.clone()
            grad_hy, = torch.autograd.grad(clamp_hy, hy, grad_hy)
        grad_input_gates, grad_hidden_gates, grad_cx, grad_input_bias, grad_hidden_bias = lingerext.lstm_cell_backward(
            grad_hy, grad_hc, input_gates, hidden_gates, input_bias, hidden_bias, cx, cy)

        grad_in = grad_input_gates.matmul(weight_ih)
        grad_hx = grad_hidden_gates.matmul(weight_hh)
        grad_w_ih = grad_input_gates.t().matmul(input)
        grad_h_ih = grad_hidden_gates.t().matmul(hidden)

        return grad_in, grad_hx, grad_cx, grad_w_ih, grad_h_ih, grad_input_bias, grad_hidden_bias, None, None, None, None, None, None, None, None, \
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class LSTMSingleONNXFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lengths, hidden_state, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse,
                input_size, hidden_size, num_layers, batch_first, dropout, bidirectional,
                data_bits, parameter_bits, o_bits,
                scale_i, scale_iw, scale_io, scale_h, scale_hw, scale_ho, scale_o,
                scale_i_reverse, scale_iw_reverse, scale_io_reverse, scale_h_reverse, scale_hw_reverse, scale_ho_reverse, scale_o_reverse,
                is_not_from_iqtensor):
        output = None
        hidden_state = None
        cell_state = None
        batch_size = None
        seq_length = None
        num_directions = 2 if bidirectional else 1
        if batch_first:
            batch_size = input.size(0)
            seq_length = input.size(
                1) if lengths is None else torch.max(lengths)
            output = torch.randn(batch_size, seq_length,
                                 hidden_size*num_directions, device=input.device)
        else:
            batch_size = input.size(1)
            seq_length = input.size(
                0) if lengths is None else torch.max(lengths)
            output = torch.randn(seq_length, batch_size,
                                 hidden_size*num_directions, device=input.device)
        hidden_state = torch.zeros(
            num_directions, batch_size, hidden_size, device=input.device)
        cell_state = torch.zeros(
            num_directions, batch_size, hidden_size, device=input.device)
        return output, hidden_state, cell_state

    @staticmethod
    def backward(ctx, gradOutput, gradHidden, gradCell):

        return None, None, None, None, None, None, None,\
            None, None, None, None,\
            None, None, None, None, None, None,\
            None, None, None,\
            None, None, None, None, None, None, None,\
            None, None, None, None, None, None, None,\
            None, None, None, None

    @staticmethod
    def symbolic(g, input, lengths, hidden_state, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                 weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse,
                 input_size, hidden_size, num_layers, batch_first, dropout, bidirectional,
                 data_bits, parameter_bits, o_bits,
                 scale_i, scale_iw, scale_io, scale_h, scale_hw, scale_ho, scale_o,
                 scale_i_reverse, scale_iw_reverse, scale_io_reverse, scale_h_reverse, scale_hw_reverse, scale_ho_reverse, scale_o_reverse,
                 is_not_from_iqtensor):

        param_dict = {'input_size_i': input_size, 'hidden_size_i': hidden_size, 'num_layers_i': num_layers,
                      'batch_first_i': batch_first, 'dropout_f': 0, 'go_forward_i': True,
                      'scale_i_f': scale_i(), 'scale_h_f': scale_h(), 'scale_iw_f': scale_iw(), 'scale_hw_f': scale_hw(),
                      'data_bits_i': data_bits, 'parameter_bits_i': parameter_bits,
                      }
        param_back_dict = {}
        if bidirectional:
            param_back_dict = {'input_size_i': input_size, 'hidden_size_i': hidden_size, 'num_layers_i': num_layers,
                               'batch_first_i': batch_first, 'dropout_f': 0, 'go_forward_i': False,
                               'scale_i_f': scale_i_reverse(), 'scale_h_f': scale_h_reverse(), 'scale_iw_f': scale_iw_reverse(), 'scale_hw_f': scale_hw_reverse(),
                               'data_bits_i': data_bits, 'parameter_bits_i': parameter_bits,
                               }
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        op_inner = None
        input_list = None
        input_back_list = None
        if is_not_from_iqtensor:
            op_inner = quantlinear(g, input, scale_i(),
                                   platform_quant, data_bits)
            input_list = [op_inner, weight_ih, weight_hh]
            input_back_list = [op_inner, weight_ih_reverse, weight_hh_reverse]
        else:
            input_list = [input, weight_ih, weight_hh]
            input_back_list = [input, weight_ih_reverse, weight_hh_reverse]
        if bias_ih is not None and bias_hh is not None:
            input_list.append(bias_ih)
            input_list.append(bias_hh)
            input_back_list.append(bias_ih_reverse)
            input_back_list.append(bias_hh_reverse)
        if o_bits is not None:
            param_dict['scale_o_f'] = scale_o()
            param_dict['o_bits_i'] = o_bits
            if bidirectional:
                param_back_dict['scale_o_f'] = scale_o_reverse()
                param_back_dict['o_bits_i'] = o_bits

        param_dict['platform_quant_s'] = platform_quant
        param_dict['outputs'] = 3
        param_back_dict['platform_quant_s'] = platform_quant
        param_back_dict['outputs'] = 3
        if lengths is None and hidden_state is None:
            lstm, hidden, cell = g.op(
                "thinker::LSTMInt", *input_list, **param_dict)
            if bidirectional:
                lstm_backward, hidden_back, cell_back = g.op(
                    "thinker::LSTMInt", *input_back_list, **param_back_dict)
                if o_bits is not None:
                    scale_o = ScalerBuffer(min(scale_o(), scale_o_reverse()))
                    args = [lstm, lstm_backward, scale_o, scale_o_reverse]
                    lstm = iqcat_sym(g, None, scale_o, None, 2,
                                     None, False, None, None, None, *args)
                else:
                    args = [lstm, lstm_backward]
                    lstm = g.op("Concat", *args, axis_i=2)
                hidden = g.op("Concat", hidden, hidden_back, axis_i=0)
                cell = g.op("Concat", cell, cell_back, axis_i=0)
        elif lengths is not None and hidden_state is None:
            input_list.insert(1, lengths)
            input_back_list.insert(1, lengths)
            lstm, hidden, cell = g.op(
                "thinker::LSTMInt", *input_list, **param_dict)
            if bidirectional:
                lstm_backward, hidden_back, cell_back = g.op(
                    "thinker::LSTMInt", *input_back_list, **param_back_dict)
                if o_bits is not None:
                    scale_o = ScalerBuffer(min(scale_o(), scale_o_reverse()))
                    args = [lstm, lstm_backward, scale_o, scale_o_reverse]
                    lstm = iqcat_sym(g, None, scale_o, None, 2,
                                     None, False, None, None, None, *args)
                else:
                    args = [lstm, lstm_backward]
                    lstm = g.op("Concat", *args, axis_i=2)
                hidden = g.op("Concat", hidden, hidden_back, axis_i=0)
                cell = g.op("Concat", cell, cell_back, axis_i=0)
        else:
            input_list.insert(1, lengths)
            input_list.insert(2, hidden_state)
            input_list.insert(3, cell_state)
            input_back_list.insert(1, lengths)
            input_back_list.insert(2, hidden_state)
            input_back_list.insert(3, cell_state)
            lstm, hidden, cell = g.op(
                "thinker::LSTMInt", *input_list, **param_dict)
            if bidirectional:
                lstm_backward, hidden_back, cell_back = g.op(
                    "thinker::LSTMInt", *input_back_list, **param_back_dict)
                if o_bits is not None:
                    scale_o = ScalerBuffer(min(scale_o(), scale_o_reverse()))
                    args = [lstm, lstm_backward, scale_o, scale_o_reverse]
                    lstm = iqcat_sym(g, None, scale_o, None, 2,
                                     None, False, None, None, None, *args)
                else:
                    args = [lstm, lstm_backward]
                    lstm = g.op("Concat", *args, axis_i=2)
                hidden = g.op("Concat", hidden, hidden_back, axis_i=0)
                cell = g.op("Concat", cell, cell_back, axis_i=0)

        return lstm, hidden, cell


class LSTMInt(nn.LSTM):
    r"""实现LSTMInt的量化训练与测试，继承自nn.LSTM,

    Args:
        input_size hidden_size num_layers bias batch_first dropout bidirectional
        与nn.GRU一致的参数
        unified(bool): 确认正反向参数统计是否一致
        data_bits(int): 输入量化位数
        parameter_bits(int): 参数量化位数
        mode(Enum): 量化方式，支持MaxValue与Qvalue
        o_bits(int, default=None):输出量化位数
        scale_i(np.float32): 统计的是LSTMP的输入scale，输入大小为(b, t, d)或(t, b, d)
        scale_h(np.float32): 统计的是每一帧计算隐藏输出的最值momentum统计得到的scale
        scale_iw(np.float32): 依据最终的模型参数计算得到，无统计
        scale_hw(np.float32): 依据最终模型参数计算得到，无统计参数
        scale_o(np.float32): 最终输出的统计scale
        scale_reverse_*(np.float32):对应反向过程中各个scale数值
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, unified=True,
                 data_bits=8, parameter_bits=8, mode=QuantMode.QValue, o_bits=None, clamp_data=None, clamp_weight=None, clamp_bias=None):
        nn.LSTM.__init__(self, input_size, hidden_size, num_layers,
                         bias, batch_first, dropout, bidirectional)
        ModuleIntConfig.__init__(
            self, data_bits=data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=o_bits)
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.unified = unified
        self.momentum = 0.1
        self.is_not_from_iqtensor = True
        self.clamp_data = clamp_data
        self.clamp_weight = clamp_weight
        self.clamp_bias = clamp_bias

        self.register_buffer('running_i', torch.zeros(1))
        self.register_buffer('running_h', torch.zeros(1))
        self.register_buffer('running_iw', torch.zeros(1))
        self.register_buffer('running_hw', torch.zeros(1))
        self.register_buffer('running_io', torch.zeros(1))
        self.register_buffer('running_ho', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))

        self.register_buffer('scale_i', torch.zeros(1))
        self.register_buffer('scale_h', torch.zeros(1))
        self.register_buffer('scale_iw', torch.zeros(1))
        self.register_buffer('scale_hw', torch.zeros(1))
        self.register_buffer('scale_io', torch.zeros(1))
        self.register_buffer('scale_ho', torch.zeros(1))
        self.register_buffer('scale_o', torch.zeros(1))
        if self.bidirectional:
            self.register_buffer('running_i_reverse', torch.zeros(1))
            self.register_buffer('running_h_reverse', torch.zeros(1))
            self.register_buffer('running_iw_reverse', torch.zeros(1))
            self.register_buffer('running_hw_reverse', torch.zeros(1))
            self.register_buffer('running_io_reverse', torch.zeros(1))
            self.register_buffer('running_ho_reverse', torch.zeros(1))
            self.register_buffer('running_o_reverse', torch.zeros(1))

            self.register_buffer('scale_i_reverse', torch.zeros(1))
            self.register_buffer('scale_h_reverse', torch.zeros(1))
            self.register_buffer('scale_iw_reverse', torch.zeros(1))
            self.register_buffer('scale_hw_reverse', torch.zeros(1))
            self.register_buffer('scale_io_reverse', torch.zeros(1))
            self.register_buffer('scale_ho_reverse', torch.zeros(1))
            self.register_buffer('scale_o_reverse', torch.zeros(1))

        self.sigmoid_table = None
        self.tanh_table = None

    def _single_direction_tensor(self, input, hidden, cell_state, layer=0, direct=0):
        step_outputs = []
        weight_ih = self.weight_ih_l0_reverse if direct == 1 else self.weight_ih_l0
        weight_hh = self.weight_hh_l0_reverse if direct == 1 else self.weight_hh_l0
        bias_ih = self.bias_ih_l0_reverse if direct == 1 else self.bias_ih_l0
        bias_hh = self.bias_hh_l0_reverse if direct == 1 else self.bias_hh_l0
        if weight_ih.dtype == torch.float32:
            weight_ih = normalize_weight_with_config(
                weight_ih, self.clamp_weight, self.training)
            weight_hh = normalize_weight_with_config(
                weight_hh, self.clamp_weight, self.training)
            bias_ih = normalize_bias_with_config(
                bias_ih, self.clamp_bias, self.training)
            bias_hh = normalize_bias_with_config(
                bias_hh, self.clamp_bias, self.training)

        running_i_tensor = self.running_i_reverse if direct == 1 and not self.unified else self.running_i
        running_h_tensor = self.running_h_reverse if direct == 1 and not self.unified else self.running_h
        running_iw_tensor = self.running_iw_reverse if direct == 1 and not self.unified else self.running_iw
        running_hw_tensor = self.running_hw_reverse if direct == 1 and not self.unified else self.running_hw
        running_io_tensor = self.running_io_reverse if direct == 1 and not self.unified else self.running_io
        running_ho_tensor = self.running_ho_reverse if direct == 1 and not self.unified else self.running_ho
        running_o_tensor = self.running_o_reverse if direct == 1 and not self.unified else self.running_o

        scale_i_tensor = self.scale_i_reverse if direct == 1 and not self.unified else self.scale_i
        scale_h_tensor = self.scale_h_reverse if direct == 1 and not self.unified else self.scale_h
        scale_iw_tensor = self.scale_iw_reverse if direct == 1 and not self.unified else self.scale_iw
        scale_hw_tensor = self.scale_hw_reverse if direct == 1 and not self.unified else self.scale_hw
        scale_io_tensor = self.scale_io_reverse if direct == 1 and not self.unified else self.scale_io
        scale_ho_tensor = self.scale_ho_reverse if direct == 1 and not self.unified else self.scale_ho
        scale_o_tensor = self.scale_o_reverse if direct == 1 and not self.unified else self.scale_o

        running_i = ScalerBuffer(running_i_tensor)
        running_h = ScalerBuffer(running_h_tensor)
        running_iw = ScalerBuffer(running_iw_tensor)
        running_hw = ScalerBuffer(running_hw_tensor)
        running_io = ScalerBuffer(running_io_tensor)
        running_ho = ScalerBuffer(running_ho_tensor)
        running_o = ScalerBuffer(running_o_tensor)
        scale_i = ScalerBuffer(scale_i_tensor)
        scale_h = ScalerBuffer(scale_h_tensor)
        scale_iw = ScalerBuffer(scale_iw_tensor)
        scale_hw = ScalerBuffer(scale_hw_tensor)
        scale_io = ScalerBuffer(scale_io_tensor)
        scale_ho = ScalerBuffer(scale_ho_tensor)
        scale_o = ScalerBuffer(scale_o_tensor)

        input = torch.cat(input.split(1, 0)[::-1]) if direct == 1 else input
        if self.training:
            if self.is_not_from_iqtensor:
                max_value_ix = get_max_value(input)
                running_i.mul_(
                    1-self.momentum).add_(self.momentum*max_value_ix)

        for input_x in input:
            hidden, cell_state = LSTMCellFunction.apply(input_x, hidden, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                                                        self.data_bits, self.parameter_bits, self.o_bits,
                                                        running_i, running_h, running_iw, running_hw, running_io, running_ho, running_o,
                                                        scale_i, scale_h, scale_iw, scale_hw, scale_io, scale_ho, scale_o,
                                                        self.momentum, self.training, self.prefix, self.dump, self.path, self.quant_mode, self.quant,
                                                        self.is_not_from_iqtensor, self.clamp_data)
            step_outputs.append(hidden)

        running_i_tensor.fill_(running_i())
        running_h_tensor.fill_(running_h())
        running_iw_tensor.fill_(running_iw())
        running_hw_tensor.fill_(running_hw())
        running_io_tensor.fill_(running_io())
        running_ho_tensor.fill_(running_ho())
        running_o_tensor.fill_(running_o())
        scale_i_tensor.fill_(scale_i())
        scale_h_tensor.fill_(scale_h())
        scale_iw_tensor.fill_(scale_iw())
        scale_hw_tensor.fill_(scale_hw())
        scale_io_tensor.fill_(scale_io())
        scale_ho_tensor.fill_(scale_ho())
        scale_o_tensor.fill_(scale_o())

        step_outputs = step_outputs[::-1] if direct == 1 else step_outputs
        output = torch.stack(step_outputs, 0)
        return output, (hidden, cell_state)

    def _single_direction_packed(self, input, hidden, cell_state, layer=0, direct=0, batch_sizes=None):
        if direct:
            return self._packed_reverse(input, hidden, cell_state, layer, direct, batch_sizes)
        else:
            return self._packed_forward(input, hidden, cell_state, layer, direct, batch_sizes)

    def _packed_forward(self, input, hidden, cell_state, layer=0, direct=0, batch_sizes=None):
        step_outputs = []
        final_hiddens = []
        weight_ih = self.weight_ih_l0_reverse if direct == 1 else self.weight_ih_l0
        weight_hh = self.weight_hh_l0_reverse if direct == 1 else self.weight_hh_l0
        bias_ih = self.bias_ih_l0_reverse if direct == 1 else self.bias_ih_l0
        bias_hh = self.bias_hh_l0_reverse if direct == 1 else self.bias_hh_l0
        if weight_ih.dtype == torch.float32:
            weight_ih = normalize_weight_with_config(
                weight_ih, self.clamp_weight, self.training)
            weight_hh = normalize_weight_with_config(
                weight_hh, self.clamp_weight, self.training)
            bias_ih = normalize_bias_with_config(
                bias_ih, self.clamp_bias, self.training)
            bias_hh = normalize_bias_with_config(
                bias_hh, self.clamp_bias, self.training)

        running_i_tensor = self.running_i_reverse if direct == 1 and not self.unified else self.running_i
        running_h_tensor = self.running_h_reverse if direct == 1 and not self.unified else self.running_h
        running_iw_tensor = self.running_iw_reverse if direct == 1 and not self.unified else self.running_iw
        running_hw_tensor = self.running_hw_reverse if direct == 1 and not self.unified else self.running_hw
        running_io_tensor = self.running_io_reverse if direct == 1 and not self.unified else self.running_io
        running_ho_tensor = self.running_ho_reverse if direct == 1 and not self.unified else self.running_ho
        running_o_tensor = self.running_o_reverse if direct == 1 and not self.unified else self.running_o

        scale_i_tensor = self.scale_i_reverse if direct == 1 and not self.unified else self.scale_i
        scale_h_tensor = self.scale_h_reverse if direct == 1 and not self.unified else self.scale_h
        scale_iw_tensor = self.scale_iw_reverse if direct == 1 and not self.unified else self.scale_iw
        scale_hw_tensor = self.scale_hw_reverse if direct == 1 and not self.unified else self.scale_hw
        scale_io_tensor = self.scale_io_reverse if direct == 1 and not self.unified else self.scale_io
        scale_ho_tensor = self.scale_ho_reverse if direct == 1 and not self.unified else self.scale_ho
        scale_o_tensor = self.scale_o_reverse if direct == 1 and not self.unified else self.scale_o

        running_i = ScalerBuffer(running_i_tensor)
        running_h = ScalerBuffer(running_h_tensor)
        running_iw = ScalerBuffer(running_iw_tensor)
        running_hw = ScalerBuffer(running_hw_tensor)
        running_io = ScalerBuffer(running_io_tensor)
        running_ho = ScalerBuffer(running_ho_tensor)
        running_o = ScalerBuffer(running_o_tensor)
        scale_i = ScalerBuffer(scale_i_tensor)
        scale_h = ScalerBuffer(scale_h_tensor)
        scale_iw = ScalerBuffer(scale_iw_tensor)
        scale_hw = ScalerBuffer(scale_hw_tensor)
        scale_io = ScalerBuffer(scale_io_tensor)
        scale_ho = ScalerBuffer(scale_ho_tensor)
        scale_o = ScalerBuffer(scale_o_tensor)

        hidden = copy.deepcopy(hidden)
        cell_state = copy.deepcopy(cell_state)
        input, batch_size_list = _unbind_packed(input, batch_sizes)
        last_batch_size = batch_size_list[0]
        if self.training:
            if self.is_not_from_iqtensor:
                max_value_ix = get_max_value(input)
                running_i.mul_(
                    1-self.momentum).add_(self.momentum*max_value_ix)

        for input_i, batch_len in zip(input, batch_size_list):
            inc = batch_len - last_batch_size
            if inc < 0:
                # 按batch的帧长排完序，由长到短，较短的帧hidden计算的次数少，直接取低位保留
                final_hiddens.append(
                    _slice((hidden, cell_state), batch_len, last_batch_size))
                hidden, cell_state = hx_slice(
                    None, (hidden, cell_state), last_batch_size, batch_len)
            hidden, cell_state = LSTMCellFunction.apply(input_i, hidden, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                                                        self.data_bits, self.parameter_bits, self.o_bits,
                                                        running_i, running_h, running_iw, running_hw, running_io, running_ho, running_o,
                                                        scale_i, scale_h, scale_iw, scale_hw, scale_io, scale_ho, scale_o,
                                                        self.momentum, self.training, self.prefix, self.dump, self.path, self.quant_mode, self.quant,
                                                        self.is_not_from_iqtensor, self.clamp_data)
            step_outputs.append(hidden)
            last_batch_size = batch_len

        running_i_tensor.fill_(running_i())
        running_h_tensor.fill_(running_h())
        running_iw_tensor.fill_(running_iw())
        running_hw_tensor.fill_(running_hw())
        running_io_tensor.fill_(running_io())
        running_ho_tensor.fill_(running_ho())
        running_o_tensor.fill_(running_o())
        scale_i_tensor.fill_(scale_i())
        scale_h_tensor.fill_(scale_h())
        scale_iw_tensor.fill_(scale_iw())
        scale_hw_tensor.fill_(scale_hw())
        scale_io_tensor.fill_(scale_io())
        scale_ho_tensor.fill_(scale_ho())
        scale_o_tensor.fill_(scale_o())

        final_hiddens.append((hidden, cell_state))
        ret_hidden = final_hiddens[::-1]
        hy_list = []
        cy_list = []
        for each in ret_hidden:
            hy_list.append(each[0])
            cy_list.append(each[1])
        hidden = torch.cat(hy_list, 0)
        cell_state = torch.cat(cy_list, 0)
        output = torch.cat(step_outputs, 0)
        return output, (hidden, cell_state)

    def _packed_reverse(self, input, hidden, cell_state, layer=0, direct=0, batch_sizes=None):
        step_outputs = []
        weight_ih = self.weight_ih_l0_reverse if direct == 1 else self.weight_ih_l0
        weight_hh = self.weight_hh_l0_reverse if direct == 1 else self.weight_hh_l0
        bias_ih = self.bias_ih_l0_reverse if direct == 1 else self.bias_ih_l0
        bias_hh = self.bias_hh_l0_reverse if direct == 1 else self.bias_hh_l0
        if weight_ih.dtype == torch.float32:
            weight_ih = normalize_weight_with_config(
                weight_ih, self.clamp_weight, self.training)
            weight_hh = normalize_weight_with_config(
                weight_hh, self.clamp_weight, self.training)
            bias_ih = normalize_bias_with_config(
                bias_ih, self.clamp_bias, self.training)
            bias_hh = normalize_bias_with_config(
                bias_hh, self.clamp_bias, self.training)

        running_i_tensor = self.running_i_reverse if direct == 1 and not self.unified else self.running_i
        running_h_tensor = self.running_h_reverse if direct == 1 and not self.unified else self.running_h
        running_iw_tensor = self.running_iw_reverse if direct == 1 and not self.unified else self.running_iw
        running_hw_tensor = self.running_hw_reverse if direct == 1 and not self.unified else self.running_hw
        running_io_tensor = self.running_io_reverse if direct == 1 and not self.unified else self.running_io
        running_ho_tensor = self.running_ho_reverse if direct == 1 and not self.unified else self.running_ho
        running_o_tensor = self.running_o_reverse if direct == 1 and not self.unified else self.running_o

        scale_i_tensor = self.scale_i_reverse if direct == 1 and not self.unified else self.scale_i
        scale_h_tensor = self.scale_h_reverse if direct == 1 and not self.unified else self.scale_h
        scale_iw_tensor = self.scale_iw_reverse if direct == 1 and not self.unified else self.scale_iw
        scale_hw_tensor = self.scale_hw_reverse if direct == 1 and not self.unified else self.scale_hw
        scale_io_tensor = self.scale_io_reverse if direct == 1 and not self.unified else self.scale_io
        scale_ho_tensor = self.scale_ho_reverse if direct == 1 and not self.unified else self.scale_ho
        scale_o_tensor = self.scale_o_reverse if direct == 1 and not self.unified else self.scale_o

        running_i = ScalerBuffer(running_i_tensor)
        running_h = ScalerBuffer(running_h_tensor)
        running_iw = ScalerBuffer(running_iw_tensor)
        running_hw = ScalerBuffer(running_hw_tensor)
        running_io = ScalerBuffer(running_io_tensor)
        running_ho = ScalerBuffer(running_ho_tensor)
        running_o = ScalerBuffer(running_o_tensor)
        scale_i = ScalerBuffer(scale_i_tensor)
        scale_h = ScalerBuffer(scale_h_tensor)
        scale_iw = ScalerBuffer(scale_iw_tensor)
        scale_hw = ScalerBuffer(scale_hw_tensor)
        scale_io = ScalerBuffer(scale_io_tensor)
        scale_ho = ScalerBuffer(scale_ho_tensor)
        scale_o = ScalerBuffer(scale_o_tensor)

        input, batch_size_list = _unbind_packed(input, batch_sizes)
        input = input[::-1]  # 按照时间t 进行反转
        batch_size_list = batch_size_list[::-1]
        input_hx = (copy.deepcopy(hidden), copy.deepcopy(cell_state))
        last_batch_size = batch_size_list[0]
        if self.training:
            if self.is_not_from_iqtensor:
                max_value_ix = get_max_value(input)
                running_i.mul_(
                    1-self.momentum).add_(self.momentum*max_value_ix)
        hidden = _slice(hidden, 0, last_batch_size)
        cell_state = _slice(cell_state, 0, last_batch_size)
        for input_i, batch_len in zip(input, batch_size_list):
            if last_batch_size != batch_len:
                # 获取input_hx高位hidden部分与上一帧的hidden进行填充，相当于补0
                hidden, cell_state = hx_slice(
                    input_hx, (hidden, cell_state), last_batch_size, batch_len)
            hidden, cell_state = LSTMCellFunction.apply(input_i, hidden, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                                                        self.data_bits, self.parameter_bits, self.o_bits,
                                                        running_i, running_h, running_iw, running_hw, running_io, running_ho, running_o,
                                                        scale_i, scale_h, scale_iw, scale_hw, scale_io, scale_ho, scale_o,
                                                        self.momentum, self.training, self.prefix, self.dump, self.path, self.quant_mode, self.quant,
                                                        self.is_not_from_iqtensor, self.clamp_data)
            step_outputs.append(hidden)
            last_batch_size = batch_len

        running_i_tensor.fill_(running_i())
        running_h_tensor.fill_(running_h())
        running_iw_tensor.fill_(running_iw())
        running_hw_tensor.fill_(running_hw())
        running_io_tensor.fill_(running_io())
        running_ho_tensor.fill_(running_ho())
        running_o_tensor.fill_(running_o())
        scale_i_tensor.fill_(scale_i())
        scale_h_tensor.fill_(scale_h())
        scale_iw_tensor.fill_(scale_iw())
        scale_hw_tensor.fill_(scale_hw())
        scale_io_tensor.fill_(scale_io())
        scale_ho_tensor.fill_(scale_ho())
        scale_o_tensor.fill_(scale_o())

        step_outputs = step_outputs[::-1]
        output = torch.cat(step_outputs, 0)
        return output, (hidden, cell_state)

    def _finetune(self, input, hidden, cell_state, layer=0, direct=0, batch_sizes=None):
        if batch_sizes is None:
            return self._single_direction_tensor(input, hidden, cell_state, layer, direct)
        else:
            return self._single_direction_packed(input, hidden, cell_state, layer, direct, batch_sizes)

    def _run_single_direction(self, input, hidden, cell_state, layer=0, direct=0, batch_sizes=None):
        return self._finetune(input, hidden, cell_state, layer, direct, batch_sizes)

    def single_direction(self, input, layer, hx, batch_sizes=None):
        hidden = hx[0]
        cell_state = hx[1]
        output, hidden = self._run_single_direction(
            input, hidden, cell_state, layer, direct=0, batch_sizes=batch_sizes)
        return output, [hidden]

    def bidirection(self, input, layer, hx, batch_sizes=None):
        hx_f = hx[0][0]
        ct_f = hx[0][1]
        hx_b = hx[1][0]
        ct_b = hx[1][1]
        fw_output, fw_hidden = self._run_single_direction(
            input, hx_f, ct_f, layer, direct=0, batch_sizes=batch_sizes)
        rev_output, rev_hidden = self._run_single_direction(
            input, hx_b, ct_b, layer, direct=1, batch_sizes=batch_sizes)
        if batch_sizes is None:
            output = torch.cat((fw_output, rev_output), fw_output.dim()-1)
        else:  # packed sequence
            output = torch.cat((fw_output, rev_output), -1)
        return output, [fw_hidden, rev_hidden]

    def lstm_forward(self, input, hiddens, batch_sizes=None):
        final_hiddens = []
        for layer_num in range(self.num_layers):
            hid = hiddens[layer_num] if hiddens is not None else None
            output, hc = self.bidirection(input, layer_num, hid, batch_sizes) if self.bidirectional else self.single_direction(
                input, layer_num, hid, batch_sizes)
            final_hiddens.extend(hc)
            input = output
            # add dropout
            if (self.dropout != 0 and self.training and layer_num < self.num_layers - 1):
                input = torch.nn.functional.dropout(input, self.dropout)
        hy = [hidden[0] for hidden in final_hiddens]
        cy = [hidden[1] for hidden in final_hiddens]
        hy = torch.stack(hy, 0)
        cy = torch.stack(cy, 0)
        return input, hy, cy

    def _generate_hiddens(self, hx):
        if hx is not None:
            assert len(hx) == 2, 'hidden(tuple) input length must be 2'
            hidden_list = _unbind(hx[0])
            cellstate_list = _unbind(hx[1])
            assert len(hidden_list) == len(cellstate_list)
            length = len(hidden_list)
            if self.bidirectional:
                assert length/self.num_layers % 2 == 0, 'hidden len must be double in bidirectional mode'

            i = 0
            hiddens = []
            while i < length:
                if self.bidirectional:
                    hiddens.append(
                        ((hidden_list[i], cellstate_list[i]), (hidden_list[i+1], cellstate_list[i+1])))
                    i += 2
                else:
                    hiddens.append((hidden_list[i], cellstate_list[i]))
                    i += 1
        else:
            hiddens = None
        return hiddens

    def forward_input_tensor(self, input, hx, batch_sizes=None):
        input = input.transpose(0, 1) if self.batch_first else input
        hiddens = self._generate_hiddens(hx)
        output, hr, ct = self.lstm_forward(input, hiddens)
        output = output.transpose(0, 1) if self.batch_first else output
        return output, hr, ct

    def forward_input_packed(self, input, hx, batch_sizes=None):
        hiddens = self._generate_hiddens(hx)
        output, hr, ct = self.lstm_forward(input, hiddens, batch_sizes)
        return output, hr, ct

    def forward(self, input, hx=None):
        orig_input = input
        if not is_in_onnx_export():
            if isinstance(orig_input, tuple):
                input, lengths, batch_first, enforce_sorted = orig_input
                if isinstance(input, IQTensor):
                    self.is_not_from_iqtensor = False
                    if input.bits != self.data_bits:
                        input = Requant.apply(
                            input, input.bits, input.scale_data, self.data_bits)
                    self.scale_i.fill_(input.scale_data)
                    self.running_i.fill_(input.running_data)
                    if self.bidirectional:
                        self.scale_i_reverse.fill_(input.scale_data)
                        self.running_i_reverse.fill_(input.running_data)
                packed_input = torch_pack_padded_sequence(
                    input, lengths, batch_first, enforce_sorted)
                input, batch_sizes, sorted_indices, unsorted_indices = packed_input
                max_batch_size = batch_sizes[0]
                max_batch_size = int(max_batch_size)
            else:
                batch_sizes = None
                max_batch_size = input.size(
                    0) if self.batch_first else input.size(1)
                sorted_indices = None
                unsorted_indices = None
                if isinstance(input, IQTensor):
                    self.is_not_from_iqtensor = False
                    if input.bits != self.data_bits:
                        input = Requant.apply(
                            input, input.bits, input.scale_data, self.data_bits)
                    self.scale_i.fill_(input.scale_data)
                    self.running_i.fill_(input.running_data)
                    if self.bidirectional:
                        self.scale_i_reverse.fill_(input.scale_data)
                        self.running_i_reverse.fill_(input.running_data)
            assert self.num_layers == 1, 'invalid num_layers, now only support num_layers = 1'
            if hx is None:
                num_directions = 2 if self.bidirectional else 1
                zeros = torch.zeros(self.num_layers * num_directions,
                                    max_batch_size, self.hidden_size,
                                    dtype=input.dtype, device=input.device)
                hx = (zeros, zeros)
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)

            self.check_forward_args(input, hx, batch_sizes)
            if batch_sizes is not None:
                output, hy, cy = self.forward_input_packed(
                    input, hx, batch_sizes)
            else:
                output, hy, cy = self.forward_input_tensor(input, hx)
            hidden = (hy, cy)

            if isinstance(orig_input, tuple):
                output_packed = PackedSequence(
                    output, batch_sizes, sorted_indices, unsorted_indices)
                output, lengths = torch_pad_packed_sequence(
                    output_packed, self.batch_first)
                if self.o_bits is not None:
                    if self.training:
                        output = Quant2IQTensor.apply(
                            output, self.o_bits, self.quant_mode, 'output')
                    else:
                        scale_o = ScalerBuffer(self.quant.running_to_scale(
                            self.running_o, self.o_bits, mode=self.quant_mode))
                        if self.bidirectional:
                            if self.unified:
                                scale_o_reverse = scale_o
                            else:
                                scale_o_reverse = ScalerBuffer(self.quant.running_to_scale(
                                    self.running_o_reverse, self.o_bits, mode=self.quant_mode))
                            scale_o = ScalerBuffer(
                                min(scale_o(), scale_o_reverse()))
                        output = from_torch_tensor(
                            output, scale_o(), self.o_bits)
                return (output, lengths), self.permute_hidden(hidden, unsorted_indices)
            else:
                if self.o_bits is not None:
                    if self.training:
                        output = Quant2IQTensor.apply(
                            output, self.o_bits, self.quant_mode, 'output')
                    else:
                        scale_o = ScalerBuffer(self.quant.running_to_scale(
                            self.running_o, self.o_bits, mode=self.quant_mode))
                        if self.bidirectional:
                            if self.unified:
                                scale_o_reverse = scale_o
                            else:
                                scale_o_reverse = ScalerBuffer(self.quant.running_to_scale(
                                    self.running_o_reverse, self.o_bits, mode=self.quant_mode))
                            scale_o = ScalerBuffer(
                                min(scale_o(), scale_o_reverse()))
                        output = from_torch_tensor(
                            output, scale_o(), self.o_bits)
                return output, self.permute_hidden(hidden, unsorted_indices)
        else:
            lengths = None
            if isinstance(orig_input, tuple):
                input, lengths, _, _ = orig_input
            else:
                input = orig_input
                lengths = None
            if isinstance(input, IQTensor):
                self.is_not_from_iqtensor = False
                if input.bits != self.data_bits:
                    input = Requant.apply(
                        input, input.bits, input.scale_data, self.data_bits)
            bias_ih = None
            bias_hh = None
            bias_ih_reverse = None
            bias_hh_reverse = None
            weight_ih = self.weight_ih_l0
            weight_hh = self.weight_hh_l0
            weight_ih_reverse = weight_ih
            weight_hh_reverse = weight_hh
            if self.bias:
                bias_ih = self.bias_ih_l0
                bias_hh = self.bias_hh_l0
                bias_ih_reverse = bias_ih
                bias_hh_reverse = bias_hh
            if self.bidirectional:
                weight_ih_reverse = self.weight_ih_l0_reverse
                weight_hh_reverse = self.weight_hh_l0_reverse
                bias_ih_reverse = self.bias_ih_l0_reverse
                bias_hh_reverse = self.bias_hh_l0_reverse

            scale_i = ScalerBuffer(self.scale_i)
            scale_iw = ScalerBuffer(self.scale_iw)
            scale_io = ScalerBuffer(self.scale_io)
            scale_h = ScalerBuffer(self.scale_h)
            scale_hw = ScalerBuffer(self.scale_hw)
            scale_ho = ScalerBuffer(self.scale_ho)
            scale_o = ScalerBuffer(self.scale_o)
            scale_i_reverse = None
            scale_iw_reverse = None
            scale_io_reverse = None
            scale_h_reverse = None
            scale_hw_reverse = None
            scale_ho_reverse = None
            scale_o_reverse = None
            hidden_state = None
            cell_state = None
            if self.bidirectional:
                scale_i_reverse = ScalerBuffer(self.scale_i_reverse)
                scale_iw_reverse = ScalerBuffer(self.scale_iw_reverse)
                scale_io_reverse = ScalerBuffer(self.scale_io_reverse)
                scale_h_reverse = ScalerBuffer(self.scale_h_reverse)
                scale_hw_reverse = ScalerBuffer(self.scale_hw_reverse)
                scale_ho_reverse = ScalerBuffer(self.scale_ho_reverse)
                scale_o_reverse = ScalerBuffer(self.scale_o_reverse)
            if hx is not None:
                hidden_state, cell_state = hx
            output = None
            hy = None
            cy = None
            if hx is not None:
                batch_size = input.size(
                    0) if self.batch_first else input.size(1)
                seq_len = input.size(
                    1) if self.batch_first else input.size(0)
                lengths = torch.tensor([seq_len for i in range(
                    batch_size)], dtype=torch.int64, device=input.device) if lengths is None else lengths
            output, hy, cy = LSTMSingleONNXFunction.apply(input, lengths, hidden_state, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                                                          weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse,
                                                          self.input_size, self.hidden_size, self.num_layers, self.batch_first, self.dropout, self.bidirectional,
                                                          self.data_bits, self.parameter_bits, self.o_bits,
                                                          scale_i, scale_iw, scale_io, scale_h, scale_hw, scale_ho, scale_o,
                                                          scale_i_reverse, scale_iw_reverse, scale_io_reverse, scale_h_reverse, scale_hw_reverse, scale_ho_reverse, scale_o_reverse,
                                                          self.is_not_from_iqtensor)
            if self.o_bits is not None:
                if self.bidirectional:
                    scale_o = ScalerBuffer(min(scale_o(), scale_o_reverse()))
                output = from_torch_tensor(output, scale_o(), self.o_bits)
            if isinstance(orig_input, tuple):
                return (output, lengths), (hy, cy)
            else:
                return output, (hy, cy)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]
                              ] = local_metadata = dict(version=self._version)
        if is_in_onnx_export():
            assert self.running_i > 0, 'invalid running_x <=0'
            scale_i = ScalerBuffer(self.scale_i.data)
            if self.is_not_from_iqtensor:
                scale_i = ScalerBuffer(self.quant.running_to_scale(
                    self.running_i, self.data_bits, mode=self.quant_mode))
                self.scale_i.data.fill_(scale_i())
            scale_h = ScalerBuffer(self.quant.running_to_scale(
                self.running_h, self.data_bits, mode=self.quant_mode))
            self.scale_h.data.fill_(scale_h())
            if self.o_bits is not None:
                scale_o = ScalerBuffer(self.quant.running_to_scale(
                    self.running_o, self.o_bits, mode=self.quant_mode))
                self.scale_o.data.fill_(scale_o())

            if self.bidirectional:
                if self.unified:
                    self.running_i_reverse.data = self.running_i.data
                    self.running_h_reverse.data = self.running_h.data
                    self.scale_i_reverse.data = self.scale_i.data
                    self.scale_h_reverse.data = self.scale_h.data
                    scale_i_reverse = scale_i
                    scale_h_reverse = scale_h
                    if self.o_bits is not None:
                        self.running_o_reverse.data = self.running_o.data
                        self.scale_o_reverse.data = self.scale_o.data
                        scale_o_reverse = scale_o
                else:
                    scale_i_reverse = ScalerBuffer(self.scale_i_reverse.data)
                    if self.is_not_from_iqtensor:
                        scale_i_reverse = ScalerBuffer(self.quant.running_to_scale(
                            self.running_i_reverse, self.data_bits, mode=self.quant_mode))
                        self.scale_i_reverse.data.fill_(scale_i_reverse())
                    scale_h_reverse = ScalerBuffer(self.quant.running_to_scale(
                        self.running_h_reverse, self.data_bits, mode=self.quant_mode))
                    self.scale_h_reverse.data.fill_(scale_h_reverse())
                    if self.o_bits is not None:
                        scale_o_reverse = ScalerBuffer(self.quant.running_to_scale(
                            self.running_o_reverse, self.o_bits, mode=self.quant_mode))
                        self.scale_o_reverse.data.fill_(scale_o_reverse())

        if self.weight_ih_l0.dtype == torch.float32:
            clamp_weight_iw = normalize_weight_with_config(
                self.weight_ih_l0, self.clamp_weight, False)
            clamp_weight_hw = normalize_weight_with_config(
                self.weight_hh_l0, self.clamp_weight, False)
            self.weight_ih_l0.data = clamp_weight_iw
            self.weight_hh_l0.data = clamp_weight_hw

            if is_in_onnx_export():
                scale_iw = ScalerBuffer(self.quant.running_to_scale(
                    self.running_iw, self.parameter_bits, mode=self.quant_mode))
                self.scale_iw.data.fill_(scale_iw())
                scale_hw = ScalerBuffer(self.quant.running_to_scale(
                    self.running_hw, self.parameter_bits, mode=self.quant_mode))
                self.scale_hw.data.fill_(scale_hw())
                q_weight_iw, scale_iw, _ = self.quant.quant(
                    clamp_weight_iw, self.parameter_bits, scale=scale_iw, mode=self.quant_mode, quant_data='weight')
                q_weight_hw, scale_hw, _ = self.quant.quant(
                    clamp_weight_hw, self.parameter_bits, scale=scale_hw, mode=self.quant_mode, quant_data='weight')
            if self.bias:
                clamp_bias_iw = normalize_bias_with_config(
                    self.bias_ih_l0, self.clamp_bias, False)
                clamp_bias_hw = normalize_bias_with_config(
                    self.bias_hh_l0, self.clamp_bias, False)
                self.bias_ih_l0.data = clamp_bias_iw
                self.bias_hh_l0.data = clamp_bias_hw
                if is_in_onnx_export():
                    if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                        q_bias_iw = (clamp_bias_iw * scale_i *
                                     scale_iw + 0.5).floor()
                        q_bias_hw = (clamp_bias_hw * scale_h *
                                     scale_hw + 0.5).floor()
                        if self.data_bits + self.parameter_bits <= 16:
                            q_bias_iw = q_bias_iw.float().int()
                            q_bias_hw = q_bias_hw.float().int()
                    else:
                        assert False, "linger only support luna quant."
            if self.bidirectional:
                clamp_weight_iw_reverse = normalize_weight_with_config(
                    self.weight_ih_l0_reverse, self.clamp_weight, False)
                clamp_weight_hw_reverse = normalize_weight_with_config(
                    self.weight_hh_l0_reverse, self.clamp_weight, False)
                self.weight_ih_l0_reverse.data = clamp_weight_iw_reverse
                self.weight_hh_l0_reverse.data = clamp_weight_hw_reverse
                if is_in_onnx_export():
                    if self.unified:
                        q_weight_iw_reverse, scale_iw_reverse, _ = self.quant.quant(
                            clamp_weight_iw_reverse, self.parameter_bits, scale=scale_iw, mode=self.quant_mode, quant_data='weight')
                        q_weight_hw_reverse, scale_hw_reverse, _ = self.quant.quant(
                            clamp_weight_hw_reverse, self.parameter_bits, scale=scale_hw, mode=self.quant_mode, quant_data='weight')
                    else:
                        scale_iw_reverse = ScalerBuffer(self.quant.running_to_scale(
                            self.running_iw_reverse, self.parameter_bits, mode=self.quant_mode))
                        scale_hw_reverse = ScalerBuffer(self.quant.running_to_scale(
                            self.running_hw_reverse, self.parameter_bits, mode=self.quant_mode))
                        q_weight_iw_reverse, scale_iw_reverse, _ = self.quant.quant(
                            clamp_weight_iw_reverse, self.parameter_bits, scale=scale_iw_reverse(), mode=self.quant_mode, quant_data='weight')
                        q_weight_hw_reverse, scale_hw_reverse, _ = self.quant.quant(
                            clamp_weight_hw_reverse, self.parameter_bits, scale=scale_hw_reverse(), mode=self.quant_mode, quant_data='weight')
                    self.scale_iw_reverse.data.fill_(scale_iw_reverse())
                    self.scale_hw_reverse.data.fill_(scale_hw_reverse())
                if self.bias:
                    clamp_bias_iw_reverse = normalize_bias_with_config(
                        self.bias_ih_l0_reverse, self.clamp_bias, False)
                    clamp_bias_hw_reverse = normalize_bias_with_config(
                        self.bias_hh_l0_reverse, self.clamp_bias, False)
                    self.bias_ih_l0_reverse.data = clamp_bias_iw_reverse
                    self.bias_hh_l0_reverse.data = clamp_bias_hw_reverse
                    if is_in_onnx_export():
                        if config.PlatFormQuant.platform_quant == PlatFormQuant.luna_quant:
                            q_bias_iw_reverse = (
                                clamp_bias_iw_reverse * scale_i_reverse * scale_iw_reverse + 0.5).floor()
                            q_bias_hw_reverse = (
                                clamp_bias_hw_reverse * scale_h_reverse * scale_hw_reverse + 0.5).floor()
                            if self.data_bits + self.parameter_bits <= 16:
                                q_bias_iw_reverse = q_bias_iw_reverse.float().int()
                                q_bias_hw_reverse = q_bias_hw_reverse.float().int()
                        else:
                            assert False, "linger only support luna quant."
            if is_in_onnx_export():
                if self.parameter_bits <= 8:
                    self.weight_ih_l0.data = q_weight_iw.char()
                    self.weight_hh_l0.data = q_weight_hw.char()
                    if self.bidirectional:
                        self.weight_ih_l0_reverse.data = q_weight_iw_reverse.char()
                        self.weight_hh_l0_reverse.data = q_weight_hw_reverse.char()
                elif self.parameter_bits <= 16:
                    self.weight_ih_l0.data = q_weight_iw.short()
                    self.weight_hh_l0.data = q_weight_hw.short()
                    if self.bidirectional:
                        self.weight_ih_l0_reverse.data = q_weight_iw_reverse.short()
                        self.weight_hh_l0_reverse.data = q_weight_hw_reverse.short()
                else:
                    self.weight_ih_l0.data = q_weight_iw.int()
                    self.weight_hh_l0.data = q_weight_hw.int()
                    if self.bidirectional:
                        self.weight_ih_l0_reverse.data = q_weight_iw_reverse.int()
                        self.weight_hh_l0_reverse.data = q_weight_hw_reverse.int()
                if self.bias:
                    self.bias_ih_l0.data = q_bias_iw.int()
                    self.bias_hh_l0.data = q_bias_hw.int()
                    if self.bidirectional:
                        self.bias_ih_l0_reverse.data = q_bias_iw_reverse.int()
                        self.bias_hh_l0_reverse.data = q_bias_hw_reverse.int()
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, self in self._modules.items():
            if self is not None:
                self.state_dict(destination, prefix + name +
                                '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination
