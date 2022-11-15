#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
from torch import _VF
from torch.nn.utils.rnn import PackedSequence
from torch.onnx import is_in_onnx_export

from ..quant import NormalizeFunction


class LSTMOnnxFakeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lengths, hidden_state, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse,
                input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, bias_B, bias_B_reverse,
                hidden_state_forward, hidden_state_reverse, cell_state_forward, cell_state_reverse,
                weight_ih_bi, weight_hh_bi, bias_B_bi):
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
            None, None, None, None, None, None, None,\
            None, None, None, None, None, None, None

    @staticmethod
    def symbolic(g, input, lengths, hidden_state, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                 weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse,
                 input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, bias_B, bias_B_reverse,
                 hidden_state_forward, hidden_state_reverse, cell_state_forward, cell_state_reverse,
                 weight_ih_bi, weight_hh_bi, bias_B_bi):
        if num_layers != 1:
            assert False, "Current intx not support num_layer!=1 onnx export !"
        param_dict = {'hidden_size_i': hidden_size, 'direction_s': "forward"}
        param_bidirectional_dict = {}
        if bidirectional:
            param_bidirectional_dict = {
                'hidden_size_i': hidden_size, 'direction_s': "bidirectional"}

        if batch_first:
            input = g.op("Transpose", *[input], perm_i=(1, 0, 2))

        input_list = [input, weight_ih, weight_hh]
        input_bidirectional_list = [input, weight_ih_bi, weight_hh_bi]
        if bias_ih is not None and bias_hh is not None:
            input_list.append(bias_B)
            input_bidirectional_list.append(bias_B_bi)

        param_dict['outputs'] = 3
        param_bidirectional_dict['outputs'] = 3

        if lengths is None and hidden_state is None:
            if not bidirectional:
                lstm, hidden, cell = g.op("LSTM", *input_list, **param_dict)
                lstm = g.op("Squeeze", *[lstm], axes_i=(1,))
                if batch_first:
                    lstm = g.op("Transpose", *[lstm], perm_i=(1, 0, 2))
            else:
                lstm_bi, hidden, cell = g.op(
                    "LSTM", *input_bidirectional_list, **param_bidirectional_dict)
                if batch_first:
                    lstm_bi = g.op(
                        "Transpose", *[lstm_bi], perm_i=(0, 2, 1, 3))
                args = [lstm_bi]
                lstm_bi = g.op('Reshape', *args, g.op('Constant',
                               value_t=torch.LongTensor([0, 0, -1])))
                lstm = g.op("Transpose", *[lstm_bi], perm_i=(1, 0, 2))

        elif lengths is not None and hidden_state is None:
            input_list.append(lengths)
            input_bidirectional_list.append(lengths)
            if not bidirectional:
                lstm, hidden, cell = g.op("LSTM", *input_list, **param_dict)
                lstm = g.op("Squeeze", *[lstm], axes_i=(1,))
                if batch_first:
                    lstm = g.op("Transpose", *[lstm], perm_i=(1, 0, 2))
            else:
                lstm_bi, hidden, cell = g.op(
                    "LSTM", *input_bidirectional_list, **param_bidirectional_dict)
                if batch_first:
                    lstm_bi = g.op(
                        "Transpose", *[lstm_bi], perm_i=(0, 2, 1, 3))
                args = [lstm_bi]
                lstm_bi = g.op('Reshape', *args, g.op('Constant',
                               value_t=torch.LongTensor([0, 0, -1])))
                lstm = g.op("Transpose", *[lstm_bi], perm_i=(1, 0, 2))

        else:
            input_list.append(lengths)
            input_list.append(hidden_state)
            input_list.append(cell_state)
            input_bidirectional_list.append(lengths)
            input_bidirectional_list.append(hidden_state)
            input_bidirectional_list.append(cell_state)
            if not bidirectional:
                lstm, hidden, cell = g.op("LSTM", *input_list, **param_dict)
                lstm = g.op("Squeeze", *[lstm], axes_i=(1,))
                if batch_first:
                    lstm = g.op("Transpose", *[lstm], perm_i=(1, 0, 2))
            else:
                lstm_bi, hidden, cell = g.op(
                    "LSTM", *input_bidirectional_list, **param_bidirectional_dict)
                if batch_first:
                    lstm_bi = g.op(
                        "Transpose", *[lstm_bi], perm_i=(0, 2, 1, 3))
                args = [lstm_bi]
                lstm_bi = g.op('Reshape', *args, g.op('Constant',
                               value_t=torch.LongTensor([0, 0, -1])))
                lstm = g.op("Transpose", *[lstm_bi], perm_i=(1, 0, 2))

        return lstm, hidden, cell


class NormalizeFastLSTM(nn.LSTM):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False,
                 normalize_data=None, normalize_weight=None, normalize_bias=None):
        nn.LSTM.__init__(self, input_size, hidden_size, num_layers,
                         bias, batch_first, dropout, bidirectional)
        self.normalize_data = normalize_data
        self.normalize_weight = normalize_weight
        self.normalize_bias = normalize_bias

    def forward(self, input, hx=None):
        if not is_in_onnx_export():
            if self.num_layers != 1:
                assert False, "Intx-NormalizeLSTM don't support num_layer!=1 !"
            orig_input = input
            if isinstance(orig_input, PackedSequence):
                input, batch_sizes, sorted_indices, unsorted_indices = input
                max_batch_size = batch_sizes[0]
                max_batch_size = int(max_batch_size)
            elif isinstance(orig_input, tuple):
                input, lengths, batch_first, enforce_sorted = orig_input
                packed_input = torch.nn.utils.rnn.pack_padded_sequence(
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

            if hx is None:
                num_directions = 2 if self.bidirectional else 1
                zeros = torch.zeros(self.num_layers * num_directions,
                                    max_batch_size, self.hidden_size,
                                    dtype=input.dtype, device=input.device)
                hx = (zeros, zeros)
            else:
                hx = self.permute_hidden(hx, sorted_indices)

            flat_weights = []
            direct = 0
            self.check_forward_args(input, hx, batch_sizes)
            weight_ih = self.weight_ih_l0_reverse if direct == 1 else self.weight_ih_l0
            weight_hh = self.weight_hh_l0_reverse if direct == 1 else self.weight_hh_l0
            if self.bias:
                bias_ih = self.bias_ih_l0_reverse if direct == 1 else self.bias_ih_l0
                bias_hh = self.bias_hh_l0_reverse if direct == 1 else self.bias_hh_l0

            if self.normalize_weight is not None:
                weight_ih = NormalizeFunction.apply(
                    weight_ih, self.normalize_weight, self.training)
                weight_hh = NormalizeFunction.apply(
                    weight_hh, self.normalize_weight, self.training)
            if bias_ih is not None and self.normalize_bias is not None:
                bias_ih = NormalizeFunction.apply(
                    bias_ih, self.normalize_bias, self.training)
                bias_hh = NormalizeFunction.apply(
                    bias_hh, self.normalize_bias, self.training)

            flat_weights.extend([weight_ih, weight_hh])
            if self.bias:
                flat_weights.extend([bias_ih, bias_hh])

            if self.bidirectional:
                direct = 1
                self.check_forward_args(input, hx, batch_sizes)
                weight_ih = self.weight_ih_l0_reverse if direct == 1 else self.weight_ih_l0
                weight_hh = self.weight_hh_l0_reverse if direct == 1 else self.weight_hh_l0
                if self.bias:
                    bias_ih = self.bias_ih_l0_reverse if direct == 1 else self.bias_ih_l0
                    bias_hh = self.bias_hh_l0_reverse if direct == 1 else self.bias_hh_l0
                if self.normalize_weight is not None:
                    weight_ih = NormalizeFunction.apply(
                        weight_ih, self.normalize_weight, self.training)
                    weight_hh = NormalizeFunction.apply(
                        weight_hh, self.normalize_weight, self.training)
                if bias_ih is not None and self.normalize_bias is not None:
                    bias_ih = NormalizeFunction.apply(
                        bias_ih, self.normalize_bias, self.training)
                    bias_hh = NormalizeFunction.apply(
                        bias_hh, self.normalize_bias, self.training)
                flat_weights.extend([weight_ih, weight_hh])
                if self.bias:
                    flat_weights.extend([bias_ih, bias_hh])

            if batch_sizes is None:
                result = _VF.lstm(input, hx,  flat_weights, self.bias, self.num_layers,
                                  self.dropout, self.training, self.bidirectional, self.batch_first)
            else:
                result = _VF.lstm(input, batch_sizes, hx,  flat_weights, self.bias,
                                  self.num_layers, self.dropout, self.training, self.bidirectional)
            output = result[0]
            hidden = result[1:]
            if self.normalize_data is not None:
                output = NormalizeFunction.apply(
                    output, self.normalize_data, self.training, True)

            if isinstance(orig_input, PackedSequence):
                output_packed = PackedSequence(
                    output, batch_sizes, sorted_indices, unsorted_indices)
                return output_packed, self.permute_hidden(hidden, unsorted_indices)
            elif isinstance(orig_input, tuple):
                output_packed = PackedSequence(
                    output, batch_sizes, sorted_indices, unsorted_indices)
                output, lengths = torch.nn.utils.rnn.pad_packed_sequence(
                    output_packed, self.batch_first)
                return (output, lengths), self.permute_hidden(hidden, unsorted_indices)
            else:
                return output, self.permute_hidden(hidden, unsorted_indices)
        else:
            orig_input = input
            if isinstance(orig_input, PackedSequence):
                assert False, "LSTM don't support PackedSequence as input for onnx export!"

            if isinstance(orig_input, tuple):
                input, lengths, _, _ = orig_input
            else:
                input = orig_input
                lengths = None

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

            hidden_state = None
            cell_state = None
            if hx is not None:
                hidden_state, cell_state = hx
            output = None
            hy = None
            cy = None

            weight_ih_chunk = weight_ih.chunk(4, 0)
            weight_ih = torch.cat(
                [weight_ih_chunk[0], weight_ih_chunk[3], weight_ih_chunk[1], weight_ih_chunk[2]], dim=0)
            weight_ih = weight_ih.unsqueeze(0)
            weight_hh_chunk = weight_hh.chunk(4, 0)
            weight_hh = torch.cat(
                [weight_hh_chunk[0], weight_hh_chunk[3], weight_hh_chunk[1], weight_hh_chunk[2]], dim=0)
            weight_hh = weight_hh.unsqueeze(0)

            if self.normalize_weight is not None:
                weight_ih = NormalizeFunction.apply(
                    weight_ih, self.normalize_weight, self.training)
                weight_hh = NormalizeFunction.apply(
                    weight_hh, self.normalize_weight, self.training)

            weight_ih_bi = None
            weight_hh_bi = None

            if self.bidirectional:
                weight_ih_chunk_reverse = weight_ih_reverse.chunk(4, 0)
                weight_ih_reverse = torch.cat(
                    [weight_ih_chunk_reverse[0], weight_ih_chunk_reverse[3], weight_ih_chunk_reverse[1], weight_ih_chunk_reverse[2]], dim=0)
                weight_ih_reverse = weight_ih_reverse.unsqueeze(0)
                weight_hh_chunk_reverse = weight_hh_reverse.chunk(4, 0)
                weight_hh_reverse = torch.cat(
                    [weight_hh_chunk_reverse[0], weight_hh_chunk_reverse[3], weight_hh_chunk_reverse[1], weight_hh_chunk_reverse[2]], dim=0)
                weight_hh_reverse = weight_hh_reverse.unsqueeze(0)
                weight_ih_bi = torch.cat([weight_ih, weight_ih_reverse], dim=0)
                weight_hh_bi = torch.cat([weight_hh, weight_hh_reverse], dim=0)

                if self.normalize_weight is not None:
                    weight_ih_bi = NormalizeFunction.apply(
                        weight_ih_bi, self.normalize_weight, self.training)
                    weight_hh_bi = NormalizeFunction.apply(
                        weight_hh_bi, self.normalize_weight, self.training)

            bias_B = None
            bias_B_reverse = None
            bias_B_bi = None
            hidden_state_forward = None
            hidden_state_reverse = None
            cell_state_forward = None
            cell_state_reverse = None
            if bias_ih is not None and bias_hh is not None:
                bias_ih_chunk = bias_ih.chunk(4, 0)
                bias_ih = torch.cat(
                    [bias_ih_chunk[0], bias_ih_chunk[3], bias_ih_chunk[1], bias_ih_chunk[2]], dim=0)
                bias_hh_chunk = bias_hh.chunk(4, 0)
                bias_hh = torch.cat(
                    [bias_hh_chunk[0], bias_hh_chunk[3], bias_hh_chunk[1], bias_hh_chunk[2]], dim=0)
                bias_B = torch.cat((bias_ih, bias_hh), dim=0)
                bias_B = bias_B.unsqueeze(0)
                if self.normalize_bias is not None:
                    bias_B = NormalizeFunction.apply(
                        bias_B, self.normalize_bias, self.training)

            if self.bidirectional and bias_ih is not None and bias_hh is not None:
                bias_ih_chunk_reverse = bias_ih_reverse.chunk(4, 0)
                bias_ih_reverse = torch.cat(
                    [bias_ih_chunk_reverse[0], bias_ih_chunk_reverse[3], bias_ih_chunk_reverse[1], bias_ih_chunk_reverse[2]], dim=0)
                bias_hh_chunk_reverse = bias_hh_reverse.chunk(4, 0)
                bias_hh_reverse = torch.cat(
                    [bias_hh_chunk_reverse[0], bias_hh_chunk_reverse[3], bias_hh_chunk_reverse[1], bias_hh_chunk_reverse[2]], dim=0)
                bias_B_reverse = torch.cat(
                    (bias_ih_reverse, bias_hh_reverse), dim=0)
                bias_B_reverse = bias_B_reverse.unsqueeze(0)
                bias_B_bi = torch.cat([bias_B, bias_B_reverse], dim=0)
                if self.normalize_bias is not None:
                    bias_B_bi = NormalizeFunction.apply(
                        bias_B_bi, self.normalize_bias, self.training)

            if self.bidirectional and hx is not None:
                hidden_state_chunk = hidden_state.chunk(2, 0)
                cell_state_chunk = cell_state.chunk(2, 0)

                hidden_state_forward = hidden_state_chunk[0]
                hidden_state_reverse = hidden_state_chunk[1]
                cell_state_forward = cell_state_chunk[0]
                cell_state_reverse = cell_state_chunk[1]

            if hx is not None:
                batch_size = input.size(
                    0) if self.batch_first else input.size(1)
                seq_len = input.size(
                    1) if self.batch_first else input.size(0)
                lengths = torch.tensor([seq_len for i in range(
                    batch_size)], dtype=torch.int32, device=input.device) if lengths is None else lengths
            if lengths is not None:
                lengths = lengths.int()
            output, hy, cy = LSTMOnnxFakeFunction.apply(input, lengths, hidden_state, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                                                        weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse,
                                                        self.input_size, self.hidden_size, self.num_layers, self.batch_first, self.dropout, self.bidirectional,
                                                        bias_B, bias_B_reverse,
                                                        hidden_state_forward, hidden_state_reverse, cell_state_forward, cell_state_reverse,
                                                        weight_ih_bi, weight_hh_bi, bias_B_bi)

            if self.normalize_data is not None:
                output = NormalizeFunction.apply(
                    output, self.normalize_data, self.training, False)

            if isinstance(orig_input, tuple):
                return (output, lengths), (hy, cy)
            else:
                return output, (hy, cy)

    def extra_repr(self):
        s = nn.GRU.extra_repr(self)
        extra_s = ',normalize_data:{normalize_data},normalize_weight:{normalize_weight},normalize_bias:{normalize_bias}'.format(
            **self.__dict__)
        return s+extra_s


__all__ = ['NormalizeFastLSTM']
