import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Union, Dict, Any
from torch.nn.utils.rnn import PackedSequence

from ..qtensor import QSigmoidFunction
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor
from ...quantizer import WQuantizer, AQuantizer, BQuantizer
from ....config import QUANT_CONFIGS
from ....utils import _unbind, _unbind_packed, _slice, hx_slice, QatMethod, PlatForm

import lingerext

def luna_requant(x_int, scale_x, scale_y):
    l_scale = scale_y - scale_x
    
    if l_scale > 0:
        x_int = x_int * pow(2, l_scale)
    else:
        x_int = (x_int * pow(2, l_scale) + 0.5).floor().int()
    return x_int

class QLSTMSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        # 转换为Q27格式的int32
        i_q27 = (input * (1 << 27) + 0.5).floor().to(torch.int32)   # float到int32需要2次舍入，这是第二次
        i_q27.clamp_(-2**31, 2**31-1)

        output_q31 = None
        if QUANT_CONFIGS.platform == PlatForm.venus:
            output_q31 = lingerext.venusa_qsigmoid_forward(i_q27.contiguous())
        elif QUANT_CONFIGS.platform == PlatForm.arcs or QUANT_CONFIGS.platform == PlatForm.mars:
            output_q31 = lingerext.venusa_qsigmoid_forward(i_q27.contiguous())
        elif QUANT_CONFIGS.platform == PlatForm.venusA:
            output_q31 = lingerext.venusa_qsigmoid_forward(i_q27.contiguous())

        output_q15 = luna_requant(output_q31.int(), 31, 15)
        output = output_q15.float() / (1 << 15)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # 使用标准sigmoid的梯度近似
        input = input.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            y = F.sigmoid(input)
            gradInput = torch.autograd.grad(y, input, grad_output)

        return gradInput

class QLSTMTanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        # 转换为Q27格式的int32
        i_q27 = (input * (1 << 27) + 0.5).floor().to(torch.int32)   # float到int32需要2次舍入，这是第二次
        i_q27.clamp_(-2**31, 2**31-1)

        output_q31 = None
        if QUANT_CONFIGS.platform == PlatForm.venus:
            output_q31 = lingerext.venusa_qtanh_forward(i_q27.contiguous())
        elif QUANT_CONFIGS.platform == PlatForm.arcs or QUANT_CONFIGS.platform == PlatForm.mars:
            output_q31 = lingerext.arcs_qtanh_forward(i_q27.contiguous())
        elif QUANT_CONFIGS.platform == PlatForm.venusA:
            output_q31 = lingerext.venusa_qtanh_forward(i_q27.contiguous())

        output_q15 = luna_requant(output_q31.int(), 31, 15)
        output = output_q15.float() / (1 << 15)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # 使用标准tanh的梯度近似
        input = input.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            y = F.tanh(input)
            gradInput = torch.autograd.grad(y, input, grad_output)

        return gradInput


class QLSTMCell(nn.Module):
    def forward(self, input_x, hidden, cx, weight_ih, weight_hh, bias_ih, bias_hh, training,
                    input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer, 
                    biasih_quantizer, biashh_quantizer, cell_quantizer):
        # step1 input.mul, hidden.mul, fake_quant to keep same value
        gi_output = F.linear(input_x, weight_ih, bias_ih)
        gh_output = F.linear(hidden, weight_hh, bias_hh)

        scale_gi = input_quantizer.scale * weightih_quantizer.scale
        scale_gh = hidden_quantizer.scale * weighthh_quantizer.scale
        gi_output = biasih_quantizer(gi_output, scale_gi) # fake_quant gi_output
        gh_output = biashh_quantizer(gh_output, scale_gh)

        gates = gi_output + gh_output   # 这一步推理时没有舍入
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate      = QLSTMSigmoidFunction.apply(ingate)
        forgetgate  = QLSTMSigmoidFunction.apply(forgetgate)
        cellgate    = QLSTMTanhFunction.apply(cellgate)
        outgate     = QLSTMSigmoidFunction.apply(outgate)
        # sigmoid和tanh结果均为Q15伪量化之后结果，已经包含舍入操作

        # fake_quant cx
        scale_cell = torch.tensor(2**15, dtype=torch.float32)
        new_cx = cell_quantizer(cx, scale_cell)
        cy1 = new_cx * forgetgate
        cy2 = ingate * cellgate
        cy = cy1 + cy2
        hy = QLSTMTanhFunction.apply(cy)    #包含一次舍入
        hy = hy * outgate
        cy = cell_quantizer(cy, scale_cell)

        return hy, cy
        
@register_qmodule(torch.nn.LSTM)
class QLSTM(QModuleMixin, nn.LSTM):
    @classmethod
    def qcreate(
        cls,
        module,
        activations_cfg: Optional[Dict[str, Any]] = None,
        weights_cfg: Optional[Dict[str, Any]] = None,
        bias_cfg: Optional[Dict[str, Any]] = None,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        lstm_module = cls(
            module.input_size,
            module.hidden_size,
            module.num_layers,
            module.bias,
            module.batch_first,
            module.dropout,
            module.bidirectional,
            module.proj_size,
            
            dtype=module.weight_ih_l0.dtype,
            device=device,
            activations_cfg=activations_cfg,
            weights_cfg=None, # 仅打开输入输出的量化
            bias_cfg=None,
            constrain=constrain,
            open_ihook = False,
            open_ohook = False
        )

        lstm_module.add_module("input_quantizer", AQuantizer(activations_cfg, None))
        lstm_module.add_module("weightih_quantizer", WQuantizer(weights_cfg, constrain))
        lstm_module.add_module("weighthh_quantizer", WQuantizer(weights_cfg, constrain))
        lstm_module.add_module("hidden_quantizer", AQuantizer(activations_cfg, None))
        lstm_module.add_module("cell_quantizer", BQuantizer(bias_cfg, None))   # cell量化器，用于保持推理一致性
        lstm_module.add_module("output_quantizer", AQuantizer(activations_cfg, constrain))
        if module.bias:
            lstm_module.add_module("biasih_quantizer", BQuantizer(bias_cfg, constrain))
            lstm_module.add_module("biashh_quantizer", BQuantizer(bias_cfg, constrain))
        else:
            lstm_module.add_module("biasih_quantizer", None)
            lstm_module.add_module("biashh_quantizer", None)

        if module.bidirectional:
            lstm_module.add_module("weightih_reverse_quantizer", WQuantizer(weights_cfg, constrain))
            lstm_module.add_module("weighthh_reverse_quantizer", WQuantizer(weights_cfg, constrain))
            lstm_module.add_module("hidden_reverse_quantizer", AQuantizer(activations_cfg, None))
            lstm_module.add_module("output_reverse_quantizer", AQuantizer(activations_cfg, constrain))
            if module.bias:
                lstm_module.add_module("biasih_reverse_quantizer", BQuantizer(bias_cfg, constrain))
                lstm_module.add_module("biashh_reverse_quantizer", BQuantizer(bias_cfg, constrain))
            else:
                lstm_module.add_module("biasih_reverse_quantizer", None)
                lstm_module.add_module("biashh_reverse_quantizer", None)

        lstm_module.weight_ih_l0 = module.weight_ih_l0
        lstm_module.weight_hh_l0 = module.weight_hh_l0
        lstm_module.bias_ih_l0 = module.bias_ih_l0
        lstm_module.bias_hh_l0 = module.bias_hh_l0

        if module.bidirectional:
            lstm_module.weight_ih_l0_reverse = module.weight_ih_l0_reverse
            lstm_module.weight_hh_l0_reverse = module.weight_hh_l0_reverse
            lstm_module.bias_ih_l0_reverse = module.bias_ih_l0_reverse
            lstm_module.bias_hh_l0_reverse = module.bias_hh_l0_reverse

        lstm_module.qlstm_cell_func = QLSTMCell()

        return lstm_module
    
    @property
    def qweight_ih_hh(self):
        fake_w_ih = self.weightih_quantizer(self.weight_ih_l0)
        fake_w_hh = self.weighthh_quantizer(self.weight_hh_l0)
        return fake_w_ih, fake_w_hh
    
    @property
    def qweight_ih_hh_reverse(self):
        if self.bidirectional:
            fake_w_ih_r = self.weightih_reverse_quantizer(self.weight_ih_l0_reverse)
            fake_w_hh_r = self.weighthh_reverse_quantizer(self.weight_hh_l0_reverse)
            return fake_w_ih_r, fake_w_hh_r
        return None, None

    @property
    def qbias_ih_hh(self):
        if self.biasih_quantizer is None:
            return self.bias_ih_l0, self.bias_hh_l0
        fake_bias_ih = self.biasih_quantizer(self.bias_ih_l0, self.weightih_quantizer.scale * self.input_quantizer.scale)
        fake_bias_hh = self.biashh_quantizer(self.bias_hh_l0, self.weighthh_quantizer.scale * self.hidden_quantizer.scale)
        return fake_bias_ih, fake_bias_hh
    
    @property
    def qbias_ih_hh_reverse(self):
        if self.bidirectional:
            if self.biasih_quantizer is None:
                return self.bias_ih_l0_reverse, self.bias_hh_l0_reverse
            fake_bias_ih_r = self.biasih_reverse_quantizer(self.bias_ih_l0_reverse, self.weightih_reverse_quantizer.scale * self.input_quantizer.scale)
            fake_bias_hh_r = self.biashh_reverse_quantizer(self.bias_hh_l0_reverse, self.weighthh_reverse_quantizer.scale * self.hidden_reverse_quantizer.scale)
            return fake_bias_ih_r, fake_bias_hh_r
        return None, None

    def quantize_lstm_input(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, QTensor):
            fake_input = from_qtensor_to_tensor(input)
            self.input_quantizer.scale.fill_(input.scale.detach())
            self.input_quantizer.data_bits = input.data_bits
        else:
            fake_input = self.input_quantizer(input) # 前向过程中会更新input_quantizer的scale
        return fake_input

    def quantize_lstm_hidden(self, hidden: torch.Tensor, direct, scale=None) -> torch.Tensor:
        if direct:
            fake_hidden = self.hidden_reverse_quantizer(hidden, scale)
        else:
            fake_hidden = self.hidden_quantizer(hidden, scale)
        return fake_hidden
    
    def quantize_lstm_out(self, input: torch.Tensor, direct) -> torch.Tensor:
        if direct:
            fake_out = self.output_reverse_quantizer(input)
        else:
            fake_out = self.output_quantizer(input)
        return fake_out
        # return from_tensor_to_qtensor(fake_out, self.output_quantizer.scale, self.output_quantizer.data_bits)
    
    def qforward(self, input, *args, **kwargs):
        hx = None if len(args) == 0 else args[0]
        if QUANT_CONFIGS.calibration:
            return self.forward_calibrate(input, hx)
        else:
            return self.forward_train(input, hx)


    def forward_calibrate(self, input, hx=None):
        with torch.no_grad():
            if self.num_layers != 1:
                assert False, "Intx-NormalizeLSTM don't support num_layer!=1 !"
            orig_input = input
            if isinstance(orig_input, PackedSequence):
                input, batch_sizes, sorted_indices, unsorted_indices = orig_input
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

            input = self.quantize_lstm_input(input)

            self.check_forward_args(input, hx, batch_sizes)

            w_ih, w_hh = self.qweight_ih_hh
            b_ih, b_hh = self.qbias_ih_hh
            output, hidden = torch.ops.aten.lstm(
                                input, hx, [w_ih, w_hh, b_ih, b_hh],
                                has_biases=self.bias,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                train=self.training,
                                bidirectional=False,
                                batch_first=self.batch_first
                            )
            hidden = hidden[0]
            cell = hidden[1]
            # hidden = self.quantize_lstm_hidden(hidden, 0)
            output = self.quantize_lstm_out(output, 0)

            if self.bidirectional:
                input_r = input.flip(0)
                w_ih_r, w_hh_r = self.qweight_ih_hh_reverse
                b_ih_r, b_hh_r = self.qbias_ih_hh_reverse
                output_r, hidden_r = torch.ops.aten.lstm(
                                input_r, hx, [w_ih_r, w_hh_r, b_ih_r, b_hh_r],
                                has_biases=self.bias,
                                num_layers=self.num_layers,
                                dropout=self.dropout,
                                train=self.training,
                                bidirectional=False,
                                batch_first=self.batch_first
                            )
                hidden_r = hidden_r[0]
                cell_r = hidden_r[1]
                # hidden_r = self.quantize_lstm_hidden(hidden_r, 1)
                output_r = self.quantize_lstm_out(output_r, 1)
                output = torch.cat((output, output_r), -1)
                hidden = torch.cat((hidden, hidden_r), -1)
                cell = torch.cat((cell, cell_r), -1)
                
            hidden = (hidden, cell)
            output = from_tensor_to_qtensor(output, self.output_quantizer.scale, self.output_quantizer.data_bits)

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

    def forward_train(self, input, hx=None):
        orig_input = input
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = orig_input
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
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        assert self.num_layers == 1, 'invalid num_layers, now only support num_layers = 1'

        self.quantize_lstm_input(input)

        # init hidden
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions, max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
            self.quantize_lstm_hidden(hx[0][0], 0, self.input_quantizer.scale)  # init hidden_quantizer
            if self.bidirectional:
                self.quantize_lstm_hidden(hx[0][1], 1, self.input_quantizer.scale)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)
            self.quantize_lstm_hidden(hx[0][0], 0, None)
            if self.bidirectional:
                self.quantize_lstm_hidden(hx[0][1], 1, None)

        self.check_forward_args(input, hx, batch_sizes)

        # forward
        if batch_sizes is None:
            output, hy, cy = self.forward_input_tensor(input, hx)
        else:
            output, hy, cy = self.forward_input_packed(input, hx, batch_sizes)
        hidden = (hy, cy)
        
        output = from_tensor_to_qtensor(output, self.output_quantizer.scale, self.output_quantizer.data_bits)

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

    def forward_input_packed(self, input, hx, batch_sizes=None):
        hiddens = self._generate_hiddens(hx)
        output, hr, ct = self.lstm_forward(input, hiddens, batch_sizes)
        return output, hr, ct

    def forward_input_tensor(self, input, hx):
        # Convert input to (seq_len, batch_size, input_size)
        input = input.transpose(0, 1) if self.batch_first else input
        hiddens = self._generate_hiddens(hx)
        output, hr, ct = self.lstm_forward(input, hiddens)
        output = output.transpose(0, 1) if self.batch_first else output
        return output, hr, ct

    def lstm_forward(self, input, hiddens, batch_sizes=None):
        final_hiddens = []
        # Go through layers
        for layer_num in range(self.num_layers):
            hid = hiddens[layer_num] if hiddens is not None else None
            output, hc = self._bidirection(input, layer_num, hid, batch_sizes) if self.bidirectional else self._single_direction(input, layer_num, hid, batch_sizes)
            final_hiddens.extend(hc)
            ## add dropout
            if (self.dropout!= 0 and self.training and layer_num < self.num_layers - 1):
                 output = torch.nn.functional.dropout(output, self.dropout)

        hy = [hidden[0] for hidden in final_hiddens]
        cy = [hidden[1] for hidden in final_hiddens]
        hy = torch.stack(hy, 0)
        cy = torch.stack(cy, 0)

        return output, hy, cy

    def _single_direction(self, input, layer, hx, batch_sizes = None):
        hidden = hx[0]
        cell_state = hx[1]
        output, hidden = self._run_single_direction(input, hidden, cell_state, layer, direct=0, batch_sizes=batch_sizes)
        return output, [hidden]

    def _bidirection(self, input, layer, hx, batch_sizes = None):
        hx_f = hx[0][0]
        ct_f = hx[0][1]
        hx_b = hx[1][0]
        ct_b = hx[1][1]
        fw_output, fw_hidden = self._run_single_direction(input, hx_f, ct_f, layer, direct=0, batch_sizes=batch_sizes)
        rev_output, rev_hidden = self._run_single_direction(input, hx_b, ct_b, layer, direct=1, batch_sizes=batch_sizes)
        if batch_sizes is None:
            output = torch.cat((fw_output, rev_output), fw_output.dim()-1)
        else:  #packed sequence
            output = torch.cat((fw_output, rev_output), -1)
        return output, [fw_hidden, rev_hidden]
    
    def _run_single_direction(self, input, hidden, cell_state, layer=0, direct=0, batch_sizes=None):
        # bidirection quantizer
        input_quantizer = self.input_quantizer
        cell_quantizer = self.cell_quantizer
        hidden_quantizer    = self.hidden_quantizer if direct == 0 else self.hidden_reverse_quantizer
        weightih_quantizer  = self.weightih_quantizer if direct == 0 else self.weightih_reverse_quantizer
        weighthh_quantizer  = self.weighthh_quantizer if direct == 0 else self.weighthh_reverse_quantizer
        biasih_quantizer  = self.biasih_quantizer if direct == 0 else self.biasih_reverse_quantizer
        biashh_quantizer  = self.biashh_quantizer if direct == 0 else self.biashh_reverse_quantizer
        output_quantizer    = self.output_quantizer if direct == 0 else self.output_reverse_quantizer

        # fake_quant hidden, weight, bias
        weight_ih, weight_hh = self.qweight_ih_hh if direct == 0 else self.qweight_ih_hh_reverse
        bias_ih, bias_hh = self.qbias_ih_hh if direct == 0 else self.qbias_ih_hh_reverse

        step_outputs = []

        if batch_sizes is None:
            # input = input if direct == 0 else torch.cat(input.split(1,0)[::-1]) 
            input = input if direct == 0 else input.flip(0).contiguous()
            for input_x in input:
                hidden, cell_state = self.qlstm_cell_func(input_x, hidden, cell_state, weight_ih, weight_hh, bias_ih, bias_hh, self.training,
                                        input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer, 
                                        biasih_quantizer, biashh_quantizer, cell_quantizer)
                hidden = self.quantize_lstm_hidden(hidden, direct)   #hidden fake_quant

                step_outputs.append(hidden)
        
            step_outputs = step_outputs[::-1] if direct == 1 else step_outputs
            output = torch.stack(step_outputs, 0)
        elif direct == 0:
            final_hiddens = []
            hidden = copy.deepcopy(hidden)
            cell_state = copy.deepcopy(cell_state)
            #split by time
            input, batch_size_list = _unbind_packed(input, batch_sizes)
            last_batch_size = batch_size_list[0]
            for input_i, batch_len in zip(input, batch_size_list):
                inc = batch_len - last_batch_size
                if inc < 0:
                    #按batch的帧长排完序，由长到短，较短的帧hidden计算的次数少，直接取低位保留
                    final_hiddens.append(_slice((hidden, cell_state) ,batch_len, last_batch_size))
                    hidden, cell_state = hx_slice(None, (hidden, cell_state), last_batch_size, batch_len)
                hidden, cell_state = self.qlstm_cell_func(input_i, hidden, cell_state, weight_ih, weight_hh, bias_ih, bias_hh, self.training,
                                                        input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer, 
                                                        biasih_quantizer, biashh_quantizer, cell_quantizer)
                hidden = self.quantize_lstm_hidden(hidden, direct)   #hidden fake_quant
                step_outputs.append(hidden)
                last_batch_size = batch_len
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
        else:
            input, batch_size_list = _unbind_packed(input, batch_sizes)
            input = input[::-1]   #按照时间t 进行反转
            # input =  torch.cat(input.split(1,0)[::-1])  if direct == 1 else input 
            batch_size_list = batch_size_list[::-1] 
            input_hx = (copy.deepcopy(hidden), copy.deepcopy(cell_state)) 
            last_batch_size = batch_size_list[0]
            
            hidden = _slice(hidden, 0, last_batch_size)
            cell_state = _slice(cell_state, 0, last_batch_size)
            for input_i,batch_len in zip(input, batch_size_list):
                if last_batch_size != batch_len:
                    #获取input_hx高位hidden部分与上一帧的hidden进行填充，相当于补0
                    hidden, cell_state = hx_slice(input_hx, (hidden, cell_state), last_batch_size, batch_len)           
                hidden, cell_state = self.qlstm_cell_func(input_i, hidden, cell_state, weight_ih, weight_hh, bias_ih, bias_hh, self.training,
                                                        input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer, 
                                                        biasih_quantizer, biashh_quantizer, cell_quantizer)
                hidden = self.quantize_lstm_hidden(hidden, direct)   #hidden fake_quant
                step_outputs.append(hidden)
                last_batch_size = batch_len
            
            step_outputs = step_outputs[::-1] 
            output = torch.cat(step_outputs, 0)

        output = self.quantize_lstm_out(output, direct)
        return output, (hidden, cell_state)
    

    def _generate_hiddens(self, hx):
        if hx is not None:
            assert len(hx) == 2, 'hidden(tuple) input length must be 2'
            hidden_list = _unbind(hx[0])
            cellstate_list = _unbind(hx[1])
            assert len(hidden_list) == len(cellstate_list) 
            length = len(hidden_list)
            if self.bidirectional:
                assert length/self.num_layers%2==0, 'hidden len must be double in bidirectional mode'

            i = 0
            hiddens = []
            while i < length:
                if self.bidirectional:
                    hiddens.append(((hidden_list[i], cellstate_list[i]), (hidden_list[i+1], cellstate_list[i+1])))
                    i += 2
                else:
                    hiddens.append((hidden_list[i], cellstate_list[i]))
                    i+= 1
        else:
            hiddens = None
        return hiddens

