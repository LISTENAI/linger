import torch
from torch import _VF
from .cmodule import CModuleMixin, register_cmodule
from .cutils import static_clip, dyn_clip_weight
from typing import Optional, Union, Dict, Any
from torch.nn.utils.rnn import PackedSequence

@register_cmodule(torch.nn.LSTM)
class CLSTM(CModuleMixin, torch.nn.LSTM):
    @classmethod
    def ccreate(
        cls,
        module,
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
            constrain=constrain,
            open_ihook = False,
            open_ohook = False
        )

        return lstm_module
    
    
    def forward(self, input, *args, **kwargs):
        hx = None if len(args) == 0 else args[0]
        return self.forward_(input, hx)

    def forward_(self, input, hx=None):
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

        self.check_forward_args(input, hx, batch_sizes)

        flat_weights = []
        bias_ih = bias_hh = None

        weight_ih = self.weight_ih_l0
        weight_hh = self.weight_hh_l0
        if self.bias:
            bias_ih = self.bias_ih_l0
            bias_hh = self.bias_hh_l0

        if self.clamp_weight is not None:
            weight_ih = static_clip(weight_ih, self.clamp_weight)
            weight_hh = static_clip(weight_hh, self.clamp_weight)
        else:
            weight_ih = dyn_clip_weight(weight_ih, self.clamp_factor)
            weight_hh = dyn_clip_weight(weight_hh, self.clamp_factor)

        if self.clamp_bias is not None:
            bias_ih = static_clip(bias_ih, self.clamp_bias)
            bias_hh = static_clip(bias_hh, self.clamp_bias)

        flat_weights.extend([weight_ih, weight_hh])
        if self.bias:
            flat_weights.extend([bias_ih, bias_hh])

        if self.bidirectional:
            weight_ih = self.weight_ih_l0_reverse
            weight_hh = self.weight_hh_l0_reverse
            if self.bias:
                bias_ih = self.bias_ih_l0_reverse
                bias_hh = self.bias_hh_l0_reverse
            if self.clamp_weight is not None:
                weight_ih = static_clip(weight_ih, self.clamp_weight)
                weight_hh = static_clip(weight_hh, self.clamp_weight)
            else:
                weight_ih = dyn_clip_weight(weight_ih, self.clamp_factor)
                weight_hh = dyn_clip_weight(weight_hh, self.clamp_factor)

            if self.clamp_bias is not None:
                bias_ih = static_clip(bias_ih, self.clamp_bias)
                bias_hh = static_clip(bias_hh, self.clamp_bias)

            flat_weights.extend([weight_ih, weight_hh])
            if self.bias:
                flat_weights.extend([bias_ih, bias_hh])

        if batch_sizes is None:
            result = _VF.lstm(input, hx, flat_weights, self.bias, self.num_layers,
                                self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx,  flat_weights, self.bias,
                                self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        if self.clamp_activation is not None:
            output = static_clip(output, self.clamp_activation)

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

