from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.onnx import is_in_onnx_export

from ..quant import normalize_data_with_config, normalize_weight_with_config
from ..utils import Dump, QuantMode, ScalerBuffer
from .iqtensor import IQTensor, from_torch_tensor
from .ops import ModuleIntConfig
from .requant import Requant
from ..quant import Quant


class EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse,
                weight, data_bits, parameter_bits, training, momentum, running_x, running_w, running_o, eval_scale_x, eval_scale_w, eval_scale_o,
                prefix, dump, path, mode, o_bits, quant,  is_not_from_iqtensor, clamp_data, clamp_weight, clamp_bias):
        scale_o = None
        if training:
            ctx.clamp_data = clamp_data
            ctx.clamp_weight = clamp_weight
            ctx.clamp_bias = clamp_bias
            ctx.bits = o_bits, parameter_bits
            saved_tensors = [input, weight]
            ctx.params = [num_embeddings, embedding_dim, padding_idx,
                          max_norm, norm_type, scale_grad_by_freq, sparse, ]

            # weights = normalize_weight_with_config(weight, clamp_weight, False)
            weights = weight
            q_weights, scale_w, max_value_w = quant.quant(
                weights, parameter_bits, mode=mode, quant_data='weight')
            running_w.mul_(1-momentum).add_(momentum*max_value_w)
            q_weights = q_weights.float() if data_bits + \
                parameter_bits <= 16 else q_weights.double()
            q_outputs = F.embedding(
                input, q_weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

            ctx.save_for_backward(*saved_tensors)

            outputs = quant.dequant(q_outputs, scale_w)
            if o_bits is not None:
                running_o.fill_(running_w())
                scale_o = scale_w
            ctx.scale_w = scale_w

        else:
            scale_x = None
            scale_w = None
            scale_o = eval_scale_o
            q_weights = None
            if weight.dtype == torch.float32:
                scale_x = ScalerBuffer(quant.running_to_scale(
                    running_x, data_bits, mode=mode))
                weights = weight
                # weights = normalize_weight_with_config(
                #     weight, clamp_weight, False)
                q_weights, scale_w, _ = quant.quant(
                    weights, parameter_bits, mode=mode, quant_data='weight')
                scale_w = ScalerBuffer(scale_w)
                if o_bits is not None:
                    assert running_o > 0, 'invalid running_o <= 0, please finetune training'
                    scale_o = scale_w
            else:
                scale_x = eval_scale_x
                scale_w = eval_scale_w
                q_weights = weight.double()
                if o_bits is not None:
                    scale_o = eval_scale_w
            q_weights = q_weights.double()
            q_outputs = F.embedding(
                input, q_weights, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

            outputs = quant.dequant(q_outputs, scale_w)
            if dump:
                name_list = ["input", "outputs",  "q_outputs", "q_weights",
                             "scale_x", "scale_w" "scale_o", "running_x", "running_o"]
                attr_list = [input, outputs, q_outputs, q_weights, scale_x.data,
                             scale_w.data, scale_o.data, running_x.data,  running_o.data]
                Dump.dump_file(prefix, ".EmbeddingInt.",
                               zip(name_list, attr_list), path)

        if o_bits is None:
            return outputs
        elif isinstance(scale_o, float):
            return from_torch_tensor(outputs, scale_o, o_bits)
        elif isinstance(scale_o, torch.Tensor):
            return from_torch_tensor(outputs, scale_o.item(), o_bits)
        else:
            return from_torch_tensor(outputs, scale_o.data, o_bits)

    @staticmethod
    def backward(ctx, gradOutput):
        clamp_data = ctx.clamp_data
        o_bits, parameter_bits = ctx.bits
        scale_w = ctx.scale_w
        input, weight,  = ctx.saved_tensors
        num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, = ctx.params
        q_weights, _, _ = Quant.quant(
            weight.data, o_bits, scale_w, mode=QuantMode.QValue, quant_data='weight')
        f_weights = Quant.dequant(q_weights, scale_w)
        f_weights = f_weights.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            # weights = normalize_weight_with_config(weight, clamp_weight, True)
            # weights = weight
            z = F.embedding(input, f_weights, padding_idx, max_norm,
                            norm_type, scale_grad_by_freq, sparse)
            if o_bits is not None:
                z = normalize_data_with_config(
                    z, clamp_data)
            gradWeight, = torch.autograd.grad(z, (f_weights), gradOutput)
        return None, None, None, None, None, None, None, None, gradWeight,\
            None, None, None, None, None, None, None, None, None, None,\
            None, None, None, None, None, None,  None, None, None, None, None

    @staticmethod
    def symbolic(g, input, num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse,
                 weight, data_bits, parameter_bits, training, momentum, running_x, running_w, running_o, eval_scale_x, eval_scale_w, eval_scale_o,
                 prefix, dump, path, mode, o_bits, quant,  is_not_from_iqtensor, clamp_data, clamp_weight, clamp_bias):

        return g.op("thinker::Gather", weight, input, scale_w_f=eval_scale_w(), scale_o_f=eval_scale_o(), parameter_bits_i=parameter_bits, o_bits_i=o_bits)


class EmbeddingInt(nn.Embedding, ModuleIntConfig):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None,
                 data_bits=8, parameter_bits=8, mode=QuantMode.QValue, o_bits=None,
                 clamp_data=None, clamp_weight=None, clamp_bias=None):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, padding_idx,
                              max_norm, norm_type, scale_grad_by_freq, sparse, _weight)
        ModuleIntConfig.__init__(
            self, data_bits=data_bits, parameter_bits=parameter_bits, mode=mode, o_bits=o_bits)
        self.momentum = 0.1
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.is_not_from_iqtensor = True
        self.clamp_data = clamp_data
        self.clamp_weight = clamp_weight
        self.clamp_bias = clamp_bias
        self.mode = mode
        self.register_buffer('running_x', torch.zeros(1))
        self.register_buffer('running_w', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))
        self.register_buffer('scale_x', torch.zeros(1))
        self.register_buffer('scale_w', torch.zeros(1))
        self.register_buffer('scale_o', torch.zeros(1))

    def forward(self, input):
        scale_x = ScalerBuffer(self.scale_x)
        running_x = ScalerBuffer(self.running_x)
        if isinstance(input, IQTensor):
            # assert False, "embeddingint   input shouldn't be IQTensor !"
            self.is_not_from_iqtensor = False
            if input.bits != self.data_bits:
                input = Requant.apply(
                    input, input.bits, input.scale_data, self.data_bits, self.mode)
            scale_x = ScalerBuffer(input.scale_data)
            running_x = ScalerBuffer(input.running_data)
        running_w = ScalerBuffer(self.running_w)
        running_o = ScalerBuffer(self.running_o)
        scale_w = ScalerBuffer(self.scale_w)
        scale_o = ScalerBuffer(self.scale_o)

        weights = self.weight
        if self.weight.dtype == torch.float32:
            weights = normalize_weight_with_config(
                self.weight, self.clamp_weight, self.training)

        ret = EmbeddingFunction.apply(input, self.num_embeddings, self.embedding_dim, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse,
                                      weights, self.data_bits, self.parameter_bits, self.training, self.momentum,
                                      running_x, running_w, running_o, scale_x, scale_w, scale_o,
                                      self.prefix, self.dump, self.path, self.quant_mode, self.o_bits, self.quant, self.is_not_from_iqtensor,
                                      self.clamp_data, self.clamp_weight, self.clamp_bias)
        self.running_x.fill_(running_x())
        self.running_w.fill_(running_w())
        self.running_o.fill_(running_o())
        self.scale_x.fill_(scale_x())
        self.scale_w.fill_(scale_w())
        self.scale_o.fill_(scale_o())
        return ret

    def state_dict(module, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]
                              ] = local_metadata = dict(version=module._version)
        if is_in_onnx_export():
            assert module._buffers['running_w'] > 0, 'invalid running_w, please finetune first'

        if 'scale_w' in module._buffers and module._parameters['weight'].dtype == torch.float:
            weights = module._parameters['weight'].data
            q_weights, scale_mul_w, _ = module.quant.quant(
                weights, module.parameter_bits, mode=module.quant_mode)
            if is_in_onnx_export():
                scale_w = ScalerBuffer(scale_mul_w)
                module._buffers['scale_w'].data.fill_(scale_w())
                scale_o = scale_w
                module._buffers['scale_o'].data.fill_(scale_o())

            weight_tensor = module._parameters['weight']
            if is_in_onnx_export():

                if module.parameter_bits <= 8:
                    weight_tensor.data = q_weights.char()
                    weight_tensor.char()

                elif module.parameter_bits <= 16:
                    weight_tensor.data = q_weights.short()
                    weight_tensor.short()

                else:
                    weight_tensor.data = q_weights.int()
                    weight_tensor.int()

        module._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in module._modules.items():
            if module is not None:
                module.state_dict(destination, prefix +
                                  name + '.', keep_vars=keep_vars)
        for hook in module._state_dict_hooks.values():
            hook_result = hook(module, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def extra_repr(self):
        s = nn.Embedding.extra_repr(self)
        extra_s = ',clamp_data:{clamp_data},clamp_weight:{clamp_weight},clamp_bias:{clamp_bias}'.format(
            **self.__dict__)
        extra_s += ',data_bits:{data_bits},parameter_bits:{parameter_bits},o_bits:{o_bits}'.format(
            **self.__dict__)
        return s+extra_s
