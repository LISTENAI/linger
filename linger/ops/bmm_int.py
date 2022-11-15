#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.onnx import is_in_onnx_export

from ..config import config
from ..quant import Quant, normalize_data_with_config
from ..utils import Dump, QuantMode, ScalerBuffer
from .iqtensor import (IQTensor, from_torch_tensor, platform_to_string,
                       quantlinear, torch_bmm)
from .ops import ModuleIntConfig
from .requant import Requant


class BmmIntFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mat2, data_bits, training, momentum,
                running_x, running_y, running_o, scale_x, scale_y, scale_o,
                prefix, dump, path, mode, o_bits,
                quant, is_not_from_iqtensor1, is_not_from_iqtensor2, clamp_data):

        if training:
            saved_tensors = [input, mat2]
            # ctx.save_for_backward(input, mat2)
            ctx.bits = o_bits, data_bits
            ctx.clamp_data = clamp_data
            zero_point_x = input.zero_point if isinstance(
                input, IQTensor) else 0
            is_iq_tensor_x = True if isinstance(input, IQTensor) else False
            zero_point_y = mat2.zero_point if isinstance(
                input, IQTensor) else 0
            is_iq_tensor_y = True if isinstance(mat2, IQTensor) else False
            ctx.value = zero_point_x, is_iq_tensor_x, zero_point_y, is_iq_tensor_y
            if isinstance(input, IQTensor):
                q_input, _, max_value_x = quant.quant(
                    input.data, data_bits, scale_x, mode=QuantMode.QValue, quant_data='input',
                    iq_zero_point=input.zero_point)
            else:
                q_input, scale_x, max_value_x = quant.quant(
                    input.data, data_bits, mode=mode, quant_data='input')
                running_x.mul_(1-momentum).add_(momentum*max_value_x)
            if isinstance(mat2, IQTensor):
                q_mat2, _, max_value_y = quant.quant(
                    mat2.data, data_bits, scale_y, mode=QuantMode.QValue, quant_data='input', iq_zero_point=mat2.zero_point)
            else:
                q_mat2, scale_y, max_value_y = quant.quant(
                    mat2.data, data_bits, mode=mode, quant_data='input')
                running_y.mul_(1-momentum).add_(momentum*max_value_y)
            q_input = q_input.float() if data_bits <= 8 else q_input.double()
            q_mat2 = q_mat2.float() if data_bits <= 8 else q_mat2.double()
            q_outputs = torch_bmm(q_input, q_mat2)
            outputs = quant.dequant(q_outputs, scale_x * scale_y)
            out_tensor = outputs
            if o_bits is not None:
                outputs = normalize_data_with_config(outputs, clamp_data)
                out_tensor = outputs
                q_outputs, scale_o, max_value_o = quant.quant(
                    outputs, o_bits, mode=mode, quant_data='output')
                outputs = quant.dequant(q_outputs, scale_o)
                running_o.mul_(1-momentum).add_(momentum*max_value_o)
            saved_tensors += [out_tensor]
            ctx.scale = scale_x, scale_y, scale_o
            ctx.save_for_backward(*saved_tensors)

        else:
            assert running_x > 0, 'invalid running_x = 0, please finetune training before eval'
            if not isinstance(input, IQTensor):
                scale_x = ScalerBuffer(quant.running_to_scale(
                    running_x, data_bits, mode=mode))
            if not isinstance(mat2, IQTensor):
                scale_y = ScalerBuffer(quant.running_to_scale(
                    running_y, data_bits, mode=mode))
            if o_bits is not None:
                assert running_o > 0, 'invalid running_o = 0 for BmmInt'
                scale_o = ScalerBuffer(quant.running_to_scale(
                    running_o, o_bits, mode=mode))
            input_zero_point = 0
            mat2_zero_point = 0
            if isinstance(input, IQTensor):
                input_zero_point = input.zero_point
            if isinstance(mat2, IQTensor):
                mat2_zero_point = mat2.zero_point
            q_input, _, _ = quant.quant(
                input.data, data_bits, scale_x, mode=mode, quant_data='input', iq_zero_point=input_zero_point)
            q_mat2, _, _ = quant.quant(
                mat2.data, data_bits, scale_y, mode=mode, quant_data='input', iq_zero_point=mat2_zero_point)
            q_input = q_input.double()
            q_mat2 = q_mat2.double()

            q_outputs = torch_bmm(q_input, q_mat2)
            outputs = quant.dequant(q_outputs, scale_x*scale_y)
            if o_bits is not None:
                q_outputs, _, _ = quant.quant(
                    outputs, o_bits, scale_o, mode=mode, quant_data='output')
                outputs = quant.dequant(q_outputs, scale_o)

            if dump:
                name_list = ['input', 'mat2', 'outputs',
                             'q_input', 'q_mat2', 'q_outputs']
                attr_list = [input, mat2, outputs, q_input, q_mat2, q_outputs]
                Dump.dump_file(prefix, '.BmmInt.', zip(
                    name_list, attr_list), path)

        if o_bits is None:
            return outputs
        elif isinstance(scale_o, float):
            return from_torch_tensor(outputs, scale_o, o_bits)
        elif isinstance(scale_o, torch.Tensor):
            return from_torch_tensor(outputs, scale_o.item(), o_bits)
        else:
            return from_torch_tensor(outputs, scale_o.data, o_bits)

    @staticmethod
    def backward(ctx, gradoutput):
        input, mat2, output = ctx.saved_tensors
        zero_point_x, is_iq_tensor_x, zero_point_y, is_iq_tensor_y = ctx.value
        # input = input.detach().clone().requires_grad_(True)
        # mat2 = mat2.detach().clone().requires_grad_(True)
        scale_x, scale_y, scale_o = ctx.scale
        o_bits, data_bits = ctx.bits
        clamp_data = ctx.clamp_data
        grad_input = grad_mat2 = None

        if is_iq_tensor_x:
            f_input = input.data
        else:
            q_input, _, _ = Quant.quant(
                input.data, data_bits, scale_x, mode=QuantMode.QValue, quant_data='input')
            f_input = Quant.dequant(q_input, scale_x)
        f_input = f_input.detach().clone().requires_grad_(True)
        if is_iq_tensor_y:
            f_mat2 = mat2.data
        else:
            q_mat2, _, _ = Quant.quant(
                mat2.data, data_bits, scale_y, mode=QuantMode.QValue, quant_data='input')
            f_mat2 = Quant.dequant(q_mat2, scale_y)
        f_mat2 = f_mat2.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            z = torch_bmm(f_input, f_mat2)
            if o_bits is not None:
                z = normalize_data_with_config(z, clamp_data)
            grad_input, grad_mat2 = torch.autograd.grad(
                z, (f_input, f_mat2), gradoutput)

        return grad_input, grad_mat2, None, None, None, \
            None, None, None, None,\
            None, None, None, None, None, None,\
            None, None, None, None, None,\
            None

    @staticmethod
    def symbolic(g, input, mat2,
                 data_bits, training, momentum,
                 running_x, running_y, running_o, scale_x, scale_y, scale_o,
                 prefix, dump, path, mode, o_bits,
                 quant, is_not_from_iqtensor1, is_not_from_iqtensor2, clamp_data):
        op_inner1 = None
        op_inner2 = None
        platform_quant = platform_to_string(
            config.PlatFormQuant.platform_quant)
        if is_not_from_iqtensor1:
            op_inner1 = quantlinear(
                g, input, scale_x(), platform_quant, data_bits)
        if is_not_from_iqtensor2:
            op_inner2 = quantlinear(
                g, mat2, scale_y(), platform_quant, data_bits)
        param_dict = {'scale_x_f': scale_x(), 'scale_y_f': scale_y(),
                      'data_bits_i': data_bits}
        input_list = []
        if is_not_from_iqtensor1:
            input_list.append(op_inner1)
        else:
            input_list.append(input)
        if is_not_from_iqtensor2:
            input_list.append(op_inner2)
        else:
            input_list.append(mat2)
        if o_bits is not None:
            param_dict['scale_o_f'] = scale_o()
            param_dict['o_bits_i'] = o_bits
        param_dict['platform_quant_s'] = platform_quant

        return g.op("thinker::BmmInt", *input_list, **param_dict)


class BmmInt(nn.Module, ModuleIntConfig):
    def __init__(self, data_bits=8, mode=QuantMode.QValue, o_bits=None, clamp_data=None):
        nn.Module.__init__(self)
        ModuleIntConfig.__init__(
            self, data_bits=data_bits, mode=mode, o_bits=o_bits)
        self.momentum = 0.1
        self.prefix = ""
        self.dump = False
        self.path = ""
        self.is_not_from_iqtensor1 = True
        self.is_not_from_iqtensor2 = True
        self.clamp_data = clamp_data

        self.register_buffer('running_x', torch.zeros(1))
        self.register_buffer('running_y', torch.zeros(1))
        self.register_buffer('running_o', torch.zeros(1))
        self.register_buffer('scale_x', torch.zeros(1))
        self.register_buffer('scale_y', torch.zeros(1))
        self.register_buffer('scale_o', torch.zeros(1))

    def forward(self, input, mat2):
        running_x = ScalerBuffer(self.running_x)
        running_y = ScalerBuffer(self.running_y)
        running_o = ScalerBuffer(self.running_o)
        scale_x = ScalerBuffer(self.scale_x)
        scale_y = ScalerBuffer(self.scale_y)
        scale_o = ScalerBuffer(self.scale_o)
        if isinstance(input, IQTensor):
            self.is_not_from_iqtensor1 = False
            if input.bits != self.data_bits:
                input = Requant.apply(
                    input, input.bits, input.scale_data, self.data_bits)
            scale_x = ScalerBuffer(input.scale_data)
            running_x = ScalerBuffer(input.running_data)
        if isinstance(mat2, IQTensor):
            self.is_not_from_iqtensor2 = False
            if mat2.bits != self.data_bits:
                mat2 = Requant.apply(
                    mat2, mat2.bits, mat2.scale_data, self.data_bits)
            scale_y = ScalerBuffer(mat2.scale_data)
            running_y = ScalerBuffer(mat2.running_data)

        ret = BmmIntFunction.apply(input, mat2,
                                   self.data_bits, self.training, self.momentum,
                                   running_x, running_y, running_o, scale_x, scale_y, scale_o,
                                   self.prefix, self.dump, self.path, self.quant_mode, self.o_bits,
                                   self.quant, self.is_not_from_iqtensor1, self.is_not_from_iqtensor2, self.clamp_data)
        self.running_x.fill_(running_x())
        self.running_y.fill_(running_y())
        self.running_o.fill_(running_o())
        self.scale_x.fill_(scale_x())
        self.scale_y.fill_(scale_y())
        self.scale_o.fill_(scale_o())
        return ret

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination._metadata[prefix[:-1]
                              ] = local_metadata = dict(version=self._version)
        if is_in_onnx_export():
            assert self.running_x > 0, 'invalid running_x <=0'
            scale_x = ScalerBuffer(self.scale_x.data)
            scale_y = ScalerBuffer(self.scale_y.data)
            if self.is_not_from_iqtensor1:
                scale_x = ScalerBuffer(self.quant.running_to_scale(
                    self.running_x, self.data_bits, mode=self.quant_mode))
                self.scale_x.data.fill_(scale_x())
            if self.is_not_from_iqtensor2:
                scale_y = ScalerBuffer(self.quant.running_to_scale(
                    self.running_y, self.data_bits, mode=self.quant_mode))
                self.scale_y.data.fill_(scale_y())
            if self.o_bits is not None:
                scale_o = ScalerBuffer(self.quant.running_to_scale(
                    self.running_o, self.o_bits, mode=self.quant_mode))
                self.scale_o.data.fill_(scale_o())
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix +
                                  name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        ModuleIntConfig._load_from_state_dict_global(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
