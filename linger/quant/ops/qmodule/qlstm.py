import math
import copy
import torch
import torch.nn as nn
from torch import _VF

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Dict, Any
from torch.nn.utils.rnn import PackedSequence

from ..qtensor.qsigmoid import RUNTIME_SIGMOID_PLATFORMS, sigmoid_requant_to_qbits, sigmoid_runtime_qbits
from ..qtensor.qtanh import RUNTIME_TANH_PLATFORMS, tanh_requant_to_qbits, tanh_runtime_qbits
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor
from ...quantizer import WQuantizer, AQuantizer, BQuantizer
from ....config import QUANT_CONFIGS
from ....utils import _unbind, _unbind_packed, _slice, hx_slice, QatMethod, PlatForm, QuantMode
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME

import lingerext

RUNTIME_EXACT_PLATFORMS = {PlatForm.venus, PlatForm.venusA, PlatForm.arcs, PlatForm.mars}


def scale_to_bits(scale):
    scale_value = float(scale)
    scale_bits = math.log2(scale_value)
    assert abs(scale_bits - round(scale_bits)) < 1e-6, f"scale {scale_value} is not power-of-two"
    return int(round(scale_bits))


def ste_round(x, round_mode=QuantMode.floor_add):
    if round_mode == QuantMode.floor_add:
        rounded = torch.floor(x + 0.5)
    elif round_mode == QuantMode.floor:
        rounded = torch.floor(x)
    elif round_mode == QuantMode.ceil:
        rounded = torch.ceil(x)
    else:
        rounded = torch.round(x)
    return x + (rounded - x).detach()


def ste_clamp(x, min_val, max_val):
    clamped = x.clamp(min=min_val, max=max_val)
    return x + (clamped - x).detach()


def quantize_to_int_ste(x, scale_bits, clamp_min=None, clamp_max=None, round_mode=QuantMode.floor_add):
    scale = float(2 ** scale_bits)
    quantized = ste_round(x.double() * scale, round_mode)
    if clamp_min is not None or clamp_max is not None:
        quantized = ste_clamp(quantized, clamp_min, clamp_max)
    return quantized


def clamp_to_int32_ste(x):
    return ste_clamp(x, -(2**31), 2**31 - 1)


def clamp_to_bits_ste(x, data_bits):
    return ste_clamp(x, -(2 ** (data_bits - 1)), 2 ** (data_bits - 1) - 1)


class _MatmulIntBiasSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_int, w_int, b_int):
        ctx.has_bias = b_int is not None
        if ctx.has_bias:
            ctx.save_for_backward(x_int, w_int, b_int)
        else:
            ctx.save_for_backward(x_int, w_int)

        x_i64 = x_int.detach().round().to(torch.int64)
        w_i64 = w_int.detach().round().to(torch.int64)
        out_i64 = torch.matmul(x_i64.to(torch.float64), w_i64.t().to(torch.float64)).round().to(torch.int64)
        if ctx.has_bias:
            out_i64 = out_i64 + b_int.detach().round().to(torch.int64)
        return out_i64.to(torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.has_bias:
            x_int, w_int, _ = ctx.saved_tensors
        else:
            x_int, w_int = ctx.saved_tensors

        grad_output = grad_output.to(torch.float64)
        grad_x = torch.matmul(grad_output, w_int.to(torch.float64))
        grad_w = torch.matmul(grad_output.transpose(0, 1), x_int.to(torch.float64))
        grad_b = grad_output.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, grad_b


def matmul_int_with_bias_ste(x_int, w_int, b_int):
    return _MatmulIntBiasSTE.apply(x_int, w_int, b_int)


class _RequantIntSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_int, src_bits, dst_bits):
        ctx.scale = float(2 ** (dst_bits - src_bits)) if dst_bits >= src_bits else float(2 ** (-(src_bits - dst_bits)))
        x_i64 = x_int.detach().round().to(torch.int64)
        if dst_bits >= src_bits:
            out_i64 = x_i64 << (dst_bits - src_bits)
        else:
            shift = src_bits - dst_bits
            out_i64 = (x_i64.to(torch.float64) * (2.0 ** (-shift)) + 0.5).floor().to(torch.int64)
        return out_i64.to(torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to(torch.float64) * ctx.scale, None, None


def requant_int_ste(x_int, src_bits, dst_bits):
    return _RequantIntSTE.apply(x_int, src_bits, dst_bits)


class _MulRequantIntSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_int, b_int, src_bits, dst_bits):
        ctx.scale = float(2 ** (dst_bits - src_bits)) if dst_bits >= src_bits else float(2 ** (-(src_bits - dst_bits)))
        ctx.save_for_backward(a_int, b_int)

        a_i64 = a_int.detach().round().to(torch.int64)
        b_i64 = b_int.detach().round().to(torch.int64)
        prod_i64 = a_i64 * b_i64
        if dst_bits >= src_bits:
            out_i64 = prod_i64 << (dst_bits - src_bits)
        else:
            shift = src_bits - dst_bits
            out_i64 = (prod_i64.to(torch.float64) * (2.0 ** (-shift)) + 0.5).floor().to(torch.int64)
        return out_i64.to(torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        a_int, b_int = ctx.saved_tensors
        grad_output = grad_output.to(torch.float64)
        scale = ctx.scale
        grad_a = grad_output * b_int.to(torch.float64) * scale
        grad_b = grad_output * a_int.to(torch.float64) * scale
        return grad_a, grad_b, None, None


def mul_requant_int_ste(a_int, b_int, src_bits, dst_bits):
    return _MulRequantIntSTE.apply(a_int, b_int, src_bits, dst_bits)


def _runtime_activation_kernel(kind):
    if kind == "sigmoid":
        if QUANT_CONFIGS.platform == PlatForm.venus:
            return lingerext.venus_qsigmoid_forward
        if QUANT_CONFIGS.platform == PlatForm.venusA:
            return lingerext.venusa_qsigmoid_forward
        if QUANT_CONFIGS.platform in {PlatForm.arcs, PlatForm.mars}:
            return lingerext.arcs_qsigmoid_forward
    elif kind == "tanh":
        if QUANT_CONFIGS.platform == PlatForm.venus:
            return lingerext.venus_qtanh_forward
        if QUANT_CONFIGS.platform == PlatForm.venusA:
            return lingerext.venusa_qtanh_forward
        if QUANT_CONFIGS.platform in {PlatForm.arcs, PlatForm.mars}:
            return lingerext.arcs_qtanh_forward
    raise ValueError(f"unsupported platform/kind pair: {QUANT_CONFIGS.platform}, {kind}")


class _RuntimeActivationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_int, kind, input_bits, output_bits, clamp_min, clamp_max):
        ctx.kind = kind
        ctx.input_bits = input_bits
        ctx.output_bits = output_bits
        ctx.save_for_backward(input_int)

        kernel = _runtime_activation_kernel(kind)
        input_i32 = input_int.detach().round().to(torch.int64).clamp_(clamp_min, clamp_max).to(torch.int32)
        output = kernel(input_i32.contiguous())
        if kind == "sigmoid" and QUANT_CONFIGS.platform in RUNTIME_SIGMOID_PLATFORMS:
            output = sigmoid_requant_to_qbits(output, sigmoid_runtime_qbits(QUANT_CONFIGS.platform), output_bits)
        elif kind == "tanh" and QUANT_CONFIGS.platform in RUNTIME_TANH_PLATFORMS:
            output = tanh_requant_to_qbits(output, tanh_runtime_qbits(QUANT_CONFIGS.platform), output_bits)
        return output.to(torch.float64)

    @staticmethod
    def backward(ctx, grad_output):
        input_int, = ctx.saved_tensors
        input_fp = (input_int.detach().to(torch.float64) / float(2 ** ctx.input_bits)).requires_grad_(True)
        with torch.enable_grad():
            if ctx.kind == "sigmoid":
                output_scaled = torch.sigmoid(input_fp) * float(2 ** ctx.output_bits)
            else:
                output_scaled = torch.tanh(input_fp) * float(2 ** ctx.output_bits)
            grad_input_fp = torch.autograd.grad(output_scaled, input_fp, grad_output.to(torch.float64))[0]
        return grad_input_fp / float(2 ** ctx.input_bits), None, None, None, None, None


def runtime_activation_q31_from_q27_ste(input_q27, kind):
    return _RuntimeActivationSTE.apply(input_q27, kind, 27, 31, -(2**31), 2**31 - 1)


def runtime_activation_q15_from_q11_ste(input_q11, kind):
    return _RuntimeActivationSTE.apply(input_q11, kind, 11, 15, -(2**15), 2**15 - 1)


def _select_direction_state(g, state, direction):
    index = torch.tensor([direction], dtype=torch.long)
    index_node = g.op("Constant", value_t=index)
    return g.op("Gather", state, index_node, axis_i=0)

class QLSTMOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hidden_state, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse,
                hidden_size, num_layers, batch_first, bidirectional, qparam_dict = None):
        output = None
        hidden_state = None
        cell_state = None
        batch_size = None
        seq_length = None
        num_directions = 2 if bidirectional else 1
        if batch_first:
            batch_size = input.size(0)
            seq_length = input.size(1)
            output = torch.randn(batch_size, seq_length, hidden_size*num_directions, device=input.device)
        else:
            batch_size = input.size(1)
            seq_length = input.size(0)
            output = torch.randn(seq_length, batch_size, hidden_size*num_directions, device=input.device)
        hidden_state = torch.zeros(num_directions, batch_size, hidden_size, device=input.device)
        cell_state = torch.zeros(num_directions, batch_size, hidden_size, device=input.device)
        return output, hidden_state, cell_state

    @staticmethod
    def symbolic(g, input, hidden_state, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse,
                hidden_size, num_layers, batch_first, bidirectional, qparam_dict = None):

        op_type = qparam_dict.get("op_type", "QGeneric")
        is_input_qtensor = qparam_dict.get("is_input_qtensor", None)
        has_hidden = bool(qparam_dict.get("has_hidden_i", 0))
        has_cell = bool(qparam_dict.get("has_cell_i", 0))
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop('op_type', None)
        qparam_dict.pop('is_input_qtensor', None)

        qparam_dict_r = qparam_dict.get("qparam_dict_r", None)
        qparam_dict.pop('qparam_dict_r', None)

        if is_input_qtensor is False or is_input_qtensor is None:
            op_inner = quantlinear(g, input, qparam_dict['scale_x_f'], qparam_dict['platform_s'], qparam_dict['x_bits_i'], 0)
            input_list = [op_inner, weight_ih, weight_hh]
        else:
            input_list = [input, weight_ih, weight_hh]
        if has_hidden and has_cell:
            hidden_state_f = hidden_state if qparam_dict_r is None else _select_direction_state(g, hidden_state, 0)
            cell_state_f = cell_state if qparam_dict_r is None else _select_direction_state(g, cell_state, 0)
            input_list[1:1] = [hidden_state_f, cell_state_f]
        if bias_ih is not None:
            input_list.append(bias_ih)
            input_list.append(bias_hh)
        lstm, hidden, cell = g.op(node_name, *input_list, **qparam_dict)

        if qparam_dict_r is not None:   # 双向RNN
            if is_input_qtensor is False or is_input_qtensor is None:
                op_inner = quantlinear(g, input, qparam_dict_r['scale_x_f'], qparam_dict_r['platform_s'], qparam_dict_r['x_bits_i'], 0)
                input_list_r = [op_inner, weight_ih_reverse, weight_hh_reverse]
            else:
                input_list_r = [input, weight_ih_reverse, weight_hh_reverse]
            if has_hidden and has_cell:
                input_list_r[1:1] = [
                    _select_direction_state(g, hidden_state, 1),
                    _select_direction_state(g, cell_state, 1),
                ]
            if bias_ih_reverse is not None:
                input_list_r.append(bias_ih_reverse)
                input_list_r.append(bias_hh_reverse)
            lstm_r, hidden_r, cell_r = g.op(node_name, *input_list_r, **qparam_dict_r)

            # 双向LSTM需要插入QCat
            cat_node_name = f"{QDOMAIN_NAME}::QCat"
            cat_input_list = [lstm, lstm_r]
            cat_param_dict = {}
            cat_param_dict['platform_s'] = qparam_dict.get("platform_s", None)
            cat_param_dict['quant_mode_s'] = qparam_dict.get("quant_mode_s", None)
            cat_param_dict['axis_i'] = int(2)
            cat_param_dict['x_0_bits_i'] = qparam_dict.get("o_bits_i", 8)
            cat_param_dict['scale_x_0_f'] = qparam_dict.get("scale_o_f", 1.0)
            cat_param_dict['x_1_bits_i'] = qparam_dict_r.get("o_bits_i", 8)
            cat_param_dict['scale_x_1_f'] = qparam_dict_r.get("scale_o_f", 1.0)
            cat_param_dict['o_bits_i'] = qparam_dict.get("o_bits_i", 8)
            cat_param_dict['scale_o_f'] = min(qparam_dict_r.get("scale_o_f", 1.0), qparam_dict.get("scale_o_f", 1.0))
            lstm = g.op(cat_node_name, *cat_input_list, **cat_param_dict)
            hidden = g.op("Concat", hidden, hidden_r, axis_i=0)
            cell = g.op("Concat", cell, cell_r, axis_i=0)

        return lstm, hidden, cell

class QLSTMCell(nn.Module):
    def _forward_venus_exact(self, input_x, hidden, cx, weight_ih, weight_hh, bias_ih, bias_hh,
                             input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer,
                             cell_quantizer):
        active_q_in = 11
        q_i = scale_to_bits(input_quantizer.scale)
        q_h = scale_to_bits(hidden_quantizer.scale)
        q_c = scale_to_bits(cell_quantizer.scale)
        q_iw = scale_to_bits(weightih_quantizer.scale)
        q_hw = scale_to_bits(weighthh_quantizer.scale)
        q_ib = q_i + q_iw
        q_hb = q_h + q_hw

        x_int = quantize_to_int_ste(input_x, q_i)
        hidden_int = quantize_to_int_ste(hidden, q_h)
        cell_int = quantize_to_int_ste(cx, q_c)
        w_ih_int = quantize_to_int_ste(weight_ih, q_iw)
        w_hh_int = quantize_to_int_ste(weight_hh, q_hw)
        b_ih_int = quantize_to_int_ste(bias_ih, q_ib) if bias_ih is not None else None
        b_hh_int = quantize_to_int_ste(bias_hh, q_hb) if bias_hh is not None else None

        gi_int = matmul_int_with_bias_ste(x_int, w_ih_int, b_ih_int)
        gh_int = matmul_int_with_bias_ste(hidden_int, w_hh_int, b_hh_int)
        if q_ib != active_q_in:
            gi_int = requant_int_ste(gi_int, q_ib, active_q_in)
        if q_hb != active_q_in:
            gh_int = requant_int_ste(gh_int, q_hb, active_q_in)
        gi_int = clamp_to_int32_ste(gi_int)
        gh_int = clamp_to_int32_ste(gh_int)

        gates_int = clamp_to_int32_ste(gi_int + gh_int)
        G_i_int, G_f_int, G_c_int, G_o_int = gates_int.chunk(4, 1)
        G_i_q15 = runtime_activation_q15_from_q11_ste(clamp_to_int32_ste(G_i_int), "sigmoid")
        G_f_q15 = runtime_activation_q15_from_q11_ste(clamp_to_int32_ste(G_f_int), "sigmoid")
        G_c_q15 = runtime_activation_q15_from_q11_ste(clamp_to_int32_ste(G_c_int), "tanh")
        G_o_q15 = runtime_activation_q15_from_q11_ste(clamp_to_int32_ste(G_o_int), "sigmoid")

        cell_q15 = requant_int_ste(cell_int, q_c, 15) if q_c != 15 else cell_int
        forget_part_q30 = mul_requant_int_ste(G_f_q15, cell_q15, 30, 30)
        input_part_q30 = mul_requant_int_ste(G_i_q15, G_c_q15, 30, 30)
        cell_next_q15 = clamp_to_int32_ste(requant_int_ste(forget_part_q30 + input_part_q30, 30, 15))
        tanh_cell_q15 = runtime_activation_q15_from_q11_ste(requant_int_ste(cell_next_q15, 15, active_q_in), "tanh")

        cy_int = requant_int_ste(cell_next_q15, 15, q_c) if q_c != 15 else cell_next_q15
        hidden_bits = getattr(hidden_quantizer, "data_bits", 8)
        hy_int = clamp_to_bits_ste(requant_int_ste(G_o_q15 * tanh_cell_q15, 30, q_h), hidden_bits)
        hy = hy_int.to(dtype=input_x.dtype) / float(2 ** q_h)
        cy = cy_int.to(dtype=input_x.dtype) / float(2 ** q_c)
        return hy, cy

    def _forward_exact_runtime(self, input_x, hidden, cx, weight_ih, weight_hh, bias_ih, bias_hh,
                               input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer,
                               cell_quantizer):
        active_q_in = 27
        q_i = scale_to_bits(input_quantizer.scale)
        q_h = scale_to_bits(hidden_quantizer.scale)
        q_c = scale_to_bits(cell_quantizer.scale)
        q_iw = scale_to_bits(weightih_quantizer.scale)
        q_hw = scale_to_bits(weighthh_quantizer.scale)
        q_ib = q_i + q_iw
        q_hb = q_h + q_hw

        x_int = quantize_to_int_ste(input_x, q_i)
        hidden_int = quantize_to_int_ste(hidden, q_h)
        cell_int = quantize_to_int_ste(cx, q_c)
        w_ih_int = quantize_to_int_ste(weight_ih, q_iw)
        w_hh_int = quantize_to_int_ste(weight_hh, q_hw)
        b_ih_int = quantize_to_int_ste(bias_ih, q_ib) if bias_ih is not None else None
        b_hh_int = quantize_to_int_ste(bias_hh, q_hb) if bias_hh is not None else None

        gi_int = matmul_int_with_bias_ste(x_int, w_ih_int, b_ih_int)
        gh_int = matmul_int_with_bias_ste(hidden_int, w_hh_int, b_hh_int)
        if q_ib != active_q_in:
            gi_int = requant_int_ste(gi_int, q_ib, active_q_in)
        if q_hb != active_q_in:
            gh_int = requant_int_ste(gh_int, q_hb, active_q_in)
        gi_int = clamp_to_int32_ste(gi_int)
        gh_int = clamp_to_int32_ste(gh_int)

        gates_int = clamp_to_int32_ste(gi_int + gh_int)
        G_i_int, G_f_int, G_c_int, G_o_int = gates_int.chunk(4, 1)
        G_i_q31 = runtime_activation_q31_from_q27_ste(clamp_to_int32_ste(G_i_int), "sigmoid")
        G_f_q31 = runtime_activation_q31_from_q27_ste(clamp_to_int32_ste(G_f_int), "sigmoid")
        G_c_q31 = runtime_activation_q31_from_q27_ste(clamp_to_int32_ste(G_c_int), "tanh")
        G_o_q31 = runtime_activation_q31_from_q27_ste(clamp_to_int32_ste(G_o_int), "sigmoid")

        G_i_q15 = requant_int_ste(G_i_q31, 31, 15)
        G_f_q15 = requant_int_ste(G_f_q31, 31, 15)
        G_c_q15 = requant_int_ste(G_c_q31, 31, 15)
        G_o_q15 = requant_int_ste(G_o_q31, 31, 15)

        cell_q27 = requant_int_ste(cell_int * G_f_q15 + G_i_q15 * G_c_q15, 30, active_q_in)
        tanh_cell_q31 = runtime_activation_q31_from_q27_ste(clamp_to_int32_ste(cell_q27), "tanh")
        tanh_cell_q15 = requant_int_ste(tanh_cell_q31, 31, 15)

        cy_int = requant_int_ste(cell_q27, active_q_in, q_c)
        hidden_bits = getattr(hidden_quantizer, "data_bits", 8)
        hy_int = clamp_to_bits_ste(requant_int_ste(G_o_q15 * tanh_cell_q15, 30, q_h), hidden_bits)
        hy = hy_int.to(dtype=input_x.dtype) / float(2 ** q_h)
        cy = cy_int.to(dtype=input_x.dtype) / float(2 ** q_c)
        return hy, cy

    def forward(self, input_x, hidden, cx, weight_ih, weight_hh, bias_ih, bias_hh,
                input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer,
                cell_quantizer):
        platform = QUANT_CONFIGS.platform
        if platform == PlatForm.venus:
            return self._forward_venus_exact(
                input_x, hidden, cx, weight_ih, weight_hh, bias_ih, bias_hh,
                input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer,
                cell_quantizer,
            )
        if platform in {PlatForm.venusA, PlatForm.arcs, PlatForm.mars}:
            return self._forward_exact_runtime(
                input_x, hidden, cx, weight_ih, weight_hh, bias_ih, bias_hh,
                input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer,
                cell_quantizer,
            )
        raise NotImplementedError(f"QLSTM exact runtime is unsupported on platform {platform}")
        
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
            self.input_quantizer.is_qtensor = True
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
        return fake_hidden.float()
    
    def quantize_lstm_out(self, input: torch.Tensor, direct) -> torch.Tensor:
        if direct:
            fake_out = self.output_reverse_quantizer(input)
        else:
            fake_out = self.output_quantizer(input)
        return fake_out
        # return from_tensor_to_qtensor(fake_out, self.output_quantizer.scale, self.output_quantizer.data_bits)
    
    def update_output_quantizer_running_data(self, direct):
        if direct:
            if self.output_reverse_quantizer.qat_method == QatMethod.TQT:
                return
            self.output_reverse_quantizer.running_data.fill_(float(self.hidden_reverse_quantizer.running_data))
        else:
            if self.output_quantizer.qat_method == QatMethod.TQT:
                return
            self.output_quantizer.running_data.fill_(float(self.hidden_quantizer.running_data))

    @staticmethod
    def _blend_exact_with_quantizer_grad(exact, quantized):
        return quantized + (exact - quantized).detach()

    def _quantize_hidden_step(self, hidden, direct):
        hidden = self.quantize_lstm_hidden(hidden, direct)
        self.update_output_quantizer_running_data(direct)
        return hidden

    def _quantize_hidden_step_ste(self, hidden, direct, hidden_quantizer, runtime_hidden_scale):
        hidden_q = self._quantize_hidden_step(hidden, direct)
        hidden_quantizer.scale.fill_(runtime_hidden_scale)
        return self._blend_exact_with_quantizer_grad(hidden, hidden_q)

    def _quantize_output_ste(self, output, direct):
        output_q = self.quantize_lstm_out(output, direct)
        return self._blend_exact_with_quantizer_grad(output, output_q)
    
    def forward(self, input, *args, **kwargs):
        hx = kwargs.get("hx", None) if len(args) == 0 else args[0]
        if torch.onnx.is_in_onnx_export():
            return self.forward_onnx_export(input, hx)
        elif QUANT_CONFIGS.calibration:
            return self.forward_calibrate(input, hx)
        else:
            return self.forward_train(input, hx)

    def forward_onnx_export(self, input, hx=None):
        if self.num_layers != 1:
            raise NotImplementedError("QLSTM ONNX export only supports num_layers == 1")

        orig_input = input
        lengths = None
        if isinstance(orig_input, tuple):
            input, lengths, _, _ = orig_input
        else:
            lengths = None

        input = self.quantize_lstm_input(input)

        if hx is not None:
            if len(hx) != 2:
                raise RuntimeError("For batched 3-D input, hx should be a tuple of two tensors")
            hidden_state, cell_state = hx
        else:
            hidden_state = None
            cell_state = None

        output = None; hy = None; cy = None

        qparam_dict = generate_onnx_qparam_dict(self, False)
        qparam_dict['has_hidden_i'] = int(hidden_state is not None)
        qparam_dict['has_cell_i'] = int(cell_state is not None)
        if 'qparam_dict_r' in qparam_dict:
            qparam_dict['qparam_dict_r']['has_hidden_i'] = qparam_dict['has_hidden_i']
            qparam_dict['qparam_dict_r']['has_cell_i'] = qparam_dict['has_cell_i']
        # fake_quant weight, bias
        weight_ih, weight_hh = self.qweight_ih_hh
        bias_ih, bias_hh = self.qbias_ih_hh
        if self.bidirectional:
            weight_ih_r, weight_hh_r = self.qweight_ih_hh_reverse
            bias_ih_r, bias_hh_r = self.qbias_ih_hh_reverse
            output, hy, cy = QLSTMOnnxFunction.apply(input, hidden_state, cell_state, 
                                weight_ih, weight_hh, bias_ih, bias_hh, 
                                weight_ih_r, weight_hh_r, bias_ih_r, bias_hh_r,
                                self.hidden_size, self.num_layers, self.batch_first, self.bidirectional, qparam_dict)
        else:
            output, hy, cy = QLSTMOnnxFunction.apply(input, hidden_state, cell_state, 
                                weight_ih, weight_hh, bias_ih, bias_hh, 
                                None, None, None, None,
                                self.hidden_size, self.num_layers, self.batch_first, self.bidirectional, qparam_dict)

        output = from_tensor_to_qtensor(output, self.output_quantizer.scale, self.output_quantizer.data_bits)

        if isinstance(orig_input, tuple):
            return (output, lengths), (hy, cy)
        else:
            return output, (hy, cy)

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

            flat_weights = []
            w_ih, w_hh = self.qweight_ih_hh
            b_ih, b_hh = self.qbias_ih_hh
            flat_weights.extend([w_ih, w_hh])
            if self.bias:
                flat_weights.extend([b_ih, b_hh])

            if self.bidirectional:
                w_ih_r, w_hh_r = self.qweight_ih_hh_reverse
                b_ih_r, b_hh_r = self.qbias_ih_hh_reverse
                flat_weights.extend([w_ih_r, w_hh_r])
                if self.bias:
                    flat_weights.extend([b_ih_r, b_hh_r])

            if batch_sizes is None:
                result = _VF.lstm(input, hx,  flat_weights, self.bias, self.num_layers,
                                  self.dropout, False, self.bidirectional, self.batch_first)
            else:
                result = _VF.lstm(input, batch_sizes, hx,  flat_weights, self.bias,
                                  self.num_layers, self.dropout, False, self.bidirectional)
            output = result[0]
            hidden = result[1:]

            hidden = self.quantize_lstm_hidden(hidden[0], 0)
            hidden_r = self.quantize_lstm_hidden(hidden[0], 1)
            output = self.quantize_lstm_out(output, 0)
            output_r = self.quantize_lstm_out(output, 1)

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
        use_exact_runtime = QUANT_CONFIGS.platform in RUNTIME_EXACT_PLATFORMS
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

        input = self.quantize_lstm_input(input)

        # init hidden
        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions, max_batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
            if use_exact_runtime:
                hidden_init_scale = self.hidden_quantizer.scale if float(self.hidden_quantizer.scale) != 1.0 else self.input_quantizer.scale
                if float(self.cell_quantizer.scale) == 1.0:
                    self.cell_quantizer.scale.fill_(float(1 << 15))
            else:
                hidden_init_scale = self.input_quantizer.scale if self.training else None
            self.quantize_lstm_hidden(hx[0][0], 0, hidden_init_scale)  # init hidden_quantizer
            if self.bidirectional:
                if use_exact_runtime:
                    hidden_init_scale_r = self.hidden_reverse_quantizer.scale if float(self.hidden_reverse_quantizer.scale) != 1.0 else self.input_quantizer.scale
                else:
                    hidden_init_scale_r = hidden_init_scale
                self.quantize_lstm_hidden(hx[0][1], 1, hidden_init_scale_r)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)
            if use_exact_runtime and float(self.cell_quantizer.scale) == 1.0:
                self.cell_quantizer.scale.fill_(float(1 << 15))
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
        use_exact_runtime = QUANT_CONFIGS.platform in RUNTIME_EXACT_PLATFORMS
        exact_runtime_training = use_exact_runtime and self.training

        # fake_quant hidden, weight, bias
        weight_ih, weight_hh = self.qweight_ih_hh if direct == 0 else self.qweight_ih_hh_reverse
        bias_ih, bias_hh = self.qbias_ih_hh if direct == 0 else self.qbias_ih_hh_reverse

        step_outputs = []
        runtime_hidden_scale = float(hidden_quantizer.scale) if exact_runtime_training else None

        if exact_runtime_training:
            def postprocess_state(step_hidden, step_cell):
                step_hidden = self._quantize_hidden_step_ste(step_hidden, direct, hidden_quantizer, runtime_hidden_scale)
                return step_hidden, step_cell

            def finalize_output(direction_output, direction_hidden, direction_cell):
                return self._quantize_output_ste(direction_output, direct), (direction_hidden, direction_cell)
        elif use_exact_runtime:
            def postprocess_state(step_hidden, step_cell):
                return step_hidden, step_cell

            def finalize_output(direction_output, direction_hidden, direction_cell):
                return direction_output, (direction_hidden, direction_cell)
        else:
            def postprocess_state(step_hidden, step_cell):
                step_hidden = self._quantize_hidden_step(step_hidden, direct)
                return step_hidden, step_cell

            def finalize_output(direction_output, direction_hidden, direction_cell):
                return self.quantize_lstm_out(direction_output, direct), (direction_hidden, direction_cell)

        if batch_sizes is None:
            # input = input if direct == 0 else torch.cat(input.split(1,0)[::-1]) 
            input = input if direct == 0 else input.flip(0).contiguous()
            for input_x in input:
                hidden, cell_state = self.qlstm_cell_func(input_x, hidden, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                                        input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer,
                                        cell_quantizer)
                hidden, cell_state = postprocess_state(hidden, cell_state)
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
                hidden, cell_state = self.qlstm_cell_func(input_i, hidden, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                                                        input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer,
                                                        cell_quantizer)
                hidden, cell_state = postprocess_state(hidden, cell_state)
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
                hidden, cell_state = self.qlstm_cell_func(input_i, hidden, cell_state, weight_ih, weight_hh, bias_ih, bias_hh,
                                                        input_quantizer, hidden_quantizer, weightih_quantizer, weighthh_quantizer,
                                                        cell_quantizer)
                hidden, cell_state = postprocess_state(hidden, cell_state)
                step_outputs.append(hidden)
                last_batch_size = batch_len
            
            step_outputs = step_outputs[::-1] 
            output = torch.cat(step_outputs, 0)

        output, state = finalize_output(output, hidden, cell_state)
        return output, state
    

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
