import torch
import torch.nn.functional as F

from ....config import QUANT_CONFIGS
from ....utils import PlatForm, QuantMode, _single, _pair


_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


def _pow2(x, *, device):
    if torch.is_tensor(x):
        return torch.pow(torch.full_like(x, 2.0, dtype=torch.float64, device=device), x.to(dtype=torch.float64))
    return torch.pow(torch.tensor(2.0, dtype=torch.float64, device=device), torch.tensor(float(x), dtype=torch.float64, device=device))


def _floor_round_half_up(x):
    return torch.floor(x + 0.5)


def _clamp_to_int32(x):
    return x.clamp(_INT32_MIN, _INT32_MAX)


def _to_int_like(x):
    return _clamp_to_int32(torch.trunc(x))


def _trunc_to_int64(x):
    return torch.trunc(x).to(dtype=torch.int64)


def _resolve_pool_divisor(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    if divisor_override is not None:
        return torch.full(
            F.avg_pool2d(
                input,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
                divisor_override=1,
            ).shape,
            float(divisor_override),
            dtype=torch.float64,
            device=input.device,
        )

    kernel = _pair(kernel_size)
    if count_include_pad:
        divisor = float(kernel[0] * kernel[1])
        return torch.full(
            F.avg_pool2d(
                input,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
                divisor_override=1,
            ).shape,
            divisor,
            dtype=torch.float64,
            device=input.device,
        )

    valid = torch.ones_like(input, dtype=torch.float64)
    return F.avg_pool2d(
        valid,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override=1,
    ).clamp_min(1.0)


def _resolve_pool_divisor_1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    if divisor_override is not None:
        return torch.full(
            F.avg_pool1d(
                input,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
            ).shape,
            float(divisor_override),
            dtype=torch.float64,
            device=input.device,
        )

    kernel = _single(kernel_size)
    if count_include_pad:
        divisor = float(kernel[0])
        return torch.full(
            F.avg_pool1d(
                input,
                kernel_size,
                stride,
                padding,
                ceil_mode,
                count_include_pad,
            ).shape,
            divisor,
            dtype=torch.float64,
            device=input.device,
        )

    valid = torch.ones_like(input, dtype=torch.float64)
    return F.avg_pool1d(
        valid,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
    ).clamp_min(1.0)


def _venus_div(dividend, divisor):
    dividend = dividend.to(dtype=torch.float64)
    divisor = divisor.to(dtype=torch.float64)
    divisor_abs = torch.abs(divisor + 0.5)
    bit_detc = torch.ceil(torch.log2(divisor_abs))
    left_shift = 31.0 - bit_detc
    divisor_f = divisor * _pow2(left_shift, device=divisor.device) / float(1 << 31)
    div_indx = torch.floor(divisor_f * 32.0) / 32.0
    sign_div_indx = torch.sign(div_indx)
    div_indx_abs = torch.abs(div_indx).clamp_min(1e-12)
    r = _floor_round_half_up((1.0 / div_indx_abs) * float(1 << 29)) / float(1 << 29)
    r = r * sign_div_indx

    for _ in range(3):
        t = _floor_round_half_up((2.0 - r * divisor_f) * float(1 << 30)) / float(1 << 30)
        r = _floor_round_half_up(r * t * float(1 << 29)) / float(1 << 29)

    quotient = r * _pow2(left_shift - 31.0, device=divisor.device) * dividend
    quotient = _floor_round_half_up(quotient)
    return _to_int_like(quotient)


def _arcs_div(dividend, divisor, shift=0):
    dividend = dividend.to(dtype=torch.float64)
    divisor = divisor.to(dtype=torch.float64)
    divisor_abs = torch.abs(divisor).clamp_min(1.0)
    msb_loc_a = torch.ceil(torch.log2(divisor_abs))
    a_scale_factor = 31.0 - msb_loc_a
    a_scale = divisor * _pow2(a_scale_factor, device=divisor.device)
    a_fl = a_scale / float(1 << 31)
    a_rcp = 1.0 / ((torch.floor(a_fl * 32.0) / 32.0).clamp_min(1.0 / 32.0))
    a_rcp = _floor_round_half_up(a_rcp * float(1 << 29)) / float(1 << 29)

    for _ in range(3):
        a_rcp = a_rcp * (_floor_round_half_up((2.0 - a_fl * a_rcp) * float(1 << 30)) / float(1 << 30))
        a_rcp = _floor_round_half_up(a_rcp * float(1 << 29)) / float(1 << 29)

    q = a_rcp * dividend * _pow2(a_scale_factor - 31.0, device=divisor.device)
    if shift & (1 << 6):
        q = torch.floor(q * float(1 << (shift & (~(1 << 6)))))
    else:
        q = torch.floor(q * float(1 << shift) + 0.5)
    return _to_int_like(q)


def _venus_a_div(dividend, divisor, shift=0):
    dividend_i = _trunc_to_int64(dividend)
    divisor = divisor.to(dtype=torch.float64)
    divisor_abs = torch.abs(divisor).clamp_min(1.0)
    divisor_log2 = torch.log2(divisor_abs)
    msb_loc_a = torch.floor(divisor_log2)
    a_scale_factor = 30.0 - msb_loc_a
    a_scale = divisor * _pow2(a_scale_factor, device=divisor.device)
    a_fl = a_scale / float(1 << 31)
    a_rcp = 1.0 / ((torch.floor(a_fl * 32.0) / 32.0).clamp_min(1.0 / 32.0))
    a_rcp = _floor_round_half_up(a_rcp * float(1 << 29))

    for _ in range(3):
        a_delt = float(1 << 61) - a_scale * a_rcp
        a_delt_s = _trunc_to_int64(a_delt) >> 29
        a_delt_d = _floor_round_half_up(a_delt_s.to(dtype=torch.float64) / 2.0)
        a_rcp_i = a_rcp * a_delt_d
        a_rcp_s = _trunc_to_int64(a_rcp_i) >> 29
        a_rcp = _floor_round_half_up(a_rcp_s.to(dtype=torch.float64) / 2.0)

    iq = _trunc_to_int64(a_rcp) * dividend_i
    new_shift = (29.0 + (31.0 - a_scale_factor) - float(shift & (~(1 << 6))) - 1.0).to(dtype=torch.int64)
    right_shift = torch.bitwise_right_shift(iq, torch.clamp(new_shift, min=0))
    left_shift = torch.bitwise_left_shift(iq, torch.clamp(-new_shift, min=0))
    sq = torch.where(new_shift >= 0, right_shift, left_shift)
    if shift & (1 << 6):
        q = torch.floor(sq.to(dtype=torch.float64) / 2.0)
    else:
        q = torch.floor(sq.to(dtype=torch.float64) / 2.0 + 0.5)
    return _to_int_like(q)


def simulate_avgpool_int_div(dividend, divisor, platform=None):
    if platform is None:
        platform = QUANT_CONFIGS.platform

    if platform == PlatForm.venus:
        return _venus_div(dividend, divisor)
    if platform == PlatForm.arcs:
        return _arcs_div(dividend, divisor, shift=0)
    if platform in {PlatForm.venusA, PlatForm.mars}:
        return _venus_a_div(dividend, divisor, shift=0)
    return _to_int_like(torch.floor(dividend.to(dtype=torch.float64) / divisor.to(dtype=torch.float64) + 0.5))


def simulate_avgpool2d(input, *, scale, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, round_mode=QuantMode.floor_add):
    scale_value = float(scale) if torch.is_tensor(scale) else float(scale)
    scale_value = max(scale_value, 1e-12)
    quant_input = input * scale_value
    if round_mode == QuantMode.floor:
        quant_input = torch.floor(quant_input)
    elif round_mode == QuantMode.round:
        quant_input = torch.round(quant_input)
    else:
        quant_input = torch.floor(quant_input + 0.5)

    sum_int = F.avg_pool2d(
        quant_input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override=1,
    )
    divisor = _resolve_pool_divisor(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    out_int = simulate_avgpool_int_div(sum_int, divisor)
    return out_int.to(dtype=input.dtype) / scale_value


def simulate_avgpool1d(input, *, scale, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, round_mode=QuantMode.floor_add):
    scale_value = float(scale) if torch.is_tensor(scale) else float(scale)
    scale_value = max(scale_value, 1e-12)
    quant_input = input * scale_value
    if round_mode == QuantMode.floor:
        quant_input = torch.floor(quant_input)
    elif round_mode == QuantMode.round:
        quant_input = torch.round(quant_input)
    else:
        quant_input = torch.floor(quant_input + 0.5)

    sum_int = F.avg_pool1d(
        quant_input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
    ) * float(_single(kernel_size)[0] if divisor_override is None else divisor_override)
    divisor = _resolve_pool_divisor_1d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )
    out_int = simulate_avgpool_int_div(sum_int, divisor)
    return out_int.to(dtype=input.dtype) / scale_value
