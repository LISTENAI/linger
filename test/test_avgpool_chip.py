import torch
import torch.nn as nn

import linger
from linger.config import QUANT_CONFIGS
from linger.quant.ops.qmodule.qavgpool_chip import _arcs_div, _venus_a_div, _venus_div
from linger.quant.qtensor import QTensor, from_qtensor_to_tensor
from linger.utils import PlatForm


def _as_tensor(x):
    return from_qtensor_to_tensor(x) if isinstance(x, QTensor) else x


def _build_quant_avgpool1d():
    return linger.init(
        nn.Sequential(nn.AvgPool1d(kernel_size=3, stride=2, padding=1, count_include_pad=False)),
        disable_submodel=[],
    )


def _build_quant_avgpool2d():
    return linger.init(
        nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)),
        disable_submodel=[],
    )


def _assert_train_eval_match(model, x):
    device = next(model.parameters(), None)
    if device is None:
        device = next(model.buffers()).device
    else:
        device = device.device

    model.train()
    x_train = x.clone().to(device).requires_grad_(True)
    y_train = model(x_train)
    y_train_tensor = _as_tensor(y_train)
    y_train_tensor.sum().backward()
    assert x_train.grad is not None
    assert x_train.grad.abs().sum().item() > 0

    model.eval()
    with torch.no_grad():
        y_eval = model(x.detach().clone().to(device))
    y_eval_tensor = _as_tensor(y_eval)

    assert torch.allclose(y_train_tensor.detach(), y_eval_tensor.detach(), atol=1e-6, rtol=0)


def test_chip_div_reference_vectors():
    dividend = torch.tensor([9.0, 10.0, 17.0, -9.0, -10.0, -17.0, 127.0, -127.0], dtype=torch.float64)
    divisor = torch.full_like(dividend, 9.0)
    expected = torch.tensor([1.0, 1.0, 2.0, -1.0, -1.0, -2.0, 14.0, -14.0], dtype=torch.float64)

    assert torch.equal(_venus_div(dividend, divisor), expected)
    assert torch.equal(_venus_a_div(dividend, divisor), expected)
    assert torch.equal(_arcs_div(dividend, divisor), expected)


def test_avgpool1d_chip_alignment():
    x = torch.tensor(
        [[[1.0, -2.0, 3.0, 4.0, -1.0, 2.0, 0.5, -0.5, 1.5],
          [-1.0, 2.0, -3.0, 1.0, 0.0, 4.0, -2.0, 3.0, -4.0]]],
        dtype=torch.float32,
    )

    original_platform = QUANT_CONFIGS.platform
    try:
        for platform in (PlatForm.venus, PlatForm.venusA, PlatForm.arcs):
            QUANT_CONFIGS.platform = platform
            model = _build_quant_avgpool1d()
            _assert_train_eval_match(model, x)
    finally:
        QUANT_CONFIGS.platform = original_platform


def test_avgpool2d_chip_alignment():
    x = torch.tensor(
        [[[[1.0, -2.0, 3.0, 4.0, -1.0],
           [0.0, 2.0, -3.0, 1.0, 2.0],
           [1.5, -0.5, 2.5, -1.5, 0.5],
           [3.0, 1.0, -2.0, 0.0, -4.0],
           [2.0, -1.0, 1.0, 2.0, -3.0]],
          [[-1.0, 2.0, -3.0, 1.0, 0.0],
           [4.0, -2.0, 3.0, -4.0, 2.0],
           [1.0, 0.5, -0.5, 2.5, -1.5],
           [0.0, -1.0, 2.0, 3.0, -2.0],
           [1.5, -2.5, 0.5, -0.5, 4.0]]]],
        dtype=torch.float32,
    )

    original_platform = QUANT_CONFIGS.platform
    try:
        for platform in (PlatForm.venus, PlatForm.venusA, PlatForm.arcs):
            QUANT_CONFIGS.platform = platform
            model = _build_quant_avgpool2d()
            _assert_train_eval_match(model, x)
    finally:
        QUANT_CONFIGS.platform = original_platform
