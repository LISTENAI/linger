import torch
import torch.nn as nn

import linger
from linger.config import QUANT_CONFIGS
from linger.quant.qtensor import QTensor, from_qtensor_to_tensor
from linger.utils import PlatForm


def _as_tensor(x):
    return from_qtensor_to_tensor(x) if isinstance(x, QTensor) else x


def _build_gru(*, bidirectional=False):
    model = nn.Sequential(
        nn.GRU(
            input_size=6,
            hidden_size=4,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
    )
    return linger.init(model, disable_submodel=[])


def _run_gru_case(model, x, packed=False):
    gru = model[0]
    device = next(model.parameters()).device
    atol = 1.0 / 128.0

    if packed:
        lengths = torch.tensor([5, 3], dtype=torch.int64)
        x = x.to(device)
        gru.train()
        output, hidden = gru((x.clone().requires_grad_(True), lengths, True, False))
        y = _as_tensor(output[0])
        loss = y.sum() + hidden.sum()
        loss.backward()

        gru.eval()
        with torch.no_grad():
            output_eval, hidden_eval = gru((x, lengths, True, False))
        y_eval = _as_tensor(output_eval[0])
        assert torch.allclose(y.detach(), y_eval.detach(), atol=atol, rtol=0)
        assert torch.allclose(hidden.detach(), hidden_eval.detach(), atol=atol, rtol=0)
    else:
        x = x.to(device).requires_grad_(True)
        gru.train()
        output, hidden = gru(x)
        y = _as_tensor(output)
        loss = y.sum() + hidden.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0

        gru.eval()
        with torch.no_grad():
            output_eval, hidden_eval = gru(x.detach())
        y_eval = _as_tensor(output_eval)
        assert torch.allclose(y.detach(), y_eval.detach(), atol=atol, rtol=0)
        assert torch.allclose(hidden.detach(), hidden_eval.detach(), atol=atol, rtol=0)


def test_gru_chip_alignment_single_direction():
    x = torch.tensor(
        [[[1.0, -2.0, 3.0, 0.5, -1.0, 2.0],
          [0.0, 2.0, -3.0, 1.0, 2.0, -1.0],
          [1.5, -0.5, 2.5, -1.5, 0.5, 3.0],
          [3.0, 1.0, -2.0, 0.0, -4.0, 2.0],
          [2.0, -1.0, 1.0, 2.0, -3.0, -0.5]],
         [[-1.0, 2.0, -3.0, 1.0, 0.0, 4.0],
          [4.0, -2.0, 3.0, -4.0, 2.0, 1.5],
          [1.0, 0.5, -0.5, 2.5, -1.5, -2.0],
          [0.0, -1.0, 2.0, 3.0, -2.0, 0.5],
          [1.5, -2.5, 0.5, -0.5, 4.0, 1.0]]],
        dtype=torch.float32,
    )

    original_platform = QUANT_CONFIGS.platform
    try:
        for platform in (PlatForm.venus, PlatForm.venusA, PlatForm.arcs):
            QUANT_CONFIGS.platform = platform
            model = _build_gru(bidirectional=False)
            _run_gru_case(model, x, packed=False)
    finally:
        QUANT_CONFIGS.platform = original_platform


def test_gru_chip_alignment_packed_bidirectional():
    x = torch.tensor(
        [[[1.0, -2.0, 3.0, 0.5, -1.0, 2.0],
          [0.0, 2.0, -3.0, 1.0, 2.0, -1.0],
          [1.5, -0.5, 2.5, -1.5, 0.5, 3.0],
          [3.0, 1.0, -2.0, 0.0, -4.0, 2.0],
          [2.0, -1.0, 1.0, 2.0, -3.0, -0.5]],
         [[-1.0, 2.0, -3.0, 1.0, 0.0, 4.0],
          [4.0, -2.0, 3.0, -4.0, 2.0, 1.5],
          [1.0, 0.5, -0.5, 2.5, -1.5, -2.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    original_platform = QUANT_CONFIGS.platform
    try:
        for platform in (PlatForm.venus, PlatForm.venusA, PlatForm.arcs):
            QUANT_CONFIGS.platform = platform
            model = _build_gru(bidirectional=True)
            _run_gru_case(model, x, packed=True)
    finally:
        QUANT_CONFIGS.platform = original_platform
