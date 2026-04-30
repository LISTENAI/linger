import os
import sys
import types
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())


def _raise_stub(*args, **kwargs):
    raise RuntimeError("lingerext stub called unexpectedly")


if "lingerext" not in sys.modules:
    sys.modules["lingerext"] = types.SimpleNamespace(
        fake_quant=_raise_stub,
        bias_quant=_raise_stub,
        fake_quant_with_grad_scale=_raise_stub,
        bias_quant_with_grad_scale=_raise_stub,
    )

import linger
from linger.config import QUANT_CONFIGS
from linger.quant.qtensor import from_tensor_to_qtensor
from linger.quant.ops.qtensor.qgelu import QGelu, venusa_gelu_i32
from linger.quant.ops.qtensor.qswish import QSwish, venusa_swish_i32
from linger.utils import PlatForm


@pytest.fixture(autouse=True)
def _setup_quant_config():
    old_platform = QUANT_CONFIGS.platform
    old_device = QUANT_CONFIGS.device
    old_calibration = QUANT_CONFIGS.calibration
    QUANT_CONFIGS.platform = PlatForm.venusA
    QUANT_CONFIGS.device = torch.device("cpu")
    QUANT_CONFIGS.calibration = False
    yield
    QUANT_CONFIGS.platform = old_platform
    QUANT_CONFIGS.device = old_device
    QUANT_CONFIGS.calibration = old_calibration


def test_piecewise_int_forward():
    sample = torch.tensor(
        [-2147483647, -123456789, -1, 0, 1, 123456789, 2147483647],
        dtype=torch.int32,
    )
    gelu_out = venusa_gelu_i32(sample)
    swish_out = venusa_swish_i32(sample)

    assert gelu_out.dtype == torch.int32
    assert swish_out.dtype == torch.int32
    assert gelu_out.tolist() == [-22920, -22026943, -86064, -86064, -86063, 101429846, 2147460727]
    assert swish_out.tolist() == [-56416, -35096076, -147248, -147248, -147247, 88360713, 2147427231]


def test_piecewise_dequant_matches_float_scale_domain():
    x = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    x_q27 = (x * (1 << 27)).round().to(torch.int32)

    gelu = venusa_gelu_i32(x_q27).float() / float(2**27)
    swish = venusa_swish_i32(x_q27).float() / float(2**27)

    gelu_ref = F.gelu(x)
    swish_ref = F.silu(x)

    assert torch.allclose(gelu, gelu_ref, atol=0.06, rtol=0.03)
    assert torch.allclose(swish, swish_ref, atol=0.06, rtol=0.03)


def test_functional_gelu_and_silu_with_qtensor_backward():
    class Net(nn.Module):
        def forward(self, x):
            x = F.gelu(x)
            return F.silu(x)

    model = linger.init(Net()).train()
    x = torch.randn(2, 4, requires_grad=True)
    x_quant = from_tensor_to_qtensor(x, torch.tensor(128, dtype=torch.int32), 8)
    out = model(x_quant)
    out.float().sum().backward()

    assert out.shape == x.shape
    assert hasattr(out, "scale")
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert any(isinstance(module, QGelu) for module in model.modules())
    assert any(isinstance(module, QSwish) for module in model.modules())


def test_nn_gelu_and_silu_quantized_training():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gelu = nn.GELU()
            self.swish = nn.SiLU()

        def forward(self, x):
            return self.swish(self.gelu(x))

    model = linger.init(Net()).train()
    x = torch.randn(2, 4, requires_grad=True)
    out = model(x)
    out.float().sum().backward()

    assert isinstance(model.gelu, QGelu)
    assert isinstance(model.swish, QSwish)
    assert out.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_qgelu_training_regression_loss_decreases():
    torch.manual_seed(2026)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            return self.fc2(x)

    model = linger.init(Net()).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(32, 8)
    target = torch.randn(32, 4)

    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        pred = model(x).float()
        loss = F.mse_loss(pred, target)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert isinstance(model.act, QGelu)
    assert losses[-1] < losses[0]
    assert losses[-1] < losses[0] * 0.9


def test_onnx_export_contains_qgelu_qswish():
    onnx = pytest.importorskip("onnx")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gelu = QGelu(activate_config=QUANT_CONFIGS.quant_info.to_dict(), num_input=1)
            self.swish = QSwish(activate_config=QUANT_CONFIGS.quant_info.to_dict(), num_input=1)

        def forward(self, x):
            return self.swish(self.gelu(x))

    model = Net().eval()

    path = os.path.join(os.getcwd(), "test_qgelu_qswish.onnx")
    linger.export(model, (torch.randn(1, 4),), path, opset_version=13)
    graph = onnx.load(path).graph
    op_types = [node.op_type for node in graph.node]
    assert "QGelu" in op_types
    assert "QSwish" in op_types
