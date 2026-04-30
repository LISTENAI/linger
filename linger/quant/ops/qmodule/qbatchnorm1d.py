import torch
from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME
from typing import Optional, Dict, Any


class QBatchNorm1dOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, qparam_dict=None):
        view_shape = (1, -1) if input.dim() == 2 else (1, -1, 1)
        return input * weight.view(*view_shape) + bias.view(*view_shape)

    @staticmethod
    def symbolic(g, input, weight, bias, qparam_dict=None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        is_input_qtensor = qparam_dict.get("is_input_qtensor", None)
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop("op_type", None)
        qparam_dict.pop("is_input_qtensor", None)
        if is_input_qtensor is False or is_input_qtensor is None:
            input = quantlinear(
                g,
                input,
                qparam_dict["scale_x_f"],
                qparam_dict["platform_s"],
                qparam_dict["x_bits_i"],
                0,
            )
        output = g.op(node_name, input, weight, bias, **qparam_dict)
        output.setType(input.type())
        return output


@register_qmodule(torch.nn.BatchNorm1d)
class QBatchNorm1d(QModuleMixin, torch.nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_export_weight", torch.empty(0))
        self.register_buffer("_export_bias", torch.empty(0))
        self._linger_prepare_onnx_export_ready = False

    @classmethod
    def qcreate(
        cls,
        module,
        activations_cfg: Optional[Dict[str, Any]] = None,
        weights_cfg: Optional[Dict[str, Any]] = None,
        bias_cfg: Optional[Dict[str, Any]] = None,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[Dict[str, Any]] = None,
    ):
        return cls(
            num_features=module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
            dtype=module.weight.dtype if module.weight is not None else torch.float32,
            device=device,
            activations_cfg=activations_cfg,
            weights_cfg=weights_cfg,
            bias_cfg=bias_cfg,
            constrain=constrain,
        )

    def _store_export_params(self, weight: torch.Tensor, bias: torch.Tensor):
        weight = weight.reshape(-1).detach()
        bias = bias.reshape(-1).detach()
        self._export_weight.resize_(weight.shape).copy_(weight)
        self._export_bias.resize_(bias.shape).copy_(bias)
        self._linger_prepare_onnx_export_ready = True
        return self._export_weight, self._export_bias

    def _batch_stats(self, input: torch.Tensor):
        if input.dim() == 2:
            dims = (0,)
        elif input.dim() == 3:
            dims = (0, 2)
        else:
            raise ValueError(f"Expected 2D or 3D input (got {input.dim()}D)")

        use_batch_stats = self.training or not self.track_running_stats or self.running_mean is None or self.running_var is None
        if use_batch_stats:
            mean = input.mean(dim=dims, keepdim=True)
            var = input.var(dim=dims, unbiased=False, keepdim=True)
            var = torch.clamp(var, min=1e-5)
        else:
            mean = self.running_mean.view(1, -1, *([1] * (input.dim() - 2)))
            var = self.running_var.view(1, -1, *([1] * (input.dim() - 2)))
        return dims, mean, var, use_batch_stats

    def _get_fake_affine(self, input: torch.Tensor):
        dims, mean, var, use_batch_stats = self._batch_stats(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if use_batch_stats and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)

            self.running_mean = (
                (1 - exponential_average_factor) * self.running_mean
                + exponential_average_factor * mean.squeeze().detach()
            )
            self.running_var = (
                (1 - exponential_average_factor) * self.running_var
                + exponential_average_factor * var.squeeze().detach()
            )

        weight = (
            self.weight
            if self.weight is not None
            else torch.ones(
                self.num_features,
                dtype=input.dtype,
                device=input.device,
            )
        )
        bias = (
            self.bias
            if self.bias is not None
            else torch.zeros(
                self.num_features,
                dtype=input.dtype,
                device=input.device,
            )
        )

        view_shape = (1, -1, *([1] * (input.dim() - 2)))
        sigma = 1.0 / torch.sqrt(var + self.eps)
        alpha = weight.view(*view_shape) * sigma
        beta = bias.view(*view_shape) - mean * alpha
        fake_alpha = self.weight_quantizer(alpha)
        fake_beta = self.bias_quantizer(
            beta,
            self.weight_quantizer.scale * self.input_quantizer.scale,
        )
        return fake_alpha, fake_beta

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            if not self._linger_prepare_onnx_export_ready:
                fake_alpha, fake_beta = self._get_fake_affine(input)
                export_weight = fake_alpha.reshape(-1).detach()
                export_bias = fake_beta.reshape(-1).detach()
            else:
                export_weight, export_bias = self._export_weight, self._export_bias
            qparam_dict = generate_onnx_qparam_dict(self, False)
            return QBatchNorm1dOnnxFunction.apply(input, export_weight, export_bias, qparam_dict)

        fake_alpha, fake_beta = self._get_fake_affine(input)
        if getattr(self, "_linger_prepare_onnx_export", False):
            self._store_export_params(fake_alpha, fake_beta)
        out = fake_alpha * input + fake_beta
        return out
