import torch
import torch.nn.functional as F
from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME
from typing import Optional, Union, Dict, Any

class QBatchNorm1dOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_mean, training, track_running_stats, running_var, weight, bias, momentum, eps, qparam_dict = None):
        if momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = momentum

        if training and track_running_stats:
            num_batches_tracked += 1
            if momentum is None:
                exponential_average_factor = 1.0 / float(num_batches_tracked)
            else:
                exponential_average_factor = momentum
        if training:
            bn_training = True
        else:
            bn_training = (running_mean is None) and (running_var is None)

        return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                running_mean
                if not training or track_running_stats
                else None,
                running_var if not training or track_running_stats else None,
                weight,
                bias,
                bn_training, 
                exponential_average_factor,
                eps,
            )
    @staticmethod
    def symbolic(g,  input, running_mean, training, track_running_stats, running_var, weight, bias, momentum, eps, qparam_dict = None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        is_input_qtensor = qparam_dict.get("is_input_qtensor", None)
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop('op_type', None)
        qparam_dict.pop('is_input_qtensor', None)
        if is_input_qtensor is False or is_input_qtensor is None:
            op_inner = quantlinear(g, input, qparam_dict['scale_x_f'], qparam_dict['platform_s'], qparam_dict['x_bits_i'], 0)
            input_list = [op_inner, weight]
        else:
            input_list = [input, weight]
        if bias is not None:
            input_list.append(bias)
        return g.op(
                node_name,
                *input_list,
                **qparam_dict
            )

@register_qmodule(torch.nn.BatchNorm1d)
class QBatchNorm1d(QModuleMixin, torch.nn.BatchNorm1d):
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

            dtype=module.weight.dtype,
            device=device,
            activations_cfg=activations_cfg,
            weights_cfg=weights_cfg,
            bias_cfg=bias_cfg,
            constrain = constrain, 
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            qparam_dict = generate_onnx_qparam_dict(self, False)
            return QBatchNorm1dOnnxFunction.apply(input, self.running_mean, self.training, self.track_running_stats, self.running_var, self.weight, self.bias, self.momentum, self.eps, qparam_dict)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)

        # 兼容 [N, C] 或 [N, C, L]
        if input.dim() == 2:
            dims = (0,)
        elif input.dim() == 3:
            dims = (0, 2)
        else:
            raise ValueError(f"Expected 2D or 3D input (got {input.dim()}D)")

        size = 1
        for d in dims:
            size *= input.size(d)

        if self.training:
            mean = input.mean(dim=dims, keepdim=True)
            var = input.var(dim=dims, unbiased=False, keepdim=True)
            var = torch.clamp(var, min=1e-5)

            # 更新运行均值与方差（无梯度）
            self.running_mean = (
                (1 - exponential_average_factor) * self.running_mean +
                exponential_average_factor * mean.squeeze().detach()
            )
            self.running_var = (
                (1 - exponential_average_factor) * self.running_var +
                exponential_average_factor * var.squeeze().detach()
            )
        else:
            mean = self.running_mean.view(1, -1, *([1] * (input.dim() - 2)))
            var = self.running_var.view(1, -1, *([1] * (input.dim() - 2)))

        # 计算仿射变换参数
        sigma = 1.0 / torch.sqrt(var + self.eps)
        alpha = self.weight.view(1, -1, *([1] * (input.dim() - 2))) * sigma
        beta = self.bias.view(1, -1, *([1] * (input.dim() - 2))) - mean * alpha

        # 伪量化
        fake_alpha = self.weight_quantizer(alpha)
        fake_beta = self.bias_quantizer(beta)

        out = fake_alpha * input + fake_beta
        return out


