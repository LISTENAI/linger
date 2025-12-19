import torch
import torch.nn.functional as F
from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Union, Dict, Any

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

    def qforward(self, input: torch.Tensor) -> torch.Tensor:
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


