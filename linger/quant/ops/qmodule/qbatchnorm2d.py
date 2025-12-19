import torch
import torch.nn.functional as F
from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Union, Dict, Any

@register_qmodule(torch.nn.BatchNorm2d)
class QBatchNorm2d(QModuleMixin, torch.nn.BatchNorm2d):
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
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        N,C,H,W= input.shape 
        size = N * H * W

        if self.training:
            mean = input.sum((0, 2, 3), keepdim=True) / size
            var = input.pow(2).sum((0, 2, 3), keepdim=True) / size - mean.pow(2)
            var = torch.clamp(var, min=1e-5)

            # Update running stats (no grad)
            self.running_mean = (
                (1 - exponential_average_factor) * self.running_mean +
                exponential_average_factor * mean.squeeze().detach()
            )
            self.running_var = (
                (1 - exponential_average_factor) * self.running_var +
                exponential_average_factor * var.squeeze().detach()
            )
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)

        sigma = 1 / torch.sqrt(var + self.eps)
        alpha = self.weight.view(1, -1, 1, 1) * sigma
        beta = self.bias.view(1, -1, 1, 1) - mean * alpha

        fake_alpha = self.weight_quantizer(alpha)
        fake_beta = self.bias_quantizer(beta)

        out = fake_alpha * input + fake_beta

        return out


