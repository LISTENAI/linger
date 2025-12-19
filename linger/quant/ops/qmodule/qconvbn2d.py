import torch
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from ....constrain.cconvbn2d import CConvBN2d, ConvBN2d
from ....onnx import generate_onnx_qparam_dict, QCustomOpSymbolic

@register_qmodule(ConvBN2d)
@register_qmodule(CConvBN2d)
class QConvBN2d(QModuleMixin, CConvBN2d):
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
        convbn_mdl = cls(
            in_channels=module.conv.in_channels,
            out_channels=module.conv.out_channels,
            kernel_size=module.conv.kernel_size,
            stride=module.conv.stride,
            padding=module.conv.padding,
            dilation=module.conv.dilation,
            groups=module.conv.groups,
            bias=module.conv.bias is not None,
            padding_mode=module.conv.padding_mode,
            eps=module.bn.eps,
            momentum=module.bn.momentum,
            affine=module.bn.affine,
            track_running_stats=module.bn.track_running_stats,

            dtype=module.conv.weight.dtype,
            device=device,
            activations_cfg=activations_cfg,
            weights_cfg=weights_cfg,
            bias_cfg=bias_cfg,
            constrain=constrain, 
        )
    
        convbn_mdl.weight = torch.nn.Parameter(
            module.conv.weight.detach().clone(),
            requires_grad=module.conv.weight.requires_grad
        )

        convbn_mdl.bias = torch.nn.Parameter(
            module.bn.bias.detach().clone(),
            requires_grad=module.bn.bias.requires_grad
        )

        convbn_mdl.in_channels = module.conv.in_channels
        convbn_mdl.out_channels = module.conv.out_channels
        convbn_mdl.dilation = module.conv.dilation
        convbn_mdl.kernel_size = module.conv.kernel_size
        convbn_mdl.padding = module.conv.padding
        convbn_mdl.stride = module.conv.stride
        convbn_mdl.groups = module.conv.groups
        convbn_mdl.output_padding = module.conv.output_padding
        convbn_mdl.padding_mode = module.conv.padding_mode
        return convbn_mdl

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # conv_rlt = self.conv(input) # for calculate bn mean and var
            conv_rlt = self.conv._conv_forward(input, self.conv.weight, self.conv.bias)
            N, C, H, W = conv_rlt.size()
            bn_size = N * H * W
            conv_rlt = conv_rlt.permute(1, 0, 2, 3).contiguous().view(C, bn_size)
            sum_ = conv_rlt.sum(1)
            sum_square_ = conv_rlt.pow(2).sum(1)
            mean_ = sum_ / bn_size
            sum_var_ = sum_square_ - sum_ * mean_
            unbias_var_ = sum_var_ / (bn_size - 1)  # 无偏方差，用 unbias_var（除 N-1）来更新 running_var（长期的、用于推理时的估计），在统计上更合理（减少估计偏差）
            unbias_var_ = torch.clamp(unbias_var_, min=0.0)
            self.bn.running_mean = (
                (1 - self.bn.momentum) * self.bn.running_mean + self.bn.momentum * mean_.detach())
            self.bn.running_var = (
                (1 - self.bn.momentum) * self.bn.running_var + self.bn.momentum * unbias_var_.detach())

            bias_var_ = sum_var_ / bn_size  # 计算当前 batch 的标准差用于“在该 batch 上归一化” —— 这是即时、直接的标准化数学操作
            bias_var_ = torch.clamp(bias_var_, min=0.0)
            inv_std_ = 1 / (bias_var_ + self.bn.eps).pow(0.5)
            bn_rlt = ((conv_rlt - mean_.unsqueeze(1)) * inv_std_.unsqueeze(1) * 
                        self.bn.weight.unsqueeze(1) + self.bn.bias.unsqueeze(1))
            bn_rlt = bn_rlt.view(C, N, H, W).permute(1, 0, 2, 3).contiguous()
            w_bn_ = self.bn.weight.div(torch.sqrt(unbias_var_ + self.bn.eps))
            new_weight = self.conv.weight.mul(w_bn_.view(-1, 1, 1, 1))
            if self.conv.bias is not None:
                b_conv_ = self.conv.bias
            else:
                b_conv_ = torch.zeros(self.conv.weight.size(0), device=input.device)
            b_bn_ = self.bn.bias - self.bn.weight.mul(mean_).div(torch.sqrt(unbias_var_ + self.bn.eps))
            new_bias = b_conv_.mul(w_bn_) + b_bn_

            alpha = 0.1

            # fake_quant new_weight and new_bias
            fake_weight = self.weight_quantizer(new_weight)
            fake_bias = self.bias_quantizer(new_bias)

            new_conv_rlt = F.conv2d(input, fake_weight, fake_bias, self.conv.stride,
                                    self.conv.padding, self.conv.dilation, self.conv.groups)
            output = alpha * bn_rlt + (1 - alpha) * new_conv_rlt
        else:
            w_bn_ = self.bn.weight.div(torch.sqrt(self.bn.eps + self.bn.running_var))
            new_weight = self.conv.weight.mul(w_bn_.view(-1, 1, 1, 1))
            if self.conv.bias is not None:
                b_conv_ = self.conv.bias
            else:
                b_conv_ = torch.zeros(self.conv.weight.size(0), device=input.device)
            b_bn_ = self.bn.bias - self.bn.weight.mul(self.bn.running_mean).div(
                torch.sqrt(self.bn.running_var + self.bn.eps))
            new_bias = b_conv_.mul(w_bn_) + b_bn_
            self.weight.data = new_weight
            self.bias.data = new_bias
            if torch.onnx.is_in_onnx_export():
                qparam_dict = generate_onnx_qparam_dict(self, False)
                output = QCustomOpSymbolic.apply(input, self.weight, self.bias, qparam_dict, self.input_quantizer.is_qtensor)
            else:
                output = F.conv2d(input, self.qweight, self.qbias, self.stride,
                              self.padding, self.dilation, self.groups)
        
        return output

