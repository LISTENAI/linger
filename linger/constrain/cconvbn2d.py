import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any
from .cmodule import CModuleMixin, register_cmodule
from .cutils import static_clip, dyn_clip_weight

class ConvBN2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 constrain: Optional[Dict[str, Any]] = None, dtype = torch.float32) -> None:
        super(ConvBN2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        self.bn = nn.BatchNorm2d(
            out_channels, eps, momentum, affine, track_running_stats)
        
        self.constrain = {} if constrain is None else constrain
        self.clamp_weight = self.constrain.get('clamp_weight_value', None)
        self.clamp_bias = self.constrain.get('clamp_bias_value', None)
        self.clamp_activation = self.constrain.get('clamp_activation_value', None)
        self.clamp_factor = self.constrain.get('clamp_factor_value', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            conv_rlt = self.conv._conv_forward(input, self.conv.weight, self.conv.bias) # for calculate bn mean and var
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
            if self.clamp_weight is not None:
                new_weight = static_clip(new_weight, self.clamp_weight)
            else:
                new_weight = dyn_clip_weight(new_weight, self.clamp_factor)

            if self.clamp_bias is not None:
                new_bias = static_clip(new_bias, self.clamp_bias)
            new_conv_rlt = F.conv2d(input, new_weight, new_bias, self.conv.stride,
                                    self.conv.padding, self.conv.dilation, self.conv.groups)
            output = alpha * bn_rlt + (1 - alpha) * new_conv_rlt
        else:
            w_bn_ = self.bn.weight.div(torch.sqrt(self.bn.eps + self.bn.running_var))
            new_weight = self.conv.weight.mul(w_bn_.view(-1, 1, 1, 1))
            if self.conv.bias is not None:
                b_conv_ = self.conv.bias
            else:
                b_conv_ = torch.zeros(self.conv.weight.size(0), device=input.device)
            # b_bn_ = self.bn.bias - self.bn.weight.mul(self.bn.running_mean).div(
            #     torch.sqrt(self.bn.running_var + self.bn.eps))
            # new_bias = b_conv_.mul(w_bn_) + b_bn_
            new_bias = (b_conv_ - self.bn.running_mean) * w_bn_ + self.bn.bias # 等价, 优化计算逻辑
            if self.clamp_weight is not None:
                new_weight = static_clip(new_weight, self.clamp_weight)
            else:
                new_weight = dyn_clip_weight(new_weight, self.clamp_factor)

            if self.clamp_bias is not None:
                new_bias = static_clip(new_bias, self.clamp_bias)
            output = F.conv2d(input, new_weight, new_bias, self.conv.stride,
                              self.conv.padding, self.conv.dilation, self.conv.groups)
        
        if self.clamp_activation is not None:
            output = static_clip(output, self.clamp_activation)

        return output

@register_cmodule(ConvBN2d)
class CConvBN2d(CModuleMixin, ConvBN2d):
    # def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
    #              eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
    #              constrain: Optional[Dict[str, Any]] = None, dtype = torch.float32) -> None:
    #     super(CConvBN2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
    #              eps, momentum, affine, track_running_stats, constrain, dtype)
        
    @classmethod
    def ccreate(
        cls,
        module,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[Dict[str, Any]] = None,
    ):
        return cls(
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
            constrain=constrain,
            open_ihook=False,
            open_ohook=False,
        )

    def extra_repr(self):
        s = nn.Conv2d.extra_repr(self.conv)
        s += ', '
        s += nn.BatchNorm2d.extra_repr(self.bn)
        extra_s = ', clamp_activation:{}, clamp_weight:{}, clamp_bias:{}, clamp_factor:{}'.format(self.clamp_activation, self.clamp_weight, self.clamp_bias, self.clamp_factor)
        return s + extra_s


