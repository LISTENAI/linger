import torch
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from ....constrain import ConvTransposeBN1d, CConvTransposeBN1d, static_clip, dyn_clip_weight
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME
from ....utils import QatMethod

class QConvTransposeBN1dOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, output_padding, dilation, groups, qparam_dict = None):
        return F.conv_transpose1d(input, weight, bias, stride, padding, 
                                  output_padding, groups, dilation)
    @staticmethod
    def symbolic(g,  input, weights, bias, stride, padding, dilation, groups, qparam_dict = None):
        op_type = qparam_dict.get("op_type", "QGeneric")
        is_input_qtensor = qparam_dict.get("is_input_qtensor", None)
        node_name = f"{QDOMAIN_NAME}::{op_type}"
        qparam_dict.pop('op_type', None)
        qparam_dict.pop('is_input_qtensor', None)
        if is_input_qtensor is False or is_input_qtensor is None:
            op_inner = quantlinear(g, input, qparam_dict['scale_x_f'], qparam_dict['platform_s'], qparam_dict['x_bits_i'], 0)
            input_list = [op_inner, weights]
        else:
            input_list = [input, weights]
        if bias is not None:
            input_list.append(bias)
        return g.op(
                node_name,
                *input_list,
                **qparam_dict
            )

@register_qmodule(ConvTransposeBN1d)
@register_qmodule(CConvTransposeBN1d)
class QConvTransposeBN1d(QModuleMixin, CConvTransposeBN1d):
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
            output_padding=module.conv.output_padding,
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
        convbn_mdl.output_padding = module.conv.output_padding
        convbn_mdl.stride = module.conv.stride
        convbn_mdl.groups = module.conv.groups
        convbn_mdl.padding_mode = module.conv.padding_mode
        return convbn_mdl

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            # conv_rlt = self.conv(input) # for calculate bn mean and var
            conv_rlt = F.conv_transpose1d(input, self.conv.weight, self.conv.bias, self.stride, self.padding, 
                                            self.output_padding, self.groups, self.dilation,)
            N, C, H = conv_rlt.size()
            bn_size = N * H
            conv_rlt = conv_rlt.permute(1, 0, 2).contiguous().view(C, bn_size)
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
            bn_rlt = bn_rlt.view(C, N, H).permute(1, 0, 2).contiguous()
            w_bn_ = self.bn.weight.div(torch.sqrt(unbias_var_ + self.bn.eps))
            new_weight = self.conv.weight.mul(w_bn_.view(1, -1, 1))
            if self.conv.bias is not None:
                b_conv_ = self.conv.bias
            else:
                b_conv_ = torch.zeros(self.conv.weight.size(1), device=input.device)
            b_bn_ = self.bn.bias - self.bn.weight.mul(mean_).div(torch.sqrt(unbias_var_ + self.bn.eps))
            new_bias = b_conv_.mul(w_bn_) + b_bn_
            # MOM策略时，clip信息无法固定给到new_weight, new_bias。TQT时会调用校准，scale自动截断
            if self.weight_quantizer.qat_method == QatMethod.MOM:
                if not self.weight_quantizer.is_init_mom_clamp_weight:
                    self.weight_quantizer.is_init_mom_clamp_weight.fill_(True) # 关闭weight_quantizer的初始化操作
                if self.weight_quantizer.clamp_value is not None:
                    new_weight = static_clip(new_weight, self.weight_quantizer.clamp_value)
                else:
                    new_weight = dyn_clip_weight(new_weight, self.weight_quantizer.clamp_factor)

                if self.bias_quantizer.clamp_value is not None:
                    new_bias = static_clip(new_bias, self.bias_quantizer.clamp_value)
            # 结束

            alpha = 0.1

            # fake_quant new_weight and new_bias
            fake_weight = self.weight_quantizer(new_weight)
            fake_bias = self.bias_quantizer(new_bias)

            new_conv_rlt = F.conv_transpose1d(input, fake_weight, fake_bias, self.stride, self.padding, 
                                            self.output_padding, self.groups, self.dilation)
            output = alpha * bn_rlt + (1 - alpha) * new_conv_rlt
        else:
            w_bn_ = self.bn.weight.div(torch.sqrt(self.bn.eps + self.bn.running_var))
            new_weight = self.conv.weight.mul(w_bn_.view(1, -1, 1))
            if self.conv.bias is not None:
                b_conv_ = self.conv.bias
            else:
                b_conv_ = torch.zeros(self.conv.weight.size(1), device=input.device)
            b_bn_ = self.bn.bias - self.bn.weight.mul(self.bn.running_mean).div(
                torch.sqrt(self.bn.running_var + self.bn.eps))
            new_bias = b_conv_.mul(w_bn_) + b_bn_
            # MOM策略时，clip信息无法固定给到new_weight, new_bias。TQT时会调用校准，scale自动截断
            if self.weight_quantizer.qat_method == QatMethod.MOM:
                if not self.weight_quantizer.is_init_mom_clamp_weight:
                    self.weight_quantizer.is_init_mom_clamp_weight.fill_(True) # 关闭weight_quantizer的初始化操作
                if self.weight_quantizer.clamp_value is not None:
                    new_weight = static_clip(new_weight, self.weight_quantizer.clamp_value)
                else:
                    new_weight = dyn_clip_weight(new_weight, self.weight_quantizer.clamp_factor)

                if self.bias_quantizer.clamp_value is not None:
                    new_bias = static_clip(new_bias, self.bias_quantizer.clamp_value)
            # 结束
            self.weight.data = new_weight
            self.bias.data = new_bias
            if torch.onnx.is_in_onnx_export():
                qparam_dict = generate_onnx_qparam_dict(self, False)
                return QConvTransposeBN1dOnnxFunction.apply(input, self.qweight, self.qbias, self.stride, self.padding, self.output_padding, self.dilation, self.groups, qparam_dict)
            else:
                output = F.conv_transpose1d(input, self.qweight, self.qbias, self.stride, self.padding, 
                                            self.output_padding, self.groups, self.dilation)
        return output

