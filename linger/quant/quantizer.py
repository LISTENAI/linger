import math
import torch
from typing import Optional, Union, Dict, Any

from .qtensor import QTensor
from .calibrate_funs import get_calibrate_function
from ..config  import QUANT_CONFIGS
from ..utils import QuantMode, QuantStrategy, ActivationType, FakeQuantMethod, QatMethod, PlatForm
import lingerext


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()

class FakeQuantOnnxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input
    
    @staticmethod
    def backward(ctx, gradOutput1):
        return gradOutput1
    
    @staticmethod
    def symbolic(g, input):
        return input

class FAKEQUANT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, data_bits, learning_data, scale_min, quant_min, quant_max):
        out, mask, scale_tmp = lingerext.fake_quant(input, data_bits, float(learning_data), float(scale_min), quant_min, quant_max)
        ctx.save_for_backward(mask.bool())
        return out, scale_tmp

    @staticmethod
    def backward(ctx, gradOutput1, gradOutput2):
        mask = ctx.saved_tensors
        return gradOutput1.masked_fill(mask[0], 0.0), None, None, None, None, None

class BIASQUANT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, data_bits, scale_bias, scale_min, quant_min, quant_max):
        out,mask = lingerext.bias_quant(input, data_bits, float(scale_bias), float(scale_min), quant_min, quant_max)
        ctx.save_for_backward(mask.bool())
        return out
    @staticmethod
    def backward(ctx, gradOutput):
        mask = ctx.saved_tensors
        return gradOutput.masked_fill(mask[0], 0.0), None, None, None, None, None

class FAKEQUANT_WITH_GRAD_SACLE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, data_bits, learning_data, scale_min, quant_min, quant_max):
        out, mask, learning_data_coff_back, scale_tmp = lingerext.fake_quant_with_grad_scale(input, data_bits, float(learning_data), float(scale_min), quant_min, quant_max)
        saved_tensors = [mask.bool(), learning_data_coff_back]
        ctx.save_for_backward(*saved_tensors)
        return out, scale_tmp

    @staticmethod
    def backward(ctx, gradOutput1, gradOutput2):
        mask, learning_data_coff_back = ctx.saved_tensors
        grad_learning_data = (gradOutput1 * learning_data_coff_back).sum()
        return gradOutput1.masked_fill(mask, 0.0), None, grad_learning_data, None, None, None

class BIASQUANT_WITH_GRAD_SACLE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, data_bits, scale_bias, scale_min, quant_min, quant_max):
        out,mask,scale_coff_back = lingerext.bias_quant_with_grad_scale(input, data_bits, float(scale_bias), float(scale_min), quant_min, quant_max)
        saved_tensors = [mask.bool(), scale_coff_back]
        ctx.save_for_backward(*saved_tensors)
        return out
    @staticmethod
    def backward(ctx, gradOutput):
        mask,  scale_coff_back = ctx.saved_tensors
        grad_scale = (gradOutput * scale_coff_back).sum()
        return gradOutput.masked_fill(mask, 0.0), None, grad_scale, None, None, None


class Quantizer(torch.nn.Module, ):
    def __init__(self, quantizer_cfg: Optional[Dict[str, Any]] = None, constrain: Optional[Dict[str, Any]] = None):
        super(Quantizer, self).__init__()
        if quantizer_cfg is None:
            quantizer_cfg = {}
        self.round_mode = quantizer_cfg.get('round_mode', QuantMode.floor_add)
        self.is_symmetry = quantizer_cfg.get('is_symmetry', True)

        self.qat_method = QUANT_CONFIGS.quant_info.qat_method
        
        # TQT策略使用，learning_data可学习，通过校准进行初始化
        if self.qat_method == QatMethod.TQT:
            self.learning_data = torch.nn.parameter.Parameter(torch.tensor([2.1]), )
            self.register_buffer("is_calibrate", torch.tensor(False, dtype=bool))
        elif self.qat_method == QatMethod.MOM:
            # MOM策略使用，running_data统计input.abs().max()获得
            self.register_buffer("running_data", torch.tensor(0.0))
            self.register_buffer("is_init", torch.tensor(False, dtype=bool))
            self.momentum = 0.1
        else:
            raise ValueError("Only TQT and MOM strategies are supported! ")
        # TQT和MOM训练时都更新scale(因为bias可能会调用当前训练步骤的scale); 推理时都通过scale(MOM为通过running_data保存的)计算以加快推理速度
        # MOM正确的scale（统计的running_data对应的scale）通过重写self.state_dict()函数保存的checkpoint中，故推理可直接使用scale;TQT一直都是正确的scale
        self.register_buffer("scale", torch.tensor(1.0))
        
    def init_quant_data(self, tensor):
        with torch.no_grad():
            if hasattr(self, "clamp_factor") and self.clamp_factor is not None:
                clamp_data = tensor.abs().mean() * self.clamp_factor
            elif hasattr(self, "clamp_value") and self.clamp_value is not None:
                clamp_data = self.clamp_value
            else:
                clamp_data = None
            tensor.data = tensor if clamp_data is None else torch.clamp(tensor, min = -clamp_data, max = clamp_data)

        calibrate_function = get_calibrate_function(self.calibrate_name)
        if hasattr(self, "activation_type") and self.activation_type == ActivationType.Relu:
            calibrate_tensor = torch.nn.functional.relu(tensor)
        else:
            calibrate_tensor = tensor
        calibrate_function(self, calibrate_tensor, self.data_bits)
        return tensor
    
    def quant_round(self, x, mode):
        if mode == QuantMode.floor_add: 
            out = ((x + 0.5).floor() - x).detach() + x
        elif mode == QuantMode.floor:
            out = (x.floor() - x).detach() + x
        else:
            out = (x.round() - x).detach() + x
        return out
       
    def fake_quant_native(self, input, scale_bias = None):
        if scale_bias is None:
            if self.qat_method == QatMethod.TQT:
                learning_data_temp = self.data_bits - 1 - self.learning_data
            elif self.qat_method == QatMethod.MOM:
                if hasattr(self, "activation_type") and self.activation_type == ActivationType.Relu:
                    abs_max = input.max()
                else:
                    abs_max = input.abs().max()
                abs_max = abs_max.clamp(min=1e-6).detach()
                if not self.is_init:
                    self.running_data = abs_max
                    self.is_init.fill_(True)
                else:
                    self.running_data = (1-self.momentum) * self.running_data + self.momentum * abs_max
                learning_data_temp = self.data_bits - 1 - self.running_data.log2()
                # self.running_data.to(input.device).mul_(1-self.momentum).add_(self.momentum * abs_max.detach())
            else:
                raise ValueError("Only TQT and MOM strategies are supported! ")
            learning_data = self.quant_round(learning_data_temp, QuantMode.round) #self.round_mode
            scale = 2**learning_data
            # scale = scale.clamp(min=1e-6, max=2**24)
        else:
            scale = scale_bias

        if hasattr(self, "min_scale") and scale > self.min_scale:
            scale = self.min_scale
            self.scale.fill_(float(scale))
        
        x_s = input * scale
        x_int = self.quant_round(x_s, self.round_mode)
        x_int = x_int.clamp(self.quant_min, self.quant_max)
        fake_input =  x_int / scale

        self.scale.fill_(float(scale))
        # self.scale.mul_(0).add_(scale) # MOM也必须要给scale初始化，便于后续(如bias)调用使用当前模块scale时保持正确

        return fake_input

    def fake_quant_native_mse(self, input, scale_bias = None):
        if scale_bias is None:
            if hasattr(self, "activation_type") and self.activation_type == ActivationType.Relu:
                abs_max = input.max()
            else:
                abs_max = input.abs().max()
            abs_max = abs_max.clamp(min=1e-6).detach()
            if not self.is_init:
                self.running_data = abs_max
                self.is_init.fill_(True)
            else:
                self.running_data = (1-self.momentum) * self.running_data + self.momentum * abs_max

            scale_top = self.quant_max / self.running_data
            scale_top = 2**self.quant_round(torch.log2(scale_top), QuantMode.ceil).double()
            inputs_q_top = (self.quant_round(input * scale_top, self.round_mode).clamp(
                self.quant_min, self.quant_max)) / scale_top
            scale_bottom = self.quant_max / self.running_data
            scale_bottom = 2**self.quant_round(torch.log2(scale_bottom), QuantMode.floor).double()
            inputs_q_bottom = (self.quant_round(input * scale_bottom, self.round_mode).clamp(
                self.quant_min, self.quant_max)) / scale_bottom
            score_top = lp_loss(input, inputs_q_top, p=2.4, reduction='all')
            score_bottom = lp_loss(input, inputs_q_bottom, p=2.4, reduction='all')

            scale = (scale_top if score_top < score_bottom else scale_bottom).float()
            
        else:
            scale = scale_bias
        
        scale = scale.clamp(min=1e-6, max=2**24)

        x_s = input * scale
        x_int = self.quant_round(x_s, self.round_mode)
        x_int = x_int.clamp(self.quant_min, self.quant_max)
        fake_input =  x_int / scale

        self.scale.fill_(float(scale))

        return fake_input

    def fake_quant_cuda(self, input, scale_bias = None):
        if hasattr(self, "min_scale"): # 只有add算子的outputquantizer有min_scale，其余情况min_scale和正常的scale相同
            min_scale = self.min_scale
        else:
            min_scale = float("inf")

        if scale_bias is None:
            if self.qat_method == QatMethod.TQT:
                out, scale_tmp = FAKEQUANT.apply(input, self.data_bits, self.learning_data, min_scale, self.quant_min, self.quant_max)
            elif self.qat_method == QatMethod.MOM:
                if hasattr(self, "activation_type") and self.activation_type == ActivationType.Relu:
                    abs_max = input.max()
                else:
                    abs_max = input.abs().max()
                abs_max = abs_max.clamp(min=1e-6).detach()
                if not self.is_init:
                    self.running_data = abs_max
                    self.is_init.fill_(True)
                else:
                    self.running_data = (1-self.momentum) * self.running_data + self.momentum * abs_max
                out, scale_tmp = FAKEQUANT.apply(input, self.data_bits, self.running_data.log2(), min_scale, self.quant_min, self.quant_max)
        else:
            out = BIASQUANT.apply(input, self.data_bits, scale_bias, min_scale, self.quant_min, self.quant_max)
            scale_tmp = scale_bias.detach()
        self.scale.fill_(scale_tmp) # MOM也必须要给scale初始化，便于后续(如bias)调用当前模块scale时保持正确
        return out

    def fake_quant_cuda_with_grad_scale(self, input, scale_bias = None):
        if hasattr(self, "min_scale"): # 只有add算子的outputquantizer有min_scale，其余情况min_scale和正常的scale相同
            min_scale = self.min_scale
        else:
            min_scale = float("inf")

        if scale_bias is None:
            if self.qat_method == QatMethod.TQT:
                out, scale_tmp = FAKEQUANT_WITH_GRAD_SACLE.apply(input, self.data_bits, self.learning_data, min_scale, self.quant_min, self.quant_max)
            elif self.qat_method == QatMethod.MOM:
                if hasattr(self, "activation_type") and self.activation_type == ActivationType.Relu:
                    abs_max = input.max()
                else:
                    abs_max = input.abs().max()
                abs_max = abs_max.clamp(min=1e-6).detach()
                if not self.is_init:
                    self.running_data = abs_max
                    self.is_init.fill_(True)
                else:
                    self.running_data = (1-self.momentum) * self.running_data + self.momentum * abs_max
                out, scale_tmp = FAKEQUANT_WITH_GRAD_SACLE.apply(input, self.data_bits, self.running_data.log2(), min_scale, self.quant_min, self.quant_max)
        else:
            out = BIASQUANT_WITH_GRAD_SACLE.apply(input, self.data_bits, scale_bias, min_scale, self.quant_min, self.quant_max)
            scale_tmp = scale_bias.detach()
        self.scale.fill_(scale_tmp) # MOM也必须要给scale初始化，便于后续(如bias)调用当前模块scale时保持正确
        return out

    def inference(self, input, scale=None):
        if hasattr(self, "min_scale"):
            min_scale = self.min_scale
        else:
            min_scale = float("inf")
        # 推理时固定走cuda路线，若为TQT模式，通过learning_data保存scale，若为MOM模式，通过running_data重新初始化scale
        if scale is None:
            if self.qat_method == QatMethod.MOM and self.running_data != 0.0:
                learning_data = self.data_bits - 1 - self.running_data.abs().max().log2()
                self.scale.fill_(float((2 ** (self.quant_round(learning_data, self.round_mode)))))
            scale = self.scale
        if QUANT_CONFIGS.quant_method == FakeQuantMethod.CUDA:
            out = BIASQUANT.apply(input, self.data_bits, scale, min_scale, self.quant_min, self.quant_max)
        else:
            out = self.fake_quant_native(input, scale)
        return out
    
    def forward(self, input, scale=None):
        if torch.onnx.is_in_onnx_export():
            return FakeQuantOnnxFunction.apply(input)
        elif QUANT_CONFIGS.calibration:
            return self.init_quant_data(input)
        elif not self.training:
            return self.inference(input, scale)
        else:
            # only weight clamp
            if self.qat_method == QatMethod.MOM and hasattr(self, "is_init_mom_clamp_weight") and self.is_init_mom_clamp_weight == False:
                with torch.no_grad():
                    if hasattr(self, "clamp_factor") and self.clamp_factor is not None:
                        clamp_data = input.abs().mean() * self.clamp_factor
                    elif hasattr(self, "clamp_value") and self.clamp_value is not None:
                        clamp_data = self.clamp_value
                    else:
                        clamp_data = None
                    input.data = input if clamp_data is None else torch.clamp(input, min = -clamp_data, max = clamp_data)
                self.is_init_mom_clamp_weight.fill_(True)

            if QUANT_CONFIGS.quant_method == FakeQuantMethod.CUDA:
                fake_input = self.fake_quant_cuda(input, scale)
            elif QUANT_CONFIGS.quant_method == FakeQuantMethod.CUDA_GS:
                fake_input = self.fake_quant_cuda_with_grad_scale(input, scale)
            else:
                fake_input = self.fake_quant_native(input, scale)
        return fake_input
    
    def state_dict(self, *args, **kwargs):
        with torch.no_grad():
            if self.qat_method == QatMethod.MOM and self.running_data != 0.0:
                learning_data = self.data_bits - 1 - self.running_data.abs().max().log2()
                self.scale.fill_(float((2 ** (self.quant_round(learning_data, self.round_mode)))))
        return super().state_dict(*args, **kwargs)
    
    @property
    def quant_min(self):
        quant_min = - 2 ** (self.data_bits - 1)
        return quant_min
    
    @property
    def quant_max(self):
        quant_max = 2 ** (self.data_bits - 1) - 1
        return quant_max


class AQuantizer(Quantizer):
    def __init__(self, quantizer_cfg: Optional[Dict[str, Any]] = None, constrain: Optional[Dict[str, Any]] = None):
        super().__init__(quantizer_cfg, constrain)

        self.data_bits = quantizer_cfg.get('activate_bits', 8)
        self.quant_strategy = quantizer_cfg.get('a_strategy', QuantStrategy.RANGE_MEAN)
        self.activation_type = quantizer_cfg.get('activation_type', None)
        self.is_bias_quantizer = False
        self.calibrate_name = quantizer_cfg.get('a_calibrate_name', "top_10")
        # self.quant_min = - 2 ** (self.data_bits - 1)
        # self.quant_max = 2 ** (self.data_bits - 1) - 1

        self.clamp_value = None if constrain is None else constrain.get('clamp_activation_value', None)
        
        self.round_mode = quantizer_cfg.get('round_mode', QuantMode.floor_add)

class WQuantizer(Quantizer):
    def __init__(self, quantizer_cfg: Optional[Dict[str, Any]] = None, constrain: Optional[Dict[str, Any]] = None):
        super().__init__(quantizer_cfg, constrain)

        self.data_bits = quantizer_cfg.get('weight_bits', 8)
        self.quant_strategy = quantizer_cfg.get('w_strategy', QuantStrategy.RANGE_MEAN)
        self.is_perchannel = quantizer_cfg.get('is_perchannel', False)
        self.is_bias_quantizer = False
        self.calibrate_name = quantizer_cfg.get('w_calibrate_name', "abs_max")
        # self.quant_min = - 2 ** (self.data_bits - 1)
        # self.quant_max = 2 ** (self.data_bits - 1) - 1

        self.clamp_value = None if constrain is None else constrain.get('clamp_weight_value', None)
        self.clamp_factor = None if constrain is None else constrain.get('clamp_factor_value', None)

        self.register_buffer("is_init_mom_clamp_weight", torch.tensor(False, dtype=bool))
        self.round_mode = QuantMode.round

class BQuantizer(Quantizer):
    def __init__(self, quantizer_cfg: Optional[Dict[str, Any]] = None, constrain: Optional[Dict[str, Any]] = None):
        super().__init__(quantizer_cfg, constrain)

        self.data_bits = quantizer_cfg.get('bias_bits', 32)
        self.quant_strategy = quantizer_cfg.get('w_strategy', QuantStrategy.RANGE_MEAN)
        self.is_bias_quantizer = True
        self.calibrate_name = quantizer_cfg.get('w_calibrate_name', "abs_max")
        # self.quant_min = - 2 ** (self.data_bits - 1)
        # self.quant_max = 2 ** (self.data_bits - 1) - 1

        self.clamp_value = None if constrain is None else constrain.get('clamp_bias_value', None)
        self.round_mode = QuantMode.round

