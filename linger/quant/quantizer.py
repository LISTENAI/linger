import math
import torch
from typing import Optional, Union, Dict, Any

from .qtensor import QTensor
from .calibrate_funs import get_calibrate_function, get_finalize_function
from .mom_fake_quant import NativeMOMFakeQuant
from ..config  import QUANT_CONFIGS
from ..utils import QuantMode, QuantStrategy, ActivationType, FakeQuantMethod, QatMethod, PlatForm
import lingerext


HIGH_PRECISION_BITS = 16
HIGH_PRECISION_SCALE_EXPONENT_MAX = 20
HIGH_PRECISION_SCALE_MAX = float(2 ** HIGH_PRECISION_SCALE_EXPONENT_MAX)


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
    def forward(ctx, input, data_bits, learning_data, scale_min, quant_min, quant_max, round_mode):
        out, mask, scale_tmp = lingerext.fake_quant(
            input, data_bits, float(learning_data), float(scale_min), quant_min, quant_max, round_mode
        )
        ctx.save_for_backward(mask.bool())
        return out, scale_tmp

    @staticmethod
    def backward(ctx, gradOutput1, gradOutput2):
        mask = ctx.saved_tensors
        return gradOutput1.masked_fill(mask[0], 0.0), None, None, None, None, None, None

class BIASQUANT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, data_bits, scale_bias, scale_min, quant_min, quant_max, round_mode):
        out,mask = lingerext.bias_quant(
            input, data_bits, float(scale_bias), float(scale_min), quant_min, quant_max, round_mode
        )
        ctx.save_for_backward(mask.bool())
        return out
    @staticmethod
    def backward(ctx, gradOutput):
        mask = ctx.saved_tensors
        return gradOutput.masked_fill(mask[0], 0.0), None, None, None, None, None, None

class FAKEQUANT_WITH_GRAD_SACLE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, data_bits, learning_data, scale_min, quant_min, quant_max, round_mode):
        out, mask, learning_data_coff_back, scale_tmp = lingerext.fake_quant_with_grad_scale(
            input, data_bits, float(learning_data), float(scale_min), quant_min, quant_max, round_mode
        )
        saved_tensors = [mask.bool(), learning_data_coff_back]
        ctx.save_for_backward(*saved_tensors)
        # The extension reduces the scale gradient to a scalar, while TQT's
        # learnable parameter is registered with shape ``[1]``.  Keep the
        # input shape so autograd receives a gradient compatible with the
        # original parameter on both CPU stubs and CUDA extensions.
        ctx.learning_data_shape = learning_data.shape if torch.is_tensor(learning_data) else None
        return out, scale_tmp

    @staticmethod
    def backward(ctx, gradOutput1, gradOutput2):
        mask, learning_data_coff_back = ctx.saved_tensors
        grad_learning_data = (gradOutput1 * learning_data_coff_back).sum()
        if ctx.learning_data_shape is not None:
            grad_learning_data = grad_learning_data.reshape(ctx.learning_data_shape)
        return gradOutput1.masked_fill(mask, 0.0), None, grad_learning_data, None, None, None, None

class BIASQUANT_WITH_GRAD_SACLE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, data_bits, scale_bias, scale_min, quant_min, quant_max, round_mode):
        out,mask,scale_coff_back = lingerext.bias_quant_with_grad_scale(
            input, data_bits, float(scale_bias), float(scale_min), quant_min, quant_max, round_mode
        )
        saved_tensors = [mask.bool(), scale_coff_back]
        ctx.save_for_backward(*saved_tensors)
        ctx.scale_bias_shape = scale_bias.shape if torch.is_tensor(scale_bias) else None
        return out
    @staticmethod
    def backward(ctx, gradOutput):
        mask,  scale_coff_back = ctx.saved_tensors
        grad_scale = (gradOutput * scale_coff_back).sum()
        if ctx.scale_bias_shape is not None:
            grad_scale = grad_scale.reshape(ctx.scale_bias_shape)
        return gradOutput.masked_fill(mask, 0.0), None, grad_scale, None, None, None, None


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
        elif self.qat_method == QatMethod.MOM:
            # MOM策略使用，running_data统计input.abs().max()获得
            self.register_buffer("running_data", torch.tensor(0.0))
            self.register_buffer("is_init", torch.tensor(False, dtype=bool))
            self.momentum = 0.1
        else:
            raise ValueError("Only TQT and MOM strategies are supported! ")
        if QUANT_CONFIGS.calibration:
            self.register_buffer("is_calibrate", torch.tensor(False, dtype=bool))
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

    def finalize_calibration(self):
        """
        Finalize calibration after all data has been accumulated.
        Called when exiting the calibration context.
        """
        if not hasattr(self, 'is_calibrate') or not self.is_calibrate.item():
            return  # 没有进行校准

        # 获取对应的finalize函数
        finalize_function = get_finalize_function(self.calibrate_name)
        if finalize_function is not None:
            finalize_function(self, self.data_bits)

        # 设置is_calibrate为True表示校准完成
        self.is_calibrate.fill_(True)
    
    def copy_from_quantizer(self, quantizer):
        if self.qat_method == QatMethod.MOM and quantizer.qat_method == QatMethod.MOM:
            self.running_data = quantizer.running_data.clone()
            if not self.is_init:
                self.is_init.fill_(quantizer.is_init)
        self.scale.copy_(self._limit_scale(quantizer.scale.detach()).to(self.scale.device))
        if self.qat_method == QatMethod.TQT and quantizer.qat_method == QatMethod.TQT:
            self.learning_data = quantizer.learning_data
    
    def quant_round(self, x, mode):
        # High precision deployment uses FP64 for the scale exponent and
        # rounding decision.  Keep this conversion here so MOM, TQT and
        # calibration share exactly the same exponent semantics.
        if self._is_high_precision() and torch.is_tensor(x) and x.is_floating_point():
            x = x.to(torch.float64)
        if mode == QuantMode.floor_add: 
            out = ((x + 0.5).floor() - x).detach() + x
        elif mode == QuantMode.floor:
            out = (x.floor() - x).detach() + x
        else:
            out = (x.round() - x).detach() + x
        return out

    def _is_high_precision(self):
        return self.data_bits >= HIGH_PRECISION_BITS

    def _limit_scale(self, scale):
        max_scale = None
        if self._is_high_precision() and not self.is_bias_quantizer:
            max_scale = HIGH_PRECISION_SCALE_MAX

        if torch.is_tensor(scale):
            if self._is_high_precision() and scale.is_floating_point():
                scale = scale.to(torch.float64)
            return scale.clamp(min=1, max=max_scale)

        scale = max(float(scale), 1.0)
        if max_scale is not None:
            scale = min(scale, max_scale)
        return scale

    def _limit_scale_exponent(self, exponent):
        if self._is_high_precision() and torch.is_tensor(exponent) and exponent.is_floating_point():
            exponent = exponent.to(torch.float64)
        max_exponent = (
            HIGH_PRECISION_SCALE_EXPONENT_MAX
            if self._is_high_precision() and not self.is_bias_quantizer
            else None
        )
        return exponent.clamp(min=0, max=max_exponent)

    def _limit_factor_for_extension(self, factor):
        if not self._is_high_precision() or self.is_bias_quantizer:
            return factor
        if torch.is_tensor(factor) and factor.is_floating_point():
            factor = factor.to(torch.float64)
        min_factor = self.data_bits - 1 - HIGH_PRECISION_SCALE_EXPONENT_MAX
        return factor.clamp(min=min_factor)

    def _to_transport_dtype(self, tensor):
        """Return the dtype crossing the quantizer/QTensor interface.

        High precision arithmetic is performed in FP64, while the chip
        transport representation for non-bias fake-quantized tensors remains
        FP32.  Bias keeps its existing FP64 path.
        """
        if self._is_high_precision() and not self.is_bias_quantizer:
            return tensor.to(torch.float32)
        return tensor

    def _fake_quant_with_scale(self, input, scale, optimize_mom=False):
        if self.is_bias_quantizer:
            compute_input = input.to(torch.float64)
            if isinstance(scale, torch.Tensor):
                compute_scale = scale.to(dtype=torch.float64, device=input.device)
            else:
                compute_scale = float(scale)
        elif self._is_high_precision():
            compute_input = input.to(torch.float64)
            if isinstance(scale, torch.Tensor):
                compute_scale = scale.to(dtype=torch.float64, device=input.device)
            else:
                compute_scale = float(scale)
        else:
            compute_input = input
            compute_scale = scale

        if optimize_mom and not (torch.is_tensor(compute_scale) and compute_scale.requires_grad):
            fake_input = NativeMOMFakeQuant.apply(
                compute_input, compute_scale, self.quant_min, self.quant_max, self.round_mode
            )
        else:
            x_s = compute_input * compute_scale
            x_int = self.quant_round(x_s, self.round_mode)
            x_int = x_int.clamp(self.quant_min, self.quant_max)
            fake_input = x_int / compute_scale
        if not self.is_bias_quantizer and self.data_bits <= 8:
            return fake_input.to(torch.float32)
        return fake_input
       
    def fake_quant_native(self, input, scale_bias = None):
        optimize_mom = (
            self.qat_method == QatMethod.MOM
            and QUANT_CONFIGS.mom_native_optimized
            # The original first observation adopts the input's dtype/device.
            # Use it for unusual mixed-dtype/device initial states as well.
            and (
                scale_bias is not None
                or (
                    input.dtype == self.running_data.dtype
                    and input.device == self.running_data.device
                    and input.device == self.is_init.device
                )
            )
        )
        if scale_bias is None:
            if self.qat_method == QatMethod.TQT:
                learning_data = self.learning_data
                if self._is_high_precision():
                    learning_data = learning_data.to(torch.float64)
                learning_data_temp = self.data_bits - 1 - learning_data
            elif self.qat_method == QatMethod.MOM:
                if optimize_mom:
                    # Statistics have no gradients. Cache initialization state
                    # until its buffer changes to avoid a scalar GPU read.
                    with torch.no_grad():
                        if hasattr(self, "activation_type") and self.activation_type == ActivationType.Relu:
                            abs_max = input.max()
                        else:
                            abs_max = input.abs().max()
                        abs_max = abs_max.clamp(min=1e-6)
                        if self._mom_buffer_initialized('is_init'):
                            self.running_data = (1-self.momentum) * self.running_data + self.momentum * abs_max
                        else:
                            self.running_data = abs_max
                            self.is_init.fill_(True)
                            self._cache_mom_buffer_initialized('is_init', True)
                else:
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
            learning_data = self._limit_scale_exponent(
                self.quant_round(learning_data_temp, QuantMode.round)
            )
            scale = torch.exp2(learning_data)
        else:
            scale = self._limit_scale(scale_bias)

        if hasattr(self, "min_scale"):
            if (
                optimize_mom
                and torch.is_tensor(scale)
                and torch.is_tensor(self.min_scale)
                and scale.dtype == self.min_scale.dtype
                and scale.device == self.min_scale.device
                and scale.shape == self.min_scale.shape
                and not self.min_scale.requires_grad
            ):
                # Preserve the original conditional's NaN behavior as well as
                # its dtype; torch.minimum would propagate a NaN bound.
                scale = torch.where(scale > self.min_scale, self.min_scale, scale)
            elif scale > self.min_scale:
                scale = self.min_scale
                if not optimize_mom:
                    self.scale.fill_(float(scale))
        
        fake_input = self._fake_quant_with_scale(input, scale, optimize_mom=optimize_mom and self.training)

        if optimize_mom and torch.is_tensor(scale):
            # copy_ casts exactly as fill_(float(scale)) does, without copying a
            # device scalar to the CPU. The computed scale remains independent
            # of this mutable buffer, which is important for backward.
            self.scale.copy_(scale.detach().reshape(()))
        else:
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
            scale_top = torch.exp2(
                self._limit_scale_exponent(
                    self.quant_round(torch.log2(scale_top), QuantMode.ceil)
                )
            )
            inputs_q_top = self._fake_quant_with_scale(input, scale_top)
            scale_bottom = self.quant_max / self.running_data
            scale_bottom = torch.exp2(
                self._limit_scale_exponent(
                    self.quant_round(torch.log2(scale_bottom), QuantMode.floor)
                )
            )
            inputs_q_bottom = self._fake_quant_with_scale(input, scale_bottom)
            score_top = lp_loss(input, inputs_q_top, p=2.4, reduction='all')
            score_bottom = lp_loss(input, inputs_q_bottom, p=2.4, reduction='all')

            scale = (scale_top if score_top < score_bottom else scale_bottom).float()
            
        else:
            scale = self._limit_scale(scale_bias)

        scale = self._limit_scale(scale)

        fake_input = self._fake_quant_with_scale(input, scale)

        self.scale.fill_(float(scale))

        return fake_input

    def fake_quant_cuda(self, input, scale_bias = None):
        # Keep MOM statistics in the original input dtype, just as the native
        # path does.  High precision conversion is applied only to the fake
        # quant arithmetic below; converting before abs_max would change the
        # running_data dtype and can move an exponent tie across a rounding
        # boundary for FP16/FP32 inputs.
        stats_input = input
        if self._is_high_precision() and torch.is_tensor(input) and input.is_floating_point():
            input = input.to(torch.float64)
        if hasattr(self, "min_scale"): # 只有add算子的outputquantizer有min_scale，其余情况min_scale和正常的scale相同
            min_scale = self.min_scale
        else:
            min_scale = float("inf")

        if scale_bias is None:
            if self.qat_method == QatMethod.TQT:
                factor = self._limit_factor_for_extension(self.learning_data)
                out, scale_tmp = FAKEQUANT.apply(input, self.data_bits, factor, min_scale, self.quant_min, self.quant_max, self.round_mode.value)
            elif self.qat_method == QatMethod.MOM:
                if hasattr(self, "activation_type") and self.activation_type == ActivationType.Relu:
                    abs_max = stats_input.max()
                else:
                    abs_max = stats_input.abs().max()
                abs_max = abs_max.clamp(min=1e-6).detach()
                if not self.is_init:
                    self.running_data = abs_max
                    self.is_init.fill_(True)
                else:
                    self.running_data = (1-self.momentum) * self.running_data + self.momentum * abs_max
                factor = self._limit_factor_for_extension(self.running_data.log2())
                out, scale_tmp = FAKEQUANT.apply(input, self.data_bits, factor, min_scale, self.quant_min, self.quant_max, self.round_mode.value)
        else:
            scale_bias = self._limit_scale(scale_bias)
            out = BIASQUANT.apply(input, self.data_bits, scale_bias, min_scale, self.quant_min, self.quant_max, self.round_mode.value)
            scale_tmp = scale_bias.detach()
        self.scale.fill_(scale_tmp) # MOM也必须要给scale初始化，便于后续(如bias)调用当前模块scale时保持正确
        return out

    def fake_quant_cuda_with_grad_scale(self, input, scale_bias = None):
        # See fake_quant_cuda: preserve the original MOM statistics dtype and
        # convert only the quantization arithmetic to FP64.
        stats_input = input
        if self._is_high_precision() and torch.is_tensor(input) and input.is_floating_point():
            input = input.to(torch.float64)
        if hasattr(self, "min_scale"): # 只有add算子的outputquantizer有min_scale，其余情况min_scale和正常的scale相同
            min_scale = self.min_scale
        else:
            min_scale = float("inf")

        if scale_bias is None:
            if self.qat_method == QatMethod.TQT:
                factor = self._limit_factor_for_extension(self.learning_data)
                out, scale_tmp = FAKEQUANT_WITH_GRAD_SACLE.apply(input, self.data_bits, factor, min_scale, self.quant_min, self.quant_max, self.round_mode.value)
            elif self.qat_method == QatMethod.MOM:
                if hasattr(self, "activation_type") and self.activation_type == ActivationType.Relu:
                    abs_max = stats_input.max()
                else:
                    abs_max = stats_input.abs().max()
                abs_max = abs_max.clamp(min=1e-6).detach()
                if not self.is_init:
                    self.running_data = abs_max
                    self.is_init.fill_(True)
                else:
                    self.running_data = (1-self.momentum) * self.running_data + self.momentum * abs_max
                factor = self._limit_factor_for_extension(self.running_data.log2())
                out, scale_tmp = FAKEQUANT_WITH_GRAD_SACLE.apply(input, self.data_bits, factor, min_scale, self.quant_min, self.quant_max, self.round_mode.value)
        else:
            scale_bias = self._limit_scale(scale_bias)
            out = BIASQUANT_WITH_GRAD_SACLE.apply(input, self.data_bits, scale_bias, min_scale, self.quant_min, self.quant_max, self.round_mode.value)
            scale_tmp = scale_bias.detach()
        self.scale.fill_(scale_tmp) # MOM也必须要给scale初始化，便于后续(如bias)调用当前模块scale时保持正确
        return out

    def inference(self, input, scale=None):
        if hasattr(self, "min_scale"):
            min_scale = self.min_scale
        else:
            min_scale = float("inf")
        # 推理时固定走cuda路线，若为TQT模式，通过learning_data保存scale，若为MOM模式，通过running_data重新初始化scale
        # if scale is None:
        #     if self.qat_method == QatMethod.MOM and self.running_data != 0.0:
        #         learning_data = self.data_bits - 1 - self.running_data.abs().max().log2()
        #         self.scale.fill_(float(torch.exp2(self.quant_round(learning_data, self.round_mode).clamp(min=0, max=23))))
        if scale is None:
            scale = self._limit_scale(self.scale)
        else:
            scale = self._limit_scale(scale)
        if hasattr(self, "min_scale") and scale > self.min_scale:
            scale = self._limit_scale(self.min_scale)
        self.scale.fill_(float(scale))
        if self._is_high_precision() and torch.is_tensor(input) and input.is_floating_point():
            input = input.to(torch.float64)
        if QUANT_CONFIGS.quant_method == FakeQuantMethod.CUDA and not self.is_bias_quantizer:
            out = BIASQUANT.apply(input, self.data_bits, scale, min_scale, self.quant_min, self.quant_max, self.round_mode.value)
        else:
            out = self.fake_quant_native(input, scale)
        return out
    
    def forward(self, input, scale=None):
        if torch.onnx.is_in_onnx_export():
            return FakeQuantOnnxFunction.apply(input)
        if QUANT_CONFIGS.calibration:
            fake_input = self.init_quant_data(input)
        elif not self.training:
            fake_input = self.inference(input, scale)
        else:
            # only weight clamp
            if self.qat_method == QatMethod.MOM and hasattr(self, "is_init_mom_clamp_weight"):
                if QUANT_CONFIGS.mom_native_optimized:
                    clamp_initialized = self._mom_buffer_initialized('is_init_mom_clamp_weight')
                else:
                    clamp_initialized = not bool(self.is_init_mom_clamp_weight == False)
            else:
                clamp_initialized = True
            if not clamp_initialized:
                with torch.no_grad():
                    if hasattr(self, "clamp_factor") and self.clamp_factor is not None:
                        clamp_data = input.abs().mean() * self.clamp_factor
                    elif hasattr(self, "clamp_value") and self.clamp_value is not None:
                        clamp_data = self.clamp_value
                    else:
                        clamp_data = None
                    input.data = input if clamp_data is None else torch.clamp(input, min = -clamp_data, max = clamp_data)
                self.is_init_mom_clamp_weight.fill_(True)
                if QUANT_CONFIGS.mom_native_optimized:
                    self._cache_mom_buffer_initialized('is_init_mom_clamp_weight', True)

            if self.is_bias_quantizer:
                fake_input = self.fake_quant_native(input, scale)
            elif QUANT_CONFIGS.quant_method == FakeQuantMethod.CUDA:
                fake_input = self.fake_quant_cuda(input, scale)
            elif QUANT_CONFIGS.quant_method == FakeQuantMethod.CUDA_GS:
                fake_input = self.fake_quant_cuda_with_grad_scale(input, scale)
            else:
                fake_input = self.fake_quant_native(input, scale)
        return self._to_transport_dtype(fake_input)

    def _mom_buffer_initialized(self, name):
        """Read an initialization flag once until its tensor or value changes.

        Buffer identity and version cover device moves, state_dict loading,
        deepcopy, and explicit in-place flag resets without checkpoint changes.
        """
        flag = getattr(self, name)
        try:
            key = (id(flag), flag._version)
        except RuntimeError:
            # Tensors constructed under inference_mode have no version counter.
            return bool(flag)
        cache_name = '_mom_cached_' + name
        cached = getattr(self, cache_name, None)
        if cached is None or cached[0] != key:
            cached = (key, bool(flag))
            setattr(self, cache_name, cached)
        return cached[1]

    def _cache_mom_buffer_initialized(self, name, initialized):
        flag = getattr(self, name)
        try:
            setattr(self, '_mom_cached_' + name, ((id(flag), flag._version), initialized))
        except RuntimeError:
            pass
    
    # def state_dict(self, *args, **kwargs):
    #     with torch.no_grad():
    #         if self.qat_method == QatMethod.MOM and self.running_data != 0.0:
    #             learning_data = self.data_bits - 1 - self.running_data.abs().max().log2()
    #             self.scale.fill_(float((torch.exp2(self.quant_round(learning_data, self.round_mode).clamp(min=0, max=23)))))
    #     return super().state_dict(*args, **kwargs)
    
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
        self.calibrate_name = quantizer_cfg.get('a_calibrate_name', "abs_max")

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

        self.clamp_value = None if constrain is None else constrain.get('clamp_bias_value', None)
        self.round_mode = QuantMode.round
