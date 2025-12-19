
from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any

from ..qconfig import _QMODULE_TABLE
from ...quantizer import WQuantizer, AQuantizer, BQuantizer
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor
from ....config import QUANT_CONFIGS
from ....onnx import generate_onnx_qparam_dict, QCustomOpSymbolic, QCustomRNNSymbolic

__all__ = ["QModuleMixin"]

class QModuleMixin(ABC):
    def __init__(
        self,
        *args, # 原始torch.module初始化所需的参数
        device: Optional[torch.device] = None,
        weights_cfg: Optional[Dict[str, Any]] = None, # 量化策略相关的参数
        activations_cfg: Optional[Dict[str, Any]] = None,
        bias_cfg: Optional[Dict[str, Any]] = None,
        constrain: Optional[Dict[str, Any]] = None,
        open_ihook: Optional[bool] = True,
        open_ohook: Optional[bool] = True,
        **kwargs,
    ):
        mro = self.__class__.__mro__
        if torch.nn.Module not in mro: # 必须和torch.nn.module一起被Qlinear类继承
            raise TypeError("Quantized modules must inherit from a torch.nn.Module class")
        if mro.index(__class__) > mro.index(torch.nn.Module): # 继承时此类必须写在前边，torch.nn.module才能被初始化
            raise TypeError(
                "QModuleMixin must be placed before any torch.nn.Module class in quantized module inheritance."
            )
        # This will setup the torch.nn.Module
        super().__init__(*args, **kwargs) # 原始linear或conv等线性module的初始化

        if weights_cfg is not None:
            self.weight_quantizer = WQuantizer(weights_cfg, constrain)
        if bias_cfg is not None:
            self.bias_quantizer = BQuantizer(bias_cfg, constrain)

        self._quantize_hooks = {}
        if open_ihook:
            self.input_quantizer  = AQuantizer(activations_cfg, None)
            self._quantize_hooks["input"] = self.register_forward_pre_hook(self.quantize_input)
        if open_ohook:
            self.output_quantizer = AQuantizer(activations_cfg, constrain)
            self._quantize_hooks["output"] = self.register_forward_hook(self.quantize_output)

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        activations_cfg: Optional[Union[str]] = None,
        *args,
        **kwargs
    ):
        # Create the quantized module on the meta device to prevent weights intialization
        weights_cfg = kwargs.get('weights_cfg', None)
        bias_cfg = kwargs.get('bias_cfg', None)
        constrain = kwargs.get('constrain', None)
        device = QUANT_CONFIGS.device
        qmodule = cls.qcreate(module, activations_cfg, weights_cfg, bias_cfg, constrain, device=device)
        if qmodule is None:
            return None

        if hasattr(module, 'weight'):
            with torch.no_grad():
                qmodule.weight = module.weight
                if hasattr(module, "bias") and module.bias is not None:
                    qmodule.bias = module.bias

        return qmodule.to(device)

    @classmethod
    def qcreate(
        cls,
        module: torch.nn.Module,
        activations_cfg: Optional[Union[str]] = None,
        weight_cfg: Optional[Union[str]] = None,
        bias_cfg: Optional[Union[str]] = None,
        constrain: Optional[Union[str]] = None,
        device: Optional[torch.device] = None,
    ):
        raise NotImplementedError

    @property
    def qweight(self):
        if not hasattr(self, "weight_quantizer"):
            return self.weight
        fake_weight = self.weight_quantizer(self.weight)
        return fake_weight

    @property
    def qbias(self):
        if not hasattr(self, "bias_quantizer") or self.bias is None:
            return self.bias
        fake_bias = self.bias_quantizer(self.bias, self.weight_quantizer.scale * self.input_quantizer.scale)
        return fake_bias

    def qforward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            weight = None if not hasattr(self, "weight_quantizer") else self.weight
            bias = None if not hasattr(self, "bias_quantizer") else self.bias
            qparam_dict = generate_onnx_qparam_dict(self, False)
            if 'GRU' in self._get_name() or 'LSTM' in self._get_name():
                self.input_quantizer.is_qtensor = True if isinstance(input, QTensor) else False
                if self.bidirectional:
                    out, _ = QCustomRNNSymbolic.apply(input, self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0, \
                                                    self.weight_ih_l0_reverse, self.weight_hh_l0_reverse, self.bias_ih_l0_reverse, self.bias_hh_l0_reverse, qparam_dict, self.input_quantizer.is_qtensor)
                else:
                    out, _ =  QCustomRNNSymbolic.apply(input, self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0, None, None, None, None, qparam_dict, self.input_quantizer.is_qtensor)
                return from_tensor_to_qtensor(out, self.output_quantizer.scale, self.output_quantizer.data_bits), _
            elif 'Embedding' in self._get_name():
                out = QCustomOpSymbolic.apply(input, weight, bias, qparam_dict, True)
                return from_tensor_to_qtensor(out, self.weight_quantizer.scale, self.weight_quantizer.data_bits)
            return QCustomOpSymbolic.apply(input, weight, bias, qparam_dict, self.input_quantizer.is_qtensor)
        else:
            return self.qforward(input, *args, **kwargs)
        
    def quantize_input(self, module: torch.nn.Module, input: torch.Tensor) -> torch.Tensor:
        input = input[0]

        if isinstance(input, QTensor):
            fake_input = from_qtensor_to_tensor(input)
            self.input_quantizer.is_qtensor = True
            self.input_quantizer.scale.fill_(input.scale.detach())
            self.input_quantizer.data_bits = input.data_bits
        else:
            fake_input = self.input_quantizer(input) # 前向过程中会更新input_quantizer的scale
        return fake_input

    def quantize_output(
        self,
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        fake_out = self.output_quantizer(output)
        return from_tensor_to_qtensor(fake_out, self.output_quantizer.scale, self.output_quantizer.data_bits)

    def extra_repr(self):
        s = ''
        extra_s = ''
        if 'Conv1d' in self._get_name() or 'ConvBN1d' in self._get_name():
            s = nn.Conv1d.extra_repr(self)
        elif 'Conv2d' in self._get_name() or 'ConvBN2d' in self._get_name():
            s = nn.Conv2d.extra_repr(self)
        elif 'MaxPool1d' in self._get_name():
            s = nn.MaxPool1d.extra_repr(self)
        elif 'MaxPool2d' in self._get_name():
            s = nn.MaxPool2d.extra_repr(self)
        elif 'AvgPool1d' in self._get_name():
            s = nn.AvgPool1d.extra_repr(self)
        elif 'AvgPool2d' in self._get_name():
            s = nn.AvgPool2d.extra_repr(self)
        elif 'ConvTranspose1d' in self._get_name():
            s = nn.ConvTranspose1d.extra_repr(self)
        elif 'ConvTranspose2d' in self._get_name():
            s = nn.ConvTranspose2d.extra_repr(self)
        elif 'BatchNorm1d' in self._get_name():
            s = nn.BatchNorm1d.extra_repr(self)
        elif 'BatchNorm2d' in self._get_name():
            s = nn.BatchNorm2d.extra_repr(self)
        elif 'Linear' in self._get_name():
            s = nn.Linear.extra_repr(self)
        elif 'Relu' in self._get_name():
            s = nn.ReLU.extra_repr(self)
        elif 'GLU' in self._get_name():
            s = nn.GLU.extra_repr(self)
        elif 'LayerNorm' in self._get_name():
            s = nn.LayerNorm.extra_repr(self)
        elif 'GRU' in self._get_name():
            s = nn.GRU.extra_repr(self)
        elif 'LSTM' in self._get_name():
            s = nn.LSTM.extra_repr(self)
        elif 'Embedding' in self._get_name():
            s = nn.Embedding.extra_repr(self)
            
        # extra_s = ', clamp_data:{}, clamp_weight:{}, clamp_bias:{}, clamp_factor:{}, activation_type:{}'.format(self.output_quantizer.clamp_value, self.weight_quantizer.clamp_value, self.bias_quantizer.clamp_value, self.weight_quantizer.clamp_factor, self.activation_type.name)
        if hasattr(self, "input_quantizer"):
            extra_s += ', data_bits:{}'.format(self.input_quantizer.data_bits)
        if hasattr(self, "output_quantizer"):
            extra_s += ', o_bits:{}'.format(self.output_quantizer.data_bits)
        if hasattr(self, "weight_quantizer"):
            extra_s += ', weight_bits:{}'.format(self.weight_quantizer.data_bits)
        if hasattr(self, "bias_quantizer"):
            extra_s += ', bias_bits:{}'.format(self.bias_quantizer.data_bits)
        extra_s += ', mode:{}'.format(QUANT_CONFIGS.quant_info.round_mode)
        extra_s += ', platform:{}'.format(QUANT_CONFIGS.platform.name)
        return s + extra_s

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        main_str = self._get_name() + '('
        if len(extra_lines) > 0:
            main_str += extra_lines[0]
        main_str += ')'
        return main_str

