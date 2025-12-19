import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from typing import Optional, Union, Dict, Any
import copy

from ....constrain.SparifyFFN import SparifyFFN, GetSparifyMask
from ...quantizer import WQuantizer, AQuantizer, BQuantizer
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor
from ....config import QUANT_CONFIGS
from ....utils import _single, _pair, _triple, QatMethod

@register_qmodule(SparifyFFN)
class QSparifyFFN(QModuleMixin, SparifyFFN):
    @classmethod
    def qcreate(
        cls,
        module,
        activations_cfg: Optional[Dict[str, Any]] = None,
        weights_cfg: Optional[Dict[str, Any]] = None,
        bias_cfg: Optional[Dict[str, Any]] = None,
        constrain: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        sp_module = cls(
            module.input_size,
            module.output_size,
            module.bias is not None,
            8,
            None,
            None,
            3,
            
            dtype=module.weight_fc1.dtype,
            device=device,
            activations_cfg=activations_cfg,
            weights_cfg=None, # 仅打开输入输出的量化
            bias_cfg=None,
            constrain = constrain, 
        )
        temp_constrain = copy.deepcopy(constrain)
        temp_constrain['clamp_activation_value'] = None
        temp_constrain['clamp_factor_value'] = 3

        sp_module.register_module("weight_fc1_quantizer", WQuantizer(weights_cfg, temp_constrain))
        sp_module.register_module("weight_fc2_quantizer", WQuantizer(weights_cfg, temp_constrain))
        temp_constrain['clamp_factor_value'] = 7
        sp_module.register_module("weight_mask_quantizer", WQuantizer(weights_cfg, temp_constrain))

        sp_module.register_module("bias_fc1_quantizer", BQuantizer(bias_cfg, None))
        sp_module.register_module("bias_fc2_quantizer", BQuantizer(bias_cfg, None))
        sp_module.register_module("bias_mask_quantizer", BQuantizer(bias_cfg, None))

        sp_module.register_module("outfc1_quantizer", AQuantizer(activations_cfg, None))
        sp_module.register_module("outmask_quantizer", AQuantizer(activations_cfg, None))

        sp_module.weight_fc1 = module.weight_fc1
        sp_module.weight_fc2 = module.weight_fc2
        sp_module.weight_mask = module.weight_mask

        sp_module.bias_fc1 = module.bias_fc1
        sp_module.bias_fc2 = module.bias_fc2
        sp_module.bias_mask = module.bias_mask

        del temp_constrain

        return sp_module
    
    @property
    def qweight_fc1(self):  
        fake_weight = self.weight_fc1_quantizer(self.weight_fc1)
        return fake_weight

    @property
    def qweight_fc2(self):
        fake_weight = self.weight_fc2_quantizer(self.weight_fc2)
        return fake_weight
    
    @property
    def qweight_mask(self):    
        fake_weight = self.weight_mask_quantizer(self.weight_mask)
        return fake_weight

    @property
    def qbias_fc1(self):
        if self.bias_fc1_quantizer is None:
            return self.bias_fc1
        fake_bias = self.bias_fc1_quantizer(self.bias_fc1, self.weight_fc1_quantizer.scale * self.input_quantizer.scale)
        return fake_bias
    
    @property
    def qbias_fc2(self):
        if self.bias_fc2_quantizer is None:
            return self.bias_fc2
        fake_bias = self.bias_fc2_quantizer(self.bias_fc2, self.weight_fc2_quantizer.scale * self.outfc1_quantizer.scale)
        return fake_bias

    @property
    def qbias_mask(self):
        if self.bias_mask_quantizer is None:
            return self.bias_mask
        fake_bias = self.bias_mask_quantizer(self.bias_mask, self.weight_mask_quantizer.scale * self.input_quantizer.scale)
        return fake_bias
    
    def quantize_outL(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, QTensor):
            fake_input = from_qtensor_to_tensor(input)
            self.outfc1_quantizer.scale.fill_(input.scale.detach())
            self.outfc1_quantizer.data_bits = input.data_bits
        else:
            fake_input = self.outfc1_quantizer(input) # 前向过程中会更新input_quantizer的scale
        return fake_input
    
    def quantize_outM(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, QTensor):
            fake_input = from_qtensor_to_tensor(input)
            self.outmask_quantizer.scale.fill_(input.scale.detach())
            self.outmask_quantizer.data_bits = input.data_bits
        else:
            fake_input = self.outmask_quantizer(input) # 前向过程中会更新input_quantizer的scale
        return fake_input

    def qforward(self, input):
        outL0 = F.linear(input, self.qweight_fc1, self.qbias_fc1)
        outL = self.quantize_outL(outL0)

        outM = F.linear(input, self.qweight_mask, self.qbias_mask)
        outM = self.quantize_outM(outM)
        outM = torch.softmax(outM, dim=-1)

        mask = GetSparifyMask.apply(outM, self.ratio)
        outM2 = mask.repeat_interleave(self.repeat_num, dim=-1)
        out1 = outL * outM2 # 浮点Tensor类型计算,相当于mask操作，不能走进qmul

        out2 = F.relu(out1)
        out = F.linear(out2, self.qweight_fc2, self.qbias_fc2)

        return out

