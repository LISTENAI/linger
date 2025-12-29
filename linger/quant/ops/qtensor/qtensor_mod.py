
import torch
from typing import Dict, Any, Optional

from ...quantizer import AQuantizer
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor
from ....config import QUANT_CONFIGS

__all__ = ["QModuleTensor"]

class QModuleTensor(torch.nn.Module):
    def __init__(self, activate_config: Optional[Dict[str, Any]] = None, num_input: int = 2, dim: int = -1, is_cat = False):
        super(QModuleTensor, self).__init__()

        self.num_input = num_input
        self.dim = dim
        self.is_cat = is_cat
        self.input_quantizer = torch.nn.ModuleList()
        for i in range(num_input):
            self.input_quantizer.append(AQuantizer(activate_config))
        self.output_quantizer = AQuantizer(activate_config)

        self._quantize_hooks = {}
        if (is_cat == True):
            self._quantize_hooks["input"] = self.register_forward_pre_hook(self.quantize_cat_input)
        else:
            self._quantize_hooks["input"] = self.register_forward_pre_hook(self.quantize_input)
        self._quantize_hooks["output"] = self.register_forward_hook(self.quantize_output)


    def quantize_input(self, module: torch.nn.Module, input) -> torch.Tensor:
        device = QUANT_CONFIGS.device
        
        # 创建处理后的输入列表
        processed_inputs = []
        
        for i in range(len(input)):
            if i < self.num_input:
                current_input = input[i]
                
                # 标准化输入 - 创建新的tensor而不是修改原元组
                if not isinstance(current_input, torch.Tensor) and not isinstance(current_input, QTensor):
                    current_input = torch.tensor(current_input, dtype=torch.float32, device=device)
                
                # 量化处理
                if isinstance(current_input, QTensor):
                    tmp_input = from_qtensor_to_tensor(current_input)
                    self.input_quantizer[i].scale.fill_(current_input.scale.detach())
                    self.input_quantizer[i].data_bits = current_input.data_bits
                else:
                    tmp_input = self.input_quantizer[i](current_input)
                processed_inputs.append(tmp_input)
            else:
                processed_inputs.append(input[i])
        
        return tuple(processed_inputs)
        
    
    def quantize_cat_input(self, module: torch.nn.Module, input_list: list) -> torch.Tensor:
        device = QUANT_CONFIGS.device

        if not input_list:
            return torch.tensor([], device=device)
        
        # 创建处理后的输入列表
        processed_inputs = []
        
        for i in range(len(input_list[0])):
            current_input = input_list[0][i]
            
            # 标准化输入 - 创建新的tensor而不是修改原元组
            if not isinstance(current_input, torch.Tensor) and not isinstance(current_input, QTensor):
                current_input = torch.tensor(current_input, dtype=torch.float32, device=device)
            
            if isinstance(current_input, QTensor):
                tmp_input = from_qtensor_to_tensor(current_input)
                self.input_quantizer[i].scale.fill_(current_input.scale.detach())
                self.input_quantizer[i].data_bits = current_input.data_bits
            else:
                tmp_input = self.input_quantizer[i](current_input)
            processed_inputs.append(tmp_input)
        
        return tuple([processed_inputs, input_list[1]])
    
    def quantize_output(
        self,
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        fake_output = self.output_quantizer(output)
        return from_tensor_to_qtensor(fake_output, self.output_quantizer.scale, self.output_quantizer.data_bits)
    
    @classmethod
    def qcreate(
        cls,
        module: torch.nn.Module,
        activate_cfg: Optional[Dict[str, Any]] = None,
        num_input: int = 2,
        dim: int = -1
    ):
        raise NotImplementedError
    
    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        activate_cfg: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ):
        # Create the quantized module on the meta device to prevent weights intialization
        num_input = kwargs.get('num_input', 1)
        dim = kwargs.get('dim', 1)
        qmodule = cls.qcreate(module, activate_cfg, num_input, dim)
        if qmodule is None:
            return None
        
        device = QUANT_CONFIGS.device

        return qmodule.to(device)
    
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self):
        extra_s = ''
        if self.num_input == 1:
            extra_s += 'data_bits:{}, o_bits:{}'.format(self.input_quantizer[0].data_bits,self.output_quantizer.data_bits)
        else:
            extra_s += 'data_x_bits:{}, data_y_bits:{}, o_bits:{}'.format(self.input_quantizer[0].data_bits,self.input_quantizer[1].data_bits,self.output_quantizer.data_bits)
        extra_s += ', mode:{}'.format(self.output_quantizer.round_mode)
        extra_s += ', platform:{}'.format(QUANT_CONFIGS.platform)

        return extra_s

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        main_str = self._get_name() + '('
        main_str += extra_lines[0]
        main_str += ')'
        return main_str

