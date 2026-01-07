
from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import QUANT_CONFIGS
from typing import Optional, Union, Dict, Any

__all__ = ["CModuleMixin", "register_cmodule", "constrain_module"]


_CMODULE_TABLE = {}


def register_cmodule(module_cls):
    """
    Used for registering a new constrain module.

    The CModule must implement two abstract methods:

    - qcreate: class method to instantiate a new CModule from an nn.Module, without copying its weights,
    - forward: instance method for constrain inference.

    The code to register a new module looks like:

    ```
    @register_cmodule(<base torch.nn.Module>)
    class MyCModule(CModuleMixin, <base torch.nn.Module>):
        <implementation>

        @classmethod
        def qcreate(cls,
                    module: torch.nn.Module,
                    weights: Optional[],
                    activations: Optional[] = None,
                    optimizer: Optional[Optimizer] = None):
            ...

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            ...
    ```

    """

    def wrapper(cls):
        _CMODULE_TABLE[module_cls] = cls
        return cls

    return wrapper

def constrain_module(
    module,
    constrain: Optional[Dict[str, Any]] = None,
):
    for cls in _CMODULE_TABLE:
        if isinstance(module, cls):
            qcls = _CMODULE_TABLE[cls]
            return qcls.from_module(module, constrain=constrain)
    return None

class CModuleMixin(ABC):
    def __init__(
        self,
        *args, # 原始torch.module初始化所需的参数
        device: Optional[torch.device] = None,
        constrain: Optional[Dict[str, Any]] = None, # 约束策略相关的参数
        open_ihook: Optional[bool] = True,
        open_ohook: Optional[bool] = True,
        **kwargs,
    ):
        mro = self.__class__.__mro__
        if torch.nn.Module not in mro: # 必须和torch.nn.module一起被Qlinear类继承
            raise TypeError("Constrain modules must inherit from a torch.nn.Module class")
        if mro.index(__class__) > mro.index(torch.nn.Module): # 继承时此类必须写在前边，torch.nn.module才能被初始化
            raise TypeError(
                "CModuleMixin must be placed before any torch.nn.Module class in constrain module inheritance."
            )
        # This will setup the torch.nn.Module
        super().__init__(*args, **kwargs) # 原始linear或conv等线性module的初始化

        constrain = {} if constrain is None else constrain
        self.clamp_weight     = constrain.get('clamp_weight_value', None)
        self.clamp_bias       = constrain.get('clamp_bias_value', None)
        self.clamp_activation = constrain.get('clamp_activation_value', None)
        self.clamp_factor     = constrain.get('clamp_factor_value', None)

        self._constrain_hooks = {}
        # if open_ihook:
        #     self._constrain_hooks["input"] = self.register_forward_pre_hook(self.constrain_input)
        if open_ohook:
            self._constrain_hooks["output"] = self.register_forward_hook(self.constrain_output)

    @classmethod
    def from_module(
        cls,
        module: torch.nn.Module,
        constrain: Optional[Union[str]] = None,
    ):
        # Create the constrain module on the meta device to prevent weights intialization
        device = QUANT_CONFIGS.device
        cmodule = cls.ccreate(module, constrain = constrain, device=device)
        if cmodule is None:
            return None

        if hasattr(module, 'weight'):
            with torch.no_grad():
                cmodule.weight = module.weight
                if hasattr(module, 'bias') and module.bias is not None:
                    cmodule.bias = module.bias

        return cmodule.to(device)

    @classmethod
    def ccreate(
        cls,
        module: torch.nn.Module,
        constrain: Optional[Union[str]] = None,
        device: Optional[torch.device] = None,
    ):
        raise NotImplementedError

    @property
    def cweight(self):
        if self.clamp_factor is not None:
            with torch.no_grad():
                clamp_data = self.weight.abs().mean() * self.clamp_factor
        else:
            clamp_data = self.clamp_weight
        return self.weight if clamp_data is None else torch.clamp(self.weight, min = -clamp_data, max = clamp_data)

    @property
    def cbias(self):
        return self.bias if self.clamp_bias is None else torch.clamp(self.bias, min = -self.clamp_bias, max = self.clamp_bias)


    def cforward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def constrain_input(self, module: torch.nn.Module, input: torch.Tensor) -> torch.Tensor:
        return input if self.clamp_activation is None else torch.clamp(input[0], min = -self.clamp_activation, max = self.clamp_activation)

    def constrain_output(
        self,
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return output if self.clamp_activation is None else torch.clamp(output, min = -self.clamp_activation, max = self.clamp_activation)
    
    def extra_repr(self):
        s = ''
        extra_s = ''
        if 'Conv2d' in self._get_name():
            s = nn.Conv2d.extra_repr(self)
        elif 'Linear' in self._get_name():
            s = nn.Linear.extra_repr(self)
        elif 'LSTM' in self._get_name():
            s = nn.LSTM.extra_repr(self)
            
        extra_s = ', clamp_activation:{}, clamp_weight:{}, clamp_bias:{}, clamp_factor:{}'.format(self.clamp_activation, self.clamp_weight, self.clamp_bias, self.clamp_factor)
        return s + extra_s

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        main_str = self._get_name() + '('
        main_str += extra_lines[0]
        main_str += ')'
        return main_str

