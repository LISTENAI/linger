import torch
import torch.nn as nn
import torch.nn.functional as F

from .qmodule import QModuleMixin
from ..qconfig import register_qmodule
from ...qtensor import QTensor, from_tensor_to_qtensor, from_qtensor_to_tensor
from ....onnx import quantlinear, generate_onnx_qparam_dict, QDOMAIN_NAME
from typing import Optional, Union, Dict, Any

@register_qmodule(nn.MaxPool2d)
class QMaxPool2d(QModuleMixin, nn.MaxPool2d):
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
            kernel_size = module.kernel_size,
            stride = module.stride,
            padding = module.padding,
            dilation = module.dilation,
            return_indices = module.return_indices,
            ceil_mode = module.ceil_mode,

            device = device,
            activations_cfg=activations_cfg,
            weights_cfg=None,
            constrain=None, 
            bias_cfg=None,
            open_ihook = False,
            open_ohook = False,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(input, QTensor):
            tmp_input = from_qtensor_to_tensor(input)
            scale = input.scale.detach()
            data_bits = input.data_bits
            out =  F.max_pool2d(
                tmp_input,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                ceil_mode=self.ceil_mode,
                return_indices=self.return_indices,
            )
            return from_tensor_to_qtensor(out, scale, data_bits)
        else:
            return F.max_pool2d(
                    input,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    ceil_mode=self.ceil_mode,
                    return_indices=self.return_indices,
                )
        

