import torch
import torch.onnx
from typing import Callable, List
from functools import partial

# ops_name
LINGER = '_linger'

LINGER_QTENSOR_LAYER_COUNTER = LINGER+"_qtensor_index_"
LINGER_QTENSOR_LAYERS_PREIFX = LINGER+'_qtensor'

self_module = []

def get_current_module():
    if len(self_module) > 0:
        return self_module[-1]
    else:
        return None

def hook_pre_forward(module, input):
    setattr(module, LINGER_QTENSOR_LAYER_COUNTER, 0)
    self_module.append(module)

def hook_forward(module, input, output):
    cur = self_module.pop()
    assert cur == module
    setattr(module, LINGER_QTENSOR_LAYER_COUNTER, 0)


_QMODULE_TABLE = {}
_QTENSOR_OP_TABLE = {}

def register_qmodule(module_cls):
    """
    Used for registering a new quantized module.

    The QModule must implement two abstract methods:

    - qcreate: class method to instantiate a new QModule from an nn.Module, without copying its weights,
    - forward: instance method for quantized inference.

    The code to register a new module looks like:

    ```
    @register_qmodule(<base torch.nn.Module>)
    class MyQModule(QModuleMixin, <base torch.nn.Module>):
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
        _QMODULE_TABLE[module_cls] = cls
        return cls

    return wrapper

def register_qtensor_op(aten_ops: List[Callable]):
    """
    Used for registering a new __torch_dispatch__ aten operation to QBytesTensor.

    The code to register a new operation looks like:

    @register_qbytestensor_op(list_of_ops)
    def foo(op, *args, **kwargs):
        <implementation>
    """

    def wrapper(op):
        for aten_op in aten_ops:
            _QTENSOR_OP_TABLE[aten_op] = partial(op, aten_op)

    return wrapper

def get_qtensor_op_dispatch(aten_op):
    return _QTENSOR_OP_TABLE.get(aten_op, None)

def get_qmodule_op(module_op):
    return _QMODULE_TABLE.get(type(module_op), None)

def quantize_module(
    module,
    activations_cfg = None,
    *args,
    **kwargs
):
    if type(module) in _QMODULE_TABLE.keys():
        qcls = _QMODULE_TABLE[type(module)]
        return qcls.from_module(module, activations_cfg, *args, **kwargs)
    return None

def quantize_tensor(
    module,
    activate_cfg = None,
    *args,
    **kwargs
):
    if module in _QMODULE_TABLE.keys():
        qcls = _QMODULE_TABLE[module]
        return qcls.from_module(module, activate_cfg, *args, **kwargs)
    return None

__all__ = ["_QMODULE_TABLE", "_QTENSOR_OP_TABLE", "LINGER_QTENSOR_LAYER_COUNTER", "LINGER_QTENSOR_LAYERS_PREIFX", \
           "get_current_module", "hook_pre_forward", "hook_forward", "register_qmodule", "register_qtensor_op", \
            "get_qmodule_op", "get_qtensor_op_dispatch", "quantize_module", "quantize_tensor"]
