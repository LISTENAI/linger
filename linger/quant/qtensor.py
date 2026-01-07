import torch
from torch._C import DisableTorchFunction
from packaging.version import Version
from .ops import *

if Version(torch.__version__) >= Version("2.0"):
    from torch.utils import _pytree as pytree
    def qfallback(callable, *args, **kwargs):
        kwargs = kwargs or {}
        args, kwargs = pytree.tree_map_only(QTensor, lambda x: x.value, (args, kwargs))
        return callable(*args, **kwargs)
else:
    from torch.utils._pytree import tree_map
    def tree_map_only(ty, fn, obj):
        def _fn(x):
            if isinstance(x, ty):
                return fn(x)
            return x
        return tree_map(_fn, obj)
    
    def qfallback(callable, *args, **kwargs):
        kwargs = kwargs or {}
        args, kwargs = tree_map_only(QTensor, lambda x: x.value, (args, kwargs))
        return callable(*args, **kwargs)

class Convert2QTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, scale, data_bits):
        s = QTensor(t, scale, data_bits)
        return s

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None

class Convert2Tensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t):
        return t.clone()

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput

def from_tensor_to_qtensor(t, scale: float = None, data_bits: int = None, zero_point=0):
    qt = Convert2QTensor.apply(t, scale, data_bits)
    return qt

def from_qtensor_to_tensor(t):
    assert isinstance(t, QTensor)
    return Convert2Tensor.apply(t)


class QTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, scale=1.0, data_bits=8, zero_point=0):
        return torch.Tensor._make_subclass(cls, data, require_grad=data.requires_grad)

    def __init__(self, data, scale=1.0, data_bits=8, zero_point=0):
        self.scale = scale
        self.data_bits = data_bits
        self.value = data

    if Version(torch.__version__) >= Version("1.7.0"):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):        
            if kwargs is None:
                kwargs = {}

            if not all(issubclass(cls, t) for t in types):
                return NotImplemented

            with torch._C.DisableTorchFunction():
                qdispatch = get_qtensor_op_dispatch(func)
                if qdispatch is not None:
                    return qdispatch(*args, **kwargs)
                ret = func(*args, **kwargs)
                return ret
            
    if Version(torch.__version__) >= Version("2.0"):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            op = func.overloadpacket
            qdispatch = get_qtensor_op_dispatch(op)
            if qdispatch is not None:
                return qdispatch(*args, **kwargs)
            return qfallback(func, *args, **kwargs)
    else:
        def __torch_dispatch__(self, func, args=(), kwargs=None):
            kwargs = kwargs or {}
            op = func.overloadpacket
            qdispatch = get_qtensor_op_dispatch(op)
            if qdispatch is not None:
                return qdispatch(*args, **kwargs)
            return qfallback(func, *args, **kwargs)

    # # 重写add方法以支持torch.add语法
    # def add(self, other, alpha=1):
    #     if not isinstance(other, (QTensor, float, int)) or self.dtype != torch.float:
    #         return super(QTensor, self).add(other, alpha=alpha)
    #     module_self = get_current_module()
    #     if module_self is None:
    #         return super(QTensor, self).add(other, alpha=alpha)
        
    #     iname_index = getattr(module_self, LINGER_QTENSOR_LAYER_COUNTER)
    #     setattr(module_self, LINGER_QTENSOR_LAYER_COUNTER, iname_index+1)
    #     return qadd(module_self, self, other, str(iname_index))
        
    # # 重写__add__方法以支持a + b语法
    # def __add__(self, other):
    #     return self.add(other)
    
    # # 重写__iadd__方法以支持a += b语法
    # def __iadd__(self, other):
    #     return self.add(other)
    
    # # 重写bmm方法
    # def bmm(self, mat2):
    #     """
    #     重写的bmm方法
    #     """
    #     # 检查是否两个张量都已经量化
    #     if not isinstance(self, QTensor) or not isinstance(mat2, QTensor):
    #         return super(QTensor, self).bmm(mat2)
    #     module_self = get_current_module()
    #     if module_self is None:
    #         return super(QTensor, self).bmm(mat2)
        
    #     iname_index = getattr(module_self, LINGER_QBMM_LAYER_COUNTER)
    #     setattr(module_self, LINGER_QBMM_LAYER_COUNTER, iname_index+1)
    #     return qbmm(module_self, self, mat2, str(iname_index))

    # def flatten(self, start_dim: int = 0, end_dim: int = -1):
    #     y = super(QTensor, self).flatten(start_dim, end_dim)
    #     if isinstance(self, QTensor):
    #         y = from_tensor_to_qtensor(self, self.scale, self.data_bits)
    #     return y



    