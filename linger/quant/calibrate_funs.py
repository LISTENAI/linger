import torch
import torch.onnx
from torch.onnx import is_in_onnx_export

_QCALIBRATE_TABLE = {}


def register_calibrate_method(module_cls):
    """
    可以从外部调用此功能,注册新的校准方法。
    函数输入有两个,分别为self和input,self表示input(权重或激活数据)对应的量化器。
    如果为TQT训练,需要根据input动态的更新self中的learning_data和scale。
    如果为PTQ量化,仅需更新scale即可。
    """

    def wrapper(cls):
        _QCALIBRATE_TABLE[module_cls] = cls
        return cls

    return wrapper


def get_calibrate_function(calibrate_name):
    return _QCALIBRATE_TABLE.get(calibrate_name, None)

@register_calibrate_method('abs_max')
def abs_max_init(self, tensor, *args):
    with torch.no_grad():
        if self.is_calibrate:
            raise ValueError("Quantizer has beem calibrated! ")
        self.learning_data.fill_(tensor.abs().max().log2())
        learning_data = self.data_bits - 1 - self.learning_data.squeeze(0)
        learning_data = self.quant_round(learning_data, self.round_mode)
        scale = 2**learning_data
        self.scale = scale.clamp(min=1e-6, max=2**24)
        self.is_calibrate.fill_(True)

@register_calibrate_method('top_10')
def top_10_init(self, tensor, *args):
    with torch.no_grad():
        if self.is_calibrate:
            raise ValueError("Quantizer has beem calibrated! ")
        if tensor.numel() > 11:
            self.learning_data.fill_((torch.topk(tensor.abs().flatten(), 10)[0][-1]).log2())
        else: # 可能有的激活没有10个元素
            self.learning_data.fill_(tensor.abs().max().log2())
        learning_data = self.data_bits - 1 - self.learning_data.squeeze(0)
        learning_data = self.quant_round(learning_data, self.round_mode)
        scale = 2**learning_data
        self.scale.fill_(scale.clamp(min=1e-6, max=2**24))
        self.is_calibrate.fill_(True)


def get_best_pow2coef_W(w, bit):
    min_int = -(2**(bit-1))
    max_int = -min_int - 1
    def fake_quant(x, scale):
        x_int = (x / scale).round()
        x_int = torch.clamp(x_int, min_int,max_int)
        x_out = x_int * scale
        return x_out
    scale = []

    for i in range(-10,10):
        scale.append(2**i)
    
    score = []
    for i in range(len(scale)):
        #计算每个scale的得分，用score_temp存储每个input的得分
        q_w = fake_quant(w, scale[i])
        score.append( (w - q_w).norm() )
    
    return (torch.tensor(scale[score.index(min(score))])).log2().to("cuda") + bit - 1

@register_calibrate_method('w_like')
def w_like_init(self, tensor, *args):
    with torch.no_grad():
        if self.is_calibrate:
            raise ValueError("Quantizer has beem calibrated! ")
        self.learning_data.fill_(get_best_pow2coef_W(tensor, args[0]))
        learning_data = self.data_bits - 1 - self.learning_data.squeeze(0)
        learning_data = self.quant_round(learning_data, self.round_mode)
        scale = 2**learning_data
        self.scale = scale.clamp(min=1e-6, max=2**24)
        self.is_calibrate.fill_(True)



