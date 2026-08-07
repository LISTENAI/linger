import torch
import torch.onnx
from torch.onnx import is_in_onnx_export
from collections import Counter

from ..config import QUANT_CONFIGS

_QCALIBRATE_TABLE = {}


def _compute_kl_divergence_torch(p, q):
    """计算KL散度，p和q需要已归一化（torch版本）"""
    mask = (p > 1e-10) & (q > 1e-10)
    p_masked = p[mask]
    q_masked = q[mask]
    if len(p_masked) == 0:
        return float("inf")
    return torch.sum(p_masked * torch.log(p_masked / q_masked)).item()


def _normalize_distr_torch(distr):
    """归一化分布（torch版本）"""
    total = torch.sum(distr)
    if total != 0:
        distr = distr / total
    return distr


def _torch_digitize(x, bins):
    """
    torch版本的digitize实现
    返回每个x元素所属的bin索引
    """
    indices = torch.bucketize(x.contiguous(), bins.contiguous(), right=True) - 1
    return indices


def tensorrt_hist_calibration_torch(fp32_tensor, num_bins=2048, num_bits=8, unsigned=False, stride=1, start_bin=128):
    """
    TensorRT直方图量化校准的torch实现

    Args:
        fp32_tensor: 输入的FP32 torch.tensor
        num_bins: 直方图的bin数量
        num_bits: 量化位数，默认8
        unsigned: 是否为无符号量化
        stride: 遍历步长
        start_bin: 起始bin位置

    Returns:
        best_amax: 最优截断阈值
        scale: 量化scale
        min_kl: 最小KL散度
    """
    # 处理负值（取绝对值）
    abs_data = torch.abs(fp32_tensor)
    abs_max_full = torch.max(abs_data).item()

    # 使用torch.histc生成直方图
    hist = torch.histc(abs_data, bins=num_bins, min=0, max=abs_max_full)

    # 生成bin_edges
    bin_edges = torch.linspace(0, abs_max_full, num_bins + 1)

    # 处理第一个bin为空的情况
    bins = hist.clone()
    if len(bins) > 1:
        bins[0] = bins[1]

    total_data = torch.sum(bins).item()

    # 计算量化后的bin数量（对称量化为128）
    nbins = 1 << (num_bits - 1 + int(unsigned))

    divergences = []

    # 从start_bin开始正向遍历
    stop = len(bins)
    for i in range(start_bin, stop + 1, stride):
        new_density_counts = torch.zeros(nbins, dtype=torch.float64)

        # 将bins[0:i]量化为nbins个bin
        space = torch.linspace(0, i, nbins + 1)

        # 记录量化前每个bin在量化后的位置
        indices = torch.arange(i, dtype=torch.float64)
        digitized_space = _torch_digitize(indices, space)

        # 处理空bin：直方图值为0的位置设为-1
        digitized_space[bins[:i] == 0] = -1

        # 计算量化后的数据分布Q
        bins_slice = bins[:i]
        for idx in range(i):
            digitized = digitized_space[idx].item()
            if digitized != -1:
                new_density_counts[digitized] += bins_slice[idx]

        # 统计每个量化bin包含的原始bin数量
        counter = Counter(digitized_space.tolist())
        for key, val in counter.items():
            if key != -1:
                new_density_counts[key] = new_density_counts[key] / val

        # 扩展回原始bin数量（此时还未归一化）
        new_density = torch.zeros(i, dtype=torch.float64)
        for idx in range(i):
            digitized = digitized_space[idx].item()
            if digitized != -1:
                new_density[idx] = new_density_counts[digitized]

        # 计算参考分布P（截断后的原始分布）
        reference_density = bins[:len(digitized_space)].clone()
        # 将i之后的bin累加到最后一个bin
        reference_density[-1] += torch.sum(bins[i:])

        # 归一化分布
        new_density = _normalize_distr_torch(new_density)
        reference_density = _normalize_distr_torch(reference_density)

        # 计算KL散度
        kl = _compute_kl_divergence_torch(reference_density, new_density)
        divergences.append(kl)

    # 找到最小KL散度对应的阈值
    divergences_tensor = torch.tensor(divergences)
    last_argmin = len(divergences) - 1 - torch.argmin(divergences_tensor.flip(0)).item()
    best_amax = bin_edges[last_argmin * stride + start_bin].item()

    # 计算scale
    scale = best_amax / nbins

    return best_amax, scale, divergences[last_argmin]


def register_calibrate_method(module_cls):
    """
    可以从外部调用此功能,注册新的校准方法。
    函数输入有两个,分别为self和input,self表示input(权重或激活数据)对应的量化器。
    如果为TQT训练,需要根据input动态的更新self中的running_data和scale。
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
        # 多轮校准模式：累积数据，暂不计算最终scale
        _is_accumulating = getattr(self, '_is_accumulating', None)
        if _is_accumulating is None:
            _is_accumulating = QUANT_CONFIGS.calibration

        if _is_accumulating:
            # 累积模式：更新running_data为全局max
            current_max = tensor.abs().max().clamp(min=1e-6).log2()
            if not hasattr(self, '_accumulated_max'):
                self._accumulated_max = current_max.item()
            else:
                self._accumulated_max = max(self._accumulated_max, current_max.item())
            self.running_data.fill_(self._accumulated_max)
            # 标记已经开始校准（用于finalize_calibration判断）
            self.is_calibrate.fill_(True)
            return

        if self.is_calibrate:
            raise ValueError("Quantizer has beem calibrated! ")
        self.running_data.fill_(tensor.abs().max().clamp(min=1e-6).log2())
        running_data = self.data_bits - 1 - self.running_data.squeeze(0)
        running_data = self._limit_scale_exponent(
            self.quant_round(running_data, self.round_mode)
        )
        scale = torch.exp2(running_data)
        self.scale.fill_(float(self._limit_scale(scale)))
        self.is_calibrate.fill_(True)

@register_calibrate_method('top_10')
def top_10_init(self, tensor, *args):
    with torch.no_grad():
        # 多轮校准模式：累积数据，暂不计算最终scale
        _is_accumulating = getattr(self, '_is_accumulating', None)
        if _is_accumulating is None:
            _is_accumulating = QUANT_CONFIGS.calibration

        if _is_accumulating:
            # 累积模式：收集所有值，finalize时计算top-10
            current_values = tensor.abs().flatten()
            if not hasattr(self, '_accumulated_values'):
                self._accumulated_values = current_values.clone()
            else:
                self._accumulated_values = torch.cat([self._accumulated_values, current_values])
            # 标记已经开始校准（用于finalize_calibration判断）
            self.is_calibrate.fill_(True)
            return

        if self.is_calibrate:
            raise ValueError("Quantizer has beem calibrated! ")
        if tensor.numel() > 11:
            self.running_data.fill_((torch.topk(tensor.abs().flatten(), 10)[0][-1]).log2())
        else: # 可能有的激活没有10个元素
            self.running_data.fill_(tensor.abs().max().clamp(min=1e-6).log2())
        running_data = self.data_bits - 1 - self.running_data.squeeze(0)
        running_data = self._limit_scale_exponent(
            self.quant_round(running_data, self.round_mode)
        )
        scale = torch.exp2(running_data)
        self.scale.fill_(float(self._limit_scale(scale)))
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



@register_calibrate_method('histogram')
def histogram_init(self, tensor, *args):
    """
    TensorRT 直方图校准方法
    使用 KL 散度来找到最优的量化 scale
    支持多轮校准：在累积模式下收集histogram数据，finalize时计算最终scale
    """
    with torch.no_grad():
        # 多轮校准模式：累积数据，暂不计算最终scale
        _is_accumulating = getattr(self, '_is_accumulating', None)
        if _is_accumulating is None:
            _is_accumulating = QUANT_CONFIGS.calibration

        if _is_accumulating:
            bit = args[0] if len(args) > 0 else 8
            num_bins = 2048

            # 计算当前batch的abs_max
            abs_data = torch.abs(tensor)
            current_abs_max = torch.max(abs_data).item()

            # 更新全局abs_max
            if not hasattr(self, '_accumulated_abs_max'):
                self._accumulated_abs_max = current_abs_max
            else:
                self._accumulated_abs_max = max(self._accumulated_abs_max, current_abs_max)

            # 使用全局abs_max生成histogram并累积
            hist = torch.histc(abs_data, bins=num_bins, min=0, max=self._accumulated_abs_max)

            # 累积histogram
            if not hasattr(self, '_accumulated_hist') or self._accumulated_hist is None:
                self._accumulated_hist = hist.clone()
            else:
                self._accumulated_hist = self._accumulated_hist + hist

            # 标记已经开始校准（用于finalize_calibration判断）
            self.is_calibrate.fill_(True)
            return

        if self.is_calibrate:
            raise ValueError("Quantizer has been calibrated! ")

        # 从 args 获取量化位数，默认为 8
        bit = args[0] if len(args) > 0 else 8

        # 使用 TensorRT 直方图校准方法 (KL散度)
        best_amax, scale, min_kl = tensorrt_hist_calibration_torch(
            tensor,
            num_bins=2048,
            num_bits=bit,
            unsigned=False,
            stride=1,
            start_bin=128
        )

        # 将 scale 转换为 log2 格式存储
        self.running_data.fill_(scale.log2().clamp(min=-24, max=24))

        running_data = self.data_bits - 1 - self.running_data.squeeze(0)
        running_data = self.quant_round(running_data, self.round_mode)
        scale = torch.exp2(running_data)
        self.scale = scale.clamp(min=1e-6, max=2**24)
        self.is_calibrate.fill_(True)

@register_calibrate_method('w_like')
def w_like_init(self, tensor, *args):
    with torch.no_grad():
        # 多轮校准模式：累积数据，暂不计算最终scale
        _is_accumulating = getattr(self, '_is_accumulating', None)
        if _is_accumulating is None:
            _is_accumulating = QUANT_CONFIGS.calibration

        if _is_accumulating:
            # 累积模式：收集tensor数据
            if not hasattr(self, '_accumulated_data') or self._accumulated_data is None:
                self._accumulated_data = tensor.clone()
            else:
                self._accumulated_data = torch.cat([self._accumulated_data.flatten(), tensor.flatten()])
            # 标记已经开始校准（用于finalize_calibration判断）
            self.is_calibrate.fill_(True)
            return

        if self.is_calibrate:
            raise ValueError("Quantizer has beem calibrated! ")
        self.running_data.fill_(get_best_pow2coef_W(tensor, args[0]))
        running_data = self.data_bits - 1 - self.running_data.squeeze(0)
        running_data = self._limit_scale_exponent(
            self.quant_round(running_data, self.round_mode)
        )
        scale = torch.exp2(running_data)
        self.scale.fill_(float(self._limit_scale(scale)))
        self.is_calibrate.fill_(True)


# ============ Finalize Calibration Functions ============
# 多轮校准时，在退出calibration上下文时调用这些finalize函数

def _finalize_histogram(self, *args):
    """
    Finalize histogram calibration - use accumulated histogram to compute final scale
    """
    if not hasattr(self, '_accumulated_hist') or self._accumulated_hist is None:
        return  # 没有累积数据，跳过

    bit = self.data_bits
    num_bins = 2048

    # 获取累积的histogram和abs_max
    bins = self._accumulated_hist
    abs_max_full = self._accumulated_abs_max

    # 处理第一个bin为空的情况
    if len(bins) > 1:
        bins[0] = bins[1]

    # 计算量化后的bin数量（对称量化为128）
    nbins = 1 << (bit - 1)

    divergences = []

    # 从start_bin开始正向遍历
    start_bin = 128
    stride = 1
    stop = len(bins)
    bin_edges = torch.linspace(0, abs_max_full, num_bins + 1)

    for i in range(start_bin, stop + 1, stride):
        new_density_counts = torch.zeros(nbins, dtype=torch.float64)

        # 将bins[0:i]量化为nbins个bin
        space = torch.linspace(0, i, nbins + 1)

        # 记录量化前每个bin在量化后的位置
        indices = torch.arange(i, dtype=torch.float64)
        digitized_space = _torch_digitize(indices, space)

        # 处理空bin：直方图值为0的位置设为-1
        digitized_space[bins[:i] == 0] = -1

        # 计算量化后的数据分布Q
        bins_slice = bins[:i]
        for idx in range(i):
            digitized = digitized_space[idx].item()
            if digitized != -1:
                new_density_counts[digitized] += bins_slice[idx]

        # 统计每个量化bin包含的原始bin数量
        counter = Counter(digitized_space.tolist())
        for key, val in counter.items():
            if key != -1:
                new_density_counts[key] = new_density_counts[key] / val

        # 扩展回原始bin数量（此时还未归一化）
        new_density = torch.zeros(i, dtype=torch.float64)
        for idx in range(i):
            digitized = digitized_space[idx].item()
            if digitized != -1:
                new_density[idx] = new_density_counts[digitized]

        # 计算参考分布P（截断后的原始分布）
        reference_density = bins[:len(digitized_space)].clone()
        # 将i之后的bin累加到最后一个bin
        reference_density[-1] += torch.sum(bins[i:])

        # 归一化分布
        new_density = _normalize_distr_torch(new_density)
        reference_density = _normalize_distr_torch(reference_density)

        # 计算KL散度
        kl = _compute_kl_divergence_torch(reference_density, new_density)
        divergences.append(kl)

    # 找到最小KL散度对应的阈值
    divergences_tensor = torch.tensor(divergences)
    last_argmin = len(divergences) - 1 - torch.argmin(divergences_tensor.flip(0)).item()
    best_amax = bin_edges[last_argmin * stride + start_bin].item()

    # 计算scale
    scale = best_amax / nbins

    # 设置最终scale
    self.running_data.fill_(scale.log2().clamp(min=-24, max=24))
    running_data = self.data_bits - 1 - self.running_data.squeeze(0)
    running_data = self.quant_round(running_data, self.round_mode)
    scale = torch.exp2(running_data)
    self.scale = scale.clamp(min=1e-6, max=2**24)

    # 清理累积数据
    self._accumulated_hist = None
    self._accumulated_abs_max = None


def _finalize_abs_max(self, *args):
    """
    Finalize abs_max calibration - running_data already contains the max value
    """
    if not hasattr(self, '_accumulated_max'):
        return  # 没有累积数据，跳过

    running_data = self.data_bits - 1 - self.running_data.squeeze(0)
    running_data = self._limit_scale_exponent(
        self.quant_round(running_data, self.round_mode)
    )
    scale = torch.exp2(running_data)
    self.scale.fill_(float(self._limit_scale(scale)))

    # 清理累积数据
    self._accumulated_max = None


def _finalize_top_10(self, *args):
    """
    Finalize top_10 calibration - compute top-10 from accumulated values
    """
    if not hasattr(self, '_accumulated_values') or self._accumulated_values is None:
        return  # 没有累积数据，跳过

    values = self._accumulated_values
    if values.numel() > 10:
        top_k = torch.topk(values.abs().flatten(), 10)[0][-1].log2()
    else:
        top_k = values.abs().max().clamp(min=1e-6).log2()

    self.running_data.fill_(top_k)
    running_data = self.data_bits - 1 - self.running_data.squeeze(0)
    running_data = self._limit_scale_exponent(
        self.quant_round(running_data, self.round_mode)
    )
    scale = torch.exp2(running_data)
    self.scale.fill_(float(self._limit_scale(scale)))

    # 清理累积数据
    self._accumulated_values = None


def _finalize_w_like(self, *args):
    """
    Finalize w_like calibration - compute best scale from accumulated data
    """
    if not hasattr(self, '_accumulated_data') or self._accumulated_data is None:
        return  # 没有累积数据，跳过

    self.running_data.fill_(get_best_pow2coef_W(self._accumulated_data, args[0]))
    running_data = self.data_bits - 1 - self.running_data.squeeze(0)
    running_data = self._limit_scale_exponent(
        self.quant_round(running_data, self.round_mode)
    )
    scale = torch.exp2(running_data)
    self.scale.fill_(float(self._limit_scale(scale)))

    # 清理累积数据
    self._accumulated_data = None


# Finalize函数映射表
_FINALIZE_TABLE = {
    'histogram': _finalize_histogram,
    'abs_max': _finalize_abs_max,
    'top_10': _finalize_top_10,
    'w_like': _finalize_w_like,
}


def get_finalize_function(calibrate_name):
    """获取指定校准方法的finalize函数"""
    return _FINALIZE_TABLE.get(calibrate_name, None)

