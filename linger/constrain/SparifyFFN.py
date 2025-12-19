#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.onnx import is_in_onnx_export
from torch.nn import init

from .cutils import static_clip, dyn_clip_weight

class GetSparifyMask(autograd.Function):
    @staticmethod
    def forward(ctx, outM, threshold):
        # 计算要保留的元素数量
        n_elements = outM.size(-1)
        k = max(1, min(n_elements, round(threshold * n_elements)))        
        # 获取前k个最大值的索引
        _, topk_indices = torch.topk(outM, k, dim=-1)        
        # 创建全False的掩码
        mask = torch.zeros_like(outM, dtype=torch.float)
        # 将前k个最大值位置设为True
        mask.scatter_(-1, topk_indices, True)
        out_mask = mask.to(torch.float)
        ctx.save_for_backward(out_mask)
        return out_mask
    @staticmethod
    def backward(ctx, gradOutput):
        (out_mask, ) = ctx.saved_tensors
        return gradOutput * out_mask, None

class SparseStrategy():
    def __init__(self, name, ratio=0.125, step_max = 3600, grad_accu=6):
        self.count_max = step_max * grad_accu # 6表示梯度累计
        self.count = 0
        self.sparse_ratio = ratio # ratio表示百分之多少稀疏度
        self.name = name

        # step模式才会有
        self.step_nums = 10
    def next_sparse_ratio(self):
        self.count = self.count + 1
        if self.count >= self.count_max:
            return self.sparse_ratio
        if self.name == "linear":
            return 1 - (1 - self.sparse_ratio) * self.count / self.count_max
        elif self.name == "sqrt":
            return 1 - (1 - self.sparse_ratio) * math.sqrt(self.count / self.count_max)
        elif self.name == "step":
            tmp = (1 - self.sparse_ratio)
            step1 = tmp * self.count / self.count_max
            step2 = math.ceil(step1 / (tmp / self.step_nums)) * (tmp / self.step_nums)
            return 1 - step2

class SparifyFFN(nn.Module):
    def __init__(self, in_feature, ou_feature, bias=True, normalize_data=None, normalize_weight=None, normalize_bias=None, normalize_factor=None, dtype = torch.float32):
        super(SparifyFFN, self).__init__()
        
        self.input_size = in_feature
        self.output_size = ou_feature
        self.bias = bias

        self.mask_group = 8
        self.ratio = 0.125

        self.normalize_data   = normalize_data
        self.normalize_weight = normalize_weight
        self.normalize_bias   = normalize_bias
        self.normalize_factor = normalize_factor

        self.weight_fc1  = nn.Parameter(torch.empty((self.output_size, self.input_size), dtype = dtype))
        self.weight_fc2  = nn.Parameter(torch.empty((self.input_size, self.output_size), dtype = dtype))
        self.weight_mask = nn.Parameter(torch.empty((self.mask_group, self.input_size), dtype = dtype))

        if bias:
            self.bias_fc1  = nn.Parameter(torch.empty(self.output_size, dtype = dtype))
            self.bias_fc2  = nn.Parameter(torch.empty(self.input_size, dtype = dtype))
            self.bias_mask = nn.Parameter(torch.empty(self.mask_group, dtype = dtype))

        self.repeat_num = int(self.output_size / self.mask_group)
        self.spa_method = SparseStrategy("sqrt", ratio=self.ratio, step_max=20000, grad_accu=12)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.output_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, open_spa_method=True) -> torch.Tensor:
        normalized_fc1_w = self.weight_fc1
        normalized_fc2_w = self.weight_fc2
        normalized_mask_w = self.weight_mask
        if self.normalize_weight is not None:
            normalized_fc1_w = static_clip(normalized_fc1_w, self.normalize_weight, self.training)
            normalized_fc2_w = static_clip(normalized_fc2_w, self.normalize_weight, self.training)
            normalized_mask_w = static_clip(normalized_mask_w, self.normalize_weight, self.training)
        elif self.normalize_factor is not None:
            normalized_fc1_w = dyn_clip_weight(normalized_fc1_w, self.normalize_factor)
            normalized_fc2_w = dyn_clip_weight(normalized_fc2_w, self.normalize_factor)
            normalized_mask_w = dyn_clip_weight(normalized_mask_w, 7)    # only support 8bit

        normalized_fc1_b = None
        normalized_fc2_b = None
        normalized_mask_b = None
        if self.bias:
            normalized_fc1_b = self.bias_fc1
            normalized_fc2_b = self.bias_fc2
            normalized_mask_b = self.bias_mask
            if self.normalize_bias is not None:
                normalized_fc1_b  = static_clip(normalized_fc1_b, self.normalize_bias, self.training)
                normalized_fc2_b  = static_clip(normalized_fc2_b, self.normalize_bias, self.training)
                normalized_mask_b = static_clip(normalized_mask_b, self.normalize_bias, self.training)

        outL = F.linear(input, normalized_fc1_w, normalized_fc1_b)
        outM1 = F.linear(input, normalized_mask_w, normalized_mask_b)
        outM = F.softmax(outM1, dim=-1)
        
        if self.training and open_spa_method:
            threshold = self.spa_method.next_sparse_ratio() # 训练时需要改成这个
        else:
            threshold = self.ratio
        
        # mask = topk_sort(outM, threshold)
        mask = GetSparifyMask.apply(outM, threshold)

        # 第二步，outM转化为bool值后，根据out_feature扩散为与其一致的大小。在这里展示block，将channel进行分组
        outM2 = mask.repeat_interleave(self.repeat_num, dim=-1)
        out1 = outL * outM2  # fc1 sparify

        out2 = F.relu(out1)

        out = F.linear(out2, normalized_fc2_w, normalized_fc2_b)

        if self.normalize_data is not None:
            # out = static_clip(out, self.normalize_data, self.training, False)
            out.clamp_(-self.normalize_data, self.normalize_data)
        # import pdb; pdb.set_trace()
        return out
            
    def extra_repr(self):
        # s = nn.GRU.extra_repr(self)
        s = 'in_feature:{},ou_feature:{}'.format(self.input_size, self.output_size)
        # s += ', open_spa_method:{}'.format(self.open_spa_method)

        extra_s = ', normalize_data:{normalize_data}, normalize_weight:{normalize_weight}, normalize_bias:{normalize_bias}, normalize_factor:{normalize_factor}'.format(
            **self.__dict__)
        return s+extra_s

__all__ = ['SparifyFFN']
