import torch
import torch.nn as nn
from linger.quant.qtensor import from_tensor_to_qtensor
import numpy as np

import linger

class QuantizationTestNet(nn.Module):
    def __init__(self, num_classes=10):
        super(QuantizationTestNet, self).__init__()
        
        # 简单的卷积层用于构建完整模型
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(16 * 4 * 4, num_classes)
    
    def quantized_tanh(self, x_float):
        """执行量化tanh并与浮点tanh比较"""
        
        float_tanh = torch.tanh(x_float)

        # 将输入量化
        scale = torch.tensor(128, dtype=torch.int32)
        x_quant = from_tensor_to_qtensor(x_float, scale, 8)
        
        quant_tanh = torch.tanh(x_quant)
        
        # 计算差异
        diff = torch.abs(float_tanh) - torch.abs(quant_tanh)

        #计算平均差异和平均浮点结果
        mean_float = torch.mean(float_tanh).item()
        mean_diff = torch.mean(diff).item()

        #相对差异
        relative_diff = mean_diff/mean_float
        
        return float_tanh, quant_tanh, relative_diff
    
    def forward(self, x):
        # 在forward中比较两种tanh
        float_result, quant_result, relative_diff = self.quantized_tanh(x)
        
        # 返回正常输出和量化比较结果
        return {
            'float_addition': float_result,
            'quant_addition': quant_result,
            'relative_difference': relative_diff
        }

def test_quantization_in_forward(model, num_tests=10):
    print("测试模型前向传播中的量化tanh比较...")
    print("=" * 60)
    
    model.eval()

    model = linger.init(model)
    # print(model)

    all_relative_diffs = []
    
    for i in range(num_tests):
        # 生成随机输入数据
        batch_size = 2
        min_val = 1e-6  # 最小正值
        x = torch.rand(batch_size, 3, 32, 32) * (1.0 - min_val) + min_val
        
        # 前向传播（包含量化比较）
        with torch.no_grad():
            outputs = model(x)
    
        relative_diff = outputs['relative_difference']
    
        all_relative_diffs.append(relative_diff)
        
        print(f"测试 {i+1}:")
        print(f"  相对差异: {relative_diff:.6f}")
        print("-" * 40)
    
    # 统计结果
    final_relative_diff = max(all_relative_diffs)
    
    print("=" * 60)
    print("最终统计结果:")
    print(f"  最大相对差异: {final_relative_diff:.6f}")
    
    return final_relative_diff

# 运行测试
if __name__ == "__main__":
    # 创建模型
    model = QuantizationTestNet(num_classes=10)
    
    print("开始量化tanh精度测试...")
    print("=" * 60)

    final_relative_diff = test_quantization_in_forward(model, 10)

    print("\n" + "=" * 60)
    print("最终评估:")
    
    threshold = 0.001  # 阈值可以根据需要调整
    
    if final_relative_diff < threshold:
        print(f"✓ 量化tanh测试成功！最大差异 {final_relative_diff:.6f} < 阈值 {threshold}")
    else:
        print(f"✗ 量化tanh测试失败！最大差异 {final_relative_diff:.6f} >= 阈值 {threshold}")
    