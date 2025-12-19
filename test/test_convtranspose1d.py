import torch
import torch.nn as nn
import numpy as np

import linger
from linger.quant.ops.qmodule import *
from linger.config import QUANT_CONFIGS

q_configs = QUANT_CONFIGS

class QuantizationTestNet(nn.Module):
    def __init__(self, num_classes=10):
        super(QuantizationTestNet, self).__init__()

        self.convtranspose1d_float = nn.ConvTranspose1d(
            in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1         
        )

        self.convtranspose1d_quant = QConvTranspose1d(
            in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1,      
            weights_cfg=q_configs.quant_info.to_dict(), 
            activations_cfg=q_configs.quant_info.to_dict(),
            bias_cfg=q_configs.quant_info.to_dict(), 
            constrain =  q_configs.clamp_info.to_dict()   
        )
        
        with torch.no_grad():
            self.convtranspose1d_float.weight.copy_(self.convtranspose1d_quant.qweight)
            if self.convtranspose1d_quant.qbias is not None:
                self.convtranspose1d_float.bias.copy_(self.convtranspose1d_quant.qbias)

    def forward(self, x):
        # x = x.view(2, 3, -1)  
        result_float = self.convtranspose1d_float(x)
        result_quant = self.convtranspose1d_quant(x)
     
        return {
            'result_float': result_float,
            'result_quant': result_quant
        }

def test_quantization_in_forward(model):
    print("测试模型前向传播中的量化convtranspose1d比较...")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # batch_size = 2
    # x = torch.rand(batch_size, 3, 32, 32).to(device)
    x = torch.randn(4, 32, 50).to(device)

    with torch.no_grad():
        model = model.to(device)
        outputs = model(x)

    float_result = outputs['result_float']
    quant_result = outputs['result_quant']

    diff = torch.abs(float_result) - torch.abs(quant_result)
    mean_float = torch.mean(float_result).item()
    mean_diff = torch.mean(diff).item()
    relative_diff = mean_diff/mean_float

    return relative_diff

def check_conv_gradients(model, input_tensor):
    model.train()
    model.zero_grad()
    output_dict = model(input_tensor)
    
    for key, output_value in output_dict.items():
        target = torch.randn_like(output_value)
        loss = torch.nn.functional.mse_loss(output_value, target)
    
    loss.backward()
    
    print("=== 卷积层梯度检查 ===")
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output_value.shape}")
    print(f"损失值: {loss.item():.6f}\n")

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            
            is_zero = torch.allclose(grad, torch.zeros_like(grad), atol=1e-8)
            grad_norm = grad.norm().item()
            
            if 'convtranspose1d_float' in name and 'convtranspose1d_quant' not in name:
                layer_type = "convtranspose1d"
            elif 'convtranspose1d_quant' in name:
                layer_type = "qconvtranspose1d"
            else:
                continue
                
            status = "✓ 全0" if is_zero else "✗ 非0"
            print(f"{layer_type:8} {name:15} | {status} | 梯度范数: {grad_norm:.2e}")
    
    return loss.item()

if __name__ == "__main__":
    model = QuantizationTestNet(num_classes=10)

    # input_tensor = torch.randn(2, 3, 32, 32)
    # input_tensor = torch.randn(4, 32, 50)
    # loss = check_conv_gradients(model, input_tensor)
    
    print("开始量化convtranspose1d精度测试...")
    print("=" * 60)

    final_relative_diff = test_quantization_in_forward(model)
    final_relative_diff = torch.abs(torch.tensor(final_relative_diff)).item()

    print("\n" + "=" * 60)
    print("最终评估:")
    
    threshold = 0.001  
    
    if final_relative_diff < threshold:
        print(f"✓ 量化convtranspose1d测试成功！最大差异 {final_relative_diff:.6f} < 阈值 {threshold}")
    else:
        print(f"✗ 量化convtranspose1d测试失败！最大差异 {final_relative_diff:.6f} >= 阈值 {threshold}")