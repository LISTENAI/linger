"""
AlexNet CIFAR-10 Multi-Round Quantization Evaluation Script

This script tests multi-round calibration functionality:
1. Loads the trained AlexNet model from alexnet_cifar10.pth
2. Applies cross_weight_equalization to the float model
3. Initializes quantization using linger.ini (yaml config)
4. Runs multi-round calibration with CIFAR-10 samples for post-quantization
5. Compares quantized model accuracy with original float model

Multi-round calibration is transparent to the user - they just loop through
the calibration data and call model_quant(inputs) for each batch.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import linger


# ============================================
# 定义AlexNet模型 (适配CIFAR-10数据集)
# ============================================
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # 第一层块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层块
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三层块
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)

        # 第四层块
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)

        # 第五层块
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 用于通道匹配的1x1卷积（残差连接）
        self.res_conv1 = nn.Conv2d(192, 384, kernel_size=1)  # conv2→conv3
        self.res_conv2 = nn.Conv2d(384, 256, kernel_size=1)  # conv3→conv4

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # 第一块
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        # 第二块
        x2 = self.relu(self.conv2(x))
        x2_pooled = self.pool2(x2)

        # 第三块 + 残差连接
        x3 = self.relu(self.conv3(x2_pooled))
        res1 = self.res_conv1(x2_pooled)
        x3 = x3 + res1  # 残差连接

        # 第四块 + 残差连接
        x4 = self.relu(self.conv4(x3))
        res2 = self.res_conv2(x3)
        x4 = x4 + res2  # 残差连接

        # 第五块
        x5 = self.relu(self.conv5(x4))
        x5 = self.pool5(x5)

        # 分类器
        x_flat = torch.flatten(x5, 1)
        out = self.classifier(x_flat)
        return out


# ============================================
# AlexNet模型定义结束
# ============================================


def get_cifar10_dataloader(data_dir, batch_size=128, train=True, num_samples=None):
    """获取CIFAR-10数据加载器 (使用torchvision.datasets.CIFAR10)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # 使用torchvision.datasets.CIFAR10加载数据
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=False,
        transform=transform
    )

    # Limit number of samples if requested
    if num_samples is not None and num_samples < len(dataset):
        indices = list(range(num_samples))
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=True
    )

    return dataloader


# ============================================
# 数据加载器结束
# ============================================


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    # 配置
    data_dir = "./data"  # 包含cifar-10-batches-py子目录
    model_path = "./alexnet_cifar10.pth"
    # config_path = "./mix_quant.yaml"
    batch_size = 100  # 多轮校准每批大小
    calibration_samples = 400  # 总校准样本数
    num_rounds = 4  # 多轮校准轮数，每轮处理calibration_samples/num_rounds个样本

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("AlexNet CIFAR-10 多轮量化校准评估")
    print("=" * 60)
    print(f"使用设备: {device}")
    print(f"数据路径: {data_dir}")
    print(f"模型路径: {model_path}")
    # print(f"配置文件: {config_path}")
    print(f"校准样本数: {calibration_samples}")
    print(f"多轮校准轮数: {num_rounds} (每轮 {calibration_samples // num_rounds} 个样本)")
    print("-" * 60)

    # 1. 创建AlexNet模型并加载权重
    print("\n[1] 创建AlexNet模型并加载权重...")
    model = AlexNet(num_classes=10)

    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print(f"  - 模型加载成功")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - 总参数量: {total_params:,}")
    print("-" * 60)

    # 2. 测试原始浮点模型
    print("\n[2] 测试原始浮点模型...")
    test_loader = get_cifar10_dataloader(data_dir, batch_size=batch_size, train=False)
    criterion = nn.CrossEntropyLoss()
    float_loss, float_acc = evaluate(model, test_loader, criterion, device)
    print(f"  - 浮点模型测试准确率: {float_acc:.2f}%")
    print("-" * 60)

    # 3-4. 进入校准模式，应用Cross Weight Equalization、linger.init和权重加载
    print("\n[3-4] 进入校准模式，应用Cross Weight Equalization、linger.init和权重加载...")

    # 准备校准数据(从训练集抽取calibration_samples条，分多轮)
    calib_loader = get_cifar10_dataloader(data_dir, batch_size=calibration_samples // num_rounds,
                                          train=True, num_samples=calibration_samples)

    # 创建dummy input用于hook追踪
    dummy_input = torch.randn(1, 3, 32, 32).to(device)

    # 应用cross_weight_equalization
    from linger.quant.cross_weight_equalization import cross_weight_equalization
    model_cwe = cross_weight_equalization(model, dummy_input)
    print(f"  - Cross Weight Equalization 完成")

    # 测试 cross_weight_equalization 后的模型是否正常
    print("\n  [3.5] 测试 cross_weight_equalization 后模型是否正常...")
    cwe_loss, cwe_acc = evaluate(model_cwe, test_loader, criterion, device)
    print(f"  - CWE后模型测试准确率: {cwe_acc:.2f}%")
    if abs(float_acc - cwe_acc) > 5.0:
        print(f"  警告: CWE后模型准确率下降超过5%，模型可能有问题!")
    else:
        print(f"  - CWE后模型准确率正常")
    print("-" * 60)

    from linger.config import QuantInfo
    QuantInfo.w_calibrate_name = "histogram"
    QuantInfo.a_calibrate_name = "histogram"

    # 进入校准模式
    with linger.calibration(model_cwe) as ctx:
        # 使用linger.ini进行量化初始化
        model_quant = linger.init(model_cwe)#, config_file=config_path)
        print(f"  - 量化初始化完成")

        # 加载权重到量化模型
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model_quant.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model_quant.load_state_dict(checkpoint, strict=False)
        print(f"  - 权重加载完成")

        # 多轮校准：用户无感知，和单轮一样使用
        print(f"\n  开始多轮校准 (共{num_rounds}轮)...")
        round_idx = 0
        for inputs, _ in calib_loader:
            round_idx += 1
            inputs = inputs.to(device)
            _ = model_quant(inputs)  # 每轮只是累积数据
            print(f"    第{round_idx}轮完成, 累积 {round_idx * (calibration_samples // num_rounds)} 个样本")

    print(f"  - 多轮校准完成, 共使用 {calibration_samples} 条数据")
    print("-" * 60)

    # 5. 测试后量化模型
    print("\n[5] 测试后量化模型...")
    quant_loss, quant_acc = evaluate(model_quant, test_loader, criterion, device)
    print(f"  - 量化模型测试准确率: {quant_acc:.2f}%")
    print("-" * 60)

    # 6. 对比结果
    print("\n" + "=" * 60)
    print("量化评估结果对比")
    print("=" * 60)
    print(f"  浮点模型准确率:     {float_acc:.2f}%")
    print(f"  量化模型准确率:     {quant_acc:.2f}%")
    print(f"  准确率损失:         {float_acc - quant_acc:.2f}%")
    print(f"  相对误差:           {abs(float_acc - quant_acc) / max(float_acc, 0.01) * 100:.2f}%")
    print("=" * 60)

    # 保存量化模型
    quant_model_path = "/train21/bitbrain/permanent/cgchu/00lingsi/00linger/linger3_x/test/singel_model_test/alexnet_cifar10_quant_multiround.pth"
    torch.save({
        'model_state_dict': model_quant.state_dict(),
        'config': {
            'num_classes': 10,
            'calibration_samples': calibration_samples,
            'num_rounds': num_rounds,
            'float_acc': float_acc,
            'quant_acc': quant_acc
        }
    }, quant_model_path)
    print(f"\n量化模型已保存到: {quant_model_path}")

    print("\n" + "=" * 60)
    print("多轮量化评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()