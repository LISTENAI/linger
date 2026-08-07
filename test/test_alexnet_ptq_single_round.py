"""
AlexNet CIFAR-10 Quantization Evaluation Script

This script:
1. Loads the trained AlexNet model from alexnet_cifar10.pth
2. Applies cross_weight_equalization to the float model
3. Initializes quantization using linger.ini (yaml config)
4. Runs calibration with 100 CIFAR-10 samples for post-quantization
5. Compares quantized model accuracy with original float model
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


def test_per_class(model, test_loader, device, num_classes=10):
    """详细测试模型：打印每个类别的准确率"""
    model.eval()

    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(len(targets)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    print("\n" + "=" * 60)
    print("各类别测试准确率:")
    print("-" * 60)
    for i in range(num_classes):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"  {class_names[i]:12s}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

    correct = sum(class_correct)
    total = sum(class_total)
    overall_acc = 100.0 * correct / total
    print("-" * 60)
    print(f"  整体准确率: {overall_acc:.2f}% ({correct}/{total})")
    print("=" * 60)

    return overall_acc


def main():
    # 配置
    data_dir = "./data"  # 包含cifar-10-batches-py子目录
    model_path = "./alexnet_cifar10.pth"
    # config_path = "./mix_quant.yaml"
    batch_size = 400
    calibration_samples = 400

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("AlexNet CIFAR-10 量化评估")
    print("=" * 60)
    print(f"使用设备: {device}")
    print(f"数据路径: {data_dir}")
    print(f"模型路径: {model_path}")
    # print(f"配置文件: {config_path}")
    print(f"校准样本数: {calibration_samples}")
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

    # 准备校准数据(从训练集抽取100条)
    calib_loader = get_cifar10_dataloader(data_dir, batch_size=calibration_samples,
                                          train=True, num_samples=calibration_samples)

    # 获取校准数据
    calib_inputs, _ = next(iter(calib_loader))
    calib_inputs = calib_inputs.to(device)

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
        print(f"  ⚠️ 警告: CWE后模型准确率下降超过5%，模型可能有问题!")
    else:
        print(f"  ✓ CWE后模型准确率正常")
    print("-" * 60)

    from linger.config import QuantInfo
    QuantInfo.w_calibrate_name       = "histogram"
    QuantInfo.a_calibrate_name       = "histogram"


    # 进入校准模式，所有操作都在calibration上下文中进行
    with linger.calibration(model_cwe):
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

        # 走一遍前向，开始校准
        _ = model_quant(calib_inputs)

    print(f"  - 校准完成, 使用了 {calibration_samples} 条数据")
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

    # 7. 打印各类别准确率对比
    print("\n[7] 各类别详细对比:")
    float_per_class = test_per_class(model, test_loader, device)
    quant_per_class = test_per_class(model_quant, test_loader, device)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print("\n类别准确率对比:")
    print("-" * 50)
    print(f"{'类别':<12} {'浮点':>10} {'量化':>10} {'差值':>10}")
    print("-" * 50)

    model.eval()
    class_correct_float = [0] * 10
    class_total_float = [0] * 10
    class_correct_quant = [0] * 10
    class_total_quant = [0] * 10

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 浮点模型
            outputs_float = model(inputs)
            _, predicted_float = outputs_float.max(1)
            for i in range(len(targets)):
                label = targets[i].item()
                class_total_float[label] += 1
                if predicted_float[i] == label:
                    class_correct_float[label] += 1

            # 量化模型
            outputs_quant = model_quant(inputs)
            _, predicted_quant = outputs_quant.max(1)
            for i in range(len(targets)):
                label = targets[i].item()
                class_total_quant[label] += 1
                if predicted_quant[i] == label:
                    class_correct_quant[label] += 1

    for i in range(10):
        float_acc_i = 100.0 * class_correct_float[i] / class_total_float[i] if class_total_float[i] > 0 else 0
        quant_acc_i = 100.0 * class_correct_quant[i] / class_total_quant[i] if class_total_quant[i] > 0 else 0
        diff = quant_acc_i - float_acc_i
        print(f"{class_names[i]:<12} {float_acc_i:>9.2f}% {quant_acc_i:>9.2f}% {diff:>+9.2f}%")

    print("-" * 50)

    # 保存量化模型
    quant_model_path = "/train21/bitbrain/permanent/cgchu/00lingsi/00linger/linger3_x/test/singel_model_test/alexnet_cifar10_quant.pth"
    torch.save({
        'model_state_dict': model_quant.state_dict(),
        'config': {
            'num_classes': 10,
            'calibration_samples': calibration_samples,
            'float_acc': float_acc,
            'quant_acc': quant_acc
        }
    }, quant_model_path)
    print(f"\n量化模型已保存到: {quant_model_path}")

    print("\n" + "=" * 60)
    print("量化评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()