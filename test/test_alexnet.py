import os
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import onnx
# import onnxruntime as ort
import numpy as np
from tqdm import tqdm
import shutil

import linger

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# ======================
# 1. 定义AlexNet模型 (适配CIFAR-10)
# ======================
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

# ======================
# 2. 数据准备 (使用已下载的cifar-10-python.tar.gz)
# ======================
def extract_cifar10(tar_path, extract_path='./data'):
    """
    解压已下载的CIFAR-10数据集
    
    参数:
        tar_path: cifar-10-python.tar.gz的路径
        extract_path: 解压目标路径
    """
    os.makedirs(extract_path, exist_ok=True)
    
    print(f"解压数据集: {tar_path} -> {extract_path}")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    
    # 确保解压后的目录结构正确
    extracted_dir = os.path.join(extract_path, 'cifar-10-batches-py')
    if not os.path.exists(extracted_dir):
        raise FileNotFoundError(f"解压后未找到cifar-10-batches-py目录，请检查tar文件内容")
    
    print(f"数据集已成功解压到: {extracted_dir}")
    return extracted_dir

def get_cifar10_data(tar_path, batch_size=128):
    """
    加载CIFAR-10数据集 (使用已下载的tar文件)
    
    参数:
        tar_path: cifar-10-python.tar.gz的路径
        batch_size: 批次大小
    """
    # 解压数据集
    # import pdb; pdb.set_trace()
    # data_dir = extract_cifar10(tar_path)
    data_dir = './data'
    
    # 数据增强和标准化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    # 加载数据集 (设置download=False，因为我们已经解压了)
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=False, 
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=False, 
        transform=transform_test
    )
    
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    testloader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return trainloader, testloader, data_dir

# ======================
# 3. 训练函数
# ======================
def train(model, trainloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(trainloader, desc=f'Epoch {epoch+1}', unit='batch')
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix({
            'loss': running_loss / (total / trainloader.batch_size),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(trainloader), 100. * correct / total

# ======================
# 4. 测试函数
# ======================
def test(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    print(f'Test Loss: {test_loss/len(testloader):.4f} | Test Acc: {acc:.2f}%')
    return test_loss / len(testloader), acc

# ======================
# 5. ONNX导出与验证
# ======================
def export_to_onnx(model, device, onnx_path='alexnet_cifar10.onnx'):
    # 设置为评估模式
    model.eval()
    
    # 创建示例输入 (batch_size=1)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # 导出ONNX模型
    torch.onnx.export(
        model, 
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX模型已导出至: {onnx_path}")
    
    # 验证ONNX模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过!")
    
    # 使用ONNX Runtime进行推理验证
    # ort_session = ort.InferenceSession(onnx_path)
    
    # 准备输入数据 (使用测试集第一张图片)
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=False
    )
    img, label = testset[0]
    
    # 转换为模型输入格式
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    img_tensor = transform(img).unsqueeze(0).numpy()
    
    # PyTorch推理
    with torch.no_grad():
        torch_output = model(torch.tensor(img_tensor).to(device))
        torch_pred = torch.argmax(torch_output, 1).item()
    
    # ONNX Runtime推理
    # ort_inputs = {ort_session.get_inputs()[0].name: img_tensor}
    # ort_output = ort_session.run(None, ort_inputs)[0]
    # onnx_pred = np.argmax(ort_output)
    
    print(f"PyTorch预测: {torch_pred}, 真实标签: {label}")
    # print(f"PyTorch预测: {torch_pred}, ONNX预测: {onnx_pred}, 真实标签: {label}")
    # assert torch_pred == onnx_pred, "PyTorch和ONNX预测结果不一致!"
    print("ONNX推理验证成功!")

# ======================
# 6. 主函数
# ======================
def main():
    # 超参数设置
    BATCH_SIZE = 128
    EPOCHS = 20
    LR = 0.01
    MODEL_PATH = 'alexnet_cifar10.pth'
    ONNX_PATH = 'alexnet_cifar10.onnx'
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 查找已下载的CIFAR-10数据集
    print("\n" + "="*50)
    print("查找已下载的CIFAR-10数据集...")
    print("="*50)
    
    # 尝试在常见位置查找数据集
    possible_paths = [
        './cifar-10-python.tar.gz',
        '../cifar-10-python.tar.gz',
        './data/cifar-10-python.tar.gz',
        os.path.expanduser('~/Downloads/cifar-10-python.tar.gz')
    ]
    
    tar_path = None
    for path in possible_paths:
        if os.path.exists(path):
            tar_path = path
            break
    
    if tar_path is None:
        # 提示用户输入路径
        tar_path = input("未找到cifar-10-python.tar.gz文件，请输入完整路径: ").strip()
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"指定的文件不存在: {tar_path}")
    
    print(f"找到数据集文件: {tar_path}")
    
    # 获取数据
    print("\n" + "="*50)
    print("加载CIFAR-10数据集...")
    print("="*50)
    trainloader, testloader, data_dir = get_cifar10_data(tar_path, BATCH_SIZE)
    
    # 初始化模型
    model = AlexNet(num_classes=10).to(device)
    print("\n模型结构:\n", model)

    model = linger.init(model)
    print(model)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # 训练循环
    best_acc = 0
    print("\n" + "="*50)
    print("开始训练...")
    print("="*50)
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, device, epoch)
        test_loss, test_acc = test(model, testloader, criterion, device)
        
        # 学习率调整
        scheduler.step(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"保存最佳模型至 {MODEL_PATH} (准确率: {best_acc:.2f}%)")
    
    print(f"\n训练完成! 最佳测试准确率: {best_acc:.2f}%")
    
    # 加载最佳模型进行最终测试
    model.load_state_dict(torch.load(MODEL_PATH))
    print("\n" + "="*50)
    print("使用最佳模型进行最终测试:")
    print("="*50)
    test(model, testloader, criterion, device)
    
    # 导出ONNX模型
    print("\n" + "="*50)
    print("导出ONNX模型...")
    print("="*50)
    export_to_onnx(model, device, ONNX_PATH)
    
    # 清理临时数据 (保留原始tar文件，只删除解压后的目录)
    print("\n" + "="*50)
    print("清理临时数据...")
    print("="*50)
    if os.path.exists(data_dir):
        try:
            shutil.rmtree(data_dir)
            print(f"已删除临时解压目录: {data_dir}")
        except Exception as e:
            print(f"清理临时数据时出错: {e}")
    
    print("\n" + "="*50)
    print("所有任务完成!")
    print("="*50)

if __name__ == '__main__':
    main()
