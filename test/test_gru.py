import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================================
# 1. 定义一个最简单的GRU网络
# ==========================================================
class GRUNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=1, num_classes=2):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, h_n = self.gru(x)          # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]             # 取最后一个时间步的输出
        out = self.fc(out)              # [batch, num_classes]
        return out

# ==========================================================
# 2. 生成随机数据 (假数据用于验证功能)
# ==========================================================
def generate_data(num_samples=1000, seq_len=5, input_size=10, num_classes=2):
    X = torch.randn(num_samples, seq_len, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

# ==========================================================
# 3. 训练与测试流程
# ==========================================================
def train_and_test():
    # 参数配置
    input_size = 10
    hidden_size = 20
    num_classes = 2
    seq_len = 5
    num_epochs = 5
    batch_size = 32
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 数据
    X_train, y_train = generate_data(800, seq_len, input_size, num_classes)
    X_test, y_test = generate_data(200, seq_len, input_size, num_classes)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # 模型、优化器、损失函数
    model = GRUNet(input_size, hidden_size, num_classes=num_classes).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}%")

    # 量化配置
    import linger
    from linger.utils import FakeQuantMethod, QatMethod
    linger.QUANT_CONFIGS.quant_method = FakeQuantMethod.CUDA
    linger.QUANT_CONFIGS.quant_info.qat_method = QatMethod.MOM
    linger.QUANT_CONFIGS.quant_info.weight_bits = 8
    linger.QUANT_CONFIGS.quant_info.activate_bits = 8

    model = linger.init(model)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_loss:.4f}")

    # 测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    acc = correct / total * 100
    print(f"Test Accuracy: {acc:.2f}%")

# ==========================================================
# 4. 运行主程序
# ==========================================================
if __name__ == "__main__":
    train_and_test()
