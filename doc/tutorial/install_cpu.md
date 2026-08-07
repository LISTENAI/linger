# LNN CPU版本安装指南

本文档介绍如何在没有NVIDIA GPU或CUDA环境的机器上安装LNN的CPU版本。

## 一、适用场景

- 使用集成显卡或AMD显卡
- 无需GPU加速，仅做功能验证或开发测试
- 教学演示或轻量级体验

**注意：** CPU版本训练效率较低，建议仅用于开发测试。生产训练请使用GPU版本。

---

## 二、环境准备

### 1. 安装Python

建议使用Anaconda管理Python环境，推荐Python 3.8 ~ 3.11版本。

### 2. 安装编译工具

CPU版本需要以下编译工具：

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ cmake

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake
```

验证安装：
```bash
gcc --version
cmake --version
```

### 3. 安装PyTorch

```bash
pip install torch torchvision 
```

验证PyTorch安装成功：
```bash
python3 -c "import torch; print(torch.__version__)"
```

---

## 三、创建虚拟环境（推荐）

建议使用Anaconda创建独立的虚拟环境：

```bash
conda create -n linger_cpu python==3.10.0
conda activate linger_cpu
pip install -U pip
pip install torch torchvision 
```

---

## 四、安装LNN CPU版本

### 1. 下载源码

```bash
git clone https://github.com/LISTENAI/linger.git
cd linger
```

### 2. 执行安装

通过设置环境变量 `LINGER_CPU_ONLY=1` 强制使用CPU模式编译：

```bash
LINGER_CPU_ONLY=1 sh install.sh
```

install.sh脚本会执行以下操作：
- 自动检测系统环境（当`LINGER_CPU_ONLY`已设置时跳过检测）
- 编译C++扩展（仅编译CPU内核，不编译CUDA内核）
- 安装Python包

**注意：** 编译过程需要数分钟，请耐心等待。

### 3. 环境变量说明

| 环境变量 | 说明 |
|---------|------|
| `LINGER_CPU_ONLY` | 设置为`1`则强制CPU模式编译；不设置则脚本自动检测 |
| `MAX_JOBS` | 编译并行任务数，默认为CPU核心数 |
| `PYTHON` | 指定Python解释器路径，默认为`python3` |

**自动检测逻辑：** 当`LINGER_CPU_ONLY`未设置时，脚本会检查：
- PyTorch是否支持CUDA（`torch.version.cuda`是否为空）
- 系统是否安装`nvcc`编译器

若两者任一不满足，则自动启用CPU模式。

### 4. 编译选项示例

```bash
# 使用8个并行任务编译
MAX_JOBS=8 LINGER_CPU_ONLY=1 sh install.sh

# 使用指定Python解释器
PYTHON=/path/to/python LINGER_CPU_ONLY=1 sh install.sh
```

---

## 五、安装验证

安装完成后，在Python中验证：

```python
Python 3.10.0 | (main, ...) [GCC x.x.x] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import linger
>>> import linger
```

若能成功导入`linger`模块而无报错，说明安装成功。

查看版本信息：
```python
>>> import linger
>>> print(linger.__version__)  # 如有版本属性
```

---

## 六、CPU版本已知限制

CPU版本存在以下功能限制：

| 功能 | 状态 | 说明 |
|------|------|------|
| 量化算子（CPU） | 正常 | `venus_qsoftmax`、`venus_qsigmoid`、`qlayernorm`等可用 |
| 量化算子（GPU） | 不可用 | 尝试调用GPU算子会抛出异常 |
| 伪量化（fake_quant） | 不可用 | 仅支持GPU模式 |
| 量化感知训练 | 部分支持 | 仅支持NATIVE模式 |

**GPU算子调用示例：** 当在CPU模式下调用GPU专用算子时，会抛出以下错误：
```
RuntimeError: lingerext was built in CPU-only mode; CUDA is unavailable.
```

---

## 七、常见问题

### 1. 编译报错：找不到torch/extension.h

确保已正确安装PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. 编译报错：gcc版本过低

建议使用gcc 9及以上版本：
```bash
# Debian/Ubuntu
sudo apt-get install -y gcc-9 g++-9
export CC=gcc-9 CXX=g++-9
```

### 3. 导入时报错：libtorch相关错误

检查PyTorch与libtorch版本是否匹配，尝试重新安装PyTorch。

---

## 八、卸载

```bash
pip uninstall linger
```

如需彻底清除编译产物：
```bash
rm -rf build/ dist/ *.egg-info/
```