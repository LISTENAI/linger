# LNN 工具链开发环境搭建指南

本文档用于搭建 Linger + Thinker 统一开发环境。Linger 负责模型量化训练与导出，Thinker 负责离线分析、资源打包、仿真验证、性能评估和端侧执行。

> **重要说明**：本文默认描述的是 **Linger + Thinker 统一环境**。如果只使用 Thinker（例如仅运行 `tpacker`、`tvalidator`、`tprofile` 或编译执行器），不需要安装 CUDA、cuDNN、PyTorch、TorchVision、TorchAudio 及其相关依赖包。

## 一、环境类型说明

根据使用场景选择对应环境，避免安装不必要的依赖。

| 使用场景 | 需要 CUDA/cuDNN | 需要 PyTorch 相关依赖 | 典型用途 |
| --- | --- | --- | --- |
| Linger + Thinker 完整环境 | GPU 训练建议安装 | 需要 | 量化训练、模型导出、Thinker 离线打包和验证 |
| Linger CPU 体验环境 | 不需要 | 需要 CPU 版 PyTorch | 功能体验、小模型调试；训练效率较低 |
| Thinker-only 环境 | 不需要 | 不需要 | 仅使用 Thinker 工具链或编译/集成执行器 |

### Thinker-only 最小依赖

仅使用 Thinker 时，建议准备以下基础组件：

- Python 环境：推荐使用 Anaconda/Miniconda 创建独立虚拟环境。
- Python 包：`onnx` 会随 Thinker Python 工具安装；如遇依赖缺失，再按提示补充安装。
- 编译工具：如需编译仿真平台或执行器，需要安装 `gcc/g++`、`cmake`、`make` 等基础编译工具。
- 不需要安装：CUDA、cuDNN、PyTorch、TorchVision、TorchAudio。

> `requirements.txt` 面向 Linger + Thinker 统一环境，包含 `torch`、`torchvision`、`torchaudio` 等量化训练依赖。Thinker-only 场景不要直接执行 `cat requirements.txt | xargs -n 1 pip install`，否则会额外安装 PyTorch 相关包。

## 二、CUDA/cuDNN 与编译工具配置

本章节仅适用于需要 Linger GPU 量化训练的完整环境。如果只使用 Thinker，请跳过本章节，直接查看 [第三章](#三本地构建-python-开发环境)。

### 1. 查看显卡驱动

Linger GPU 训练环境需要 NVIDIA 显卡和匹配的驱动。使用以下命令查看显卡驱动信息，并确认支持的最高 CUDA 版本：

```bash
nvidia-smi
```

示例输出：

![NVIDIA_INFO](/doc/image/nvidia_info.png)

注意事项：

- 如需更高版本 CUDA，可升级 NVIDIA 显卡驱动。
- 如果没有输出类似信息，说明当前机器未安装 NVIDIA 驱动、使用集成显卡或使用非 NVIDIA 显卡。此时可使用 CPU 环境体验 Linger，或使用 Thinker-only 环境。

### 2. 安装 CUDA/cuDNN

#### 方式一：手动安装

1. 查看当前 CUDA 版本：

   ```bash
   nvcc -V
   ```

   如果未安装 CUDA Toolkit，将显示类似以下信息：

   ![NVCC](/doc/image/nvcc_none.png)

2. 安装 CUDA 和 cuDNN：

   - 根据显卡驱动支持的最高 CUDA 版本，从 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) 下载并安装对应版本。
   - 验证 CUDA：

     ```bash
     nvcc -V
     ```

   - 验证 cuDNN（Linux）：

     ```bash
     cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A
     ```

     若显示版本信息，则表示安装成功。

3. 安装 GCC、CMake 与 Anaconda/Miniconda：

   - GCC 版本需与 CUDA 版本匹配，可参考下图选择：

     ![cuda_gcc](/doc/image/cuda_gcc.png)

   - CMake 推荐使用 4.2.1 或更高版本。
   - 推荐安装 Anaconda 或 Miniconda，并为项目创建独立环境。

#### 方式二：使用 Docker 镜像

Docker 镜像会预装特定版本的 CUDA、cuDNN、GCC、CMake、Anaconda、Python 和常用依赖，适合避免本地环境冲突。具体方式见 [第四章](#四docker-镜像安装方式)（新版本暂不支持）。

## 三、本地构建 Python 开发环境

建议使用 Anaconda/Miniconda 创建虚拟环境，与系统 Python 环境隔离。

### 1. Linger + Thinker 统一环境

完整环境用于量化训练和 Thinker 部署全流程。根据 Python 与 CUDA 版本对应关系创建虚拟环境，例如 CUDA 12.2 环境推荐使用 Python 3.10：

```bash
conda create -n linger_thinker_3x python=3.10
conda activate linger_thinker_3x
pip install -U pip
cat requirements.txt | xargs -n 1 pip install
```

说明：

- `requirements.txt` 中包含 PyTorch、TorchVision、TorchAudio、ONNX 等依赖，适合 Linger + Thinker 统一环境。
- 项目默认配置偏向 CUDA 12.2 配套软件版本；如使用其它 CUDA 版本，请同步调整 PyTorch 与 CUDA 版本匹配关系。

### 2. Thinker-only 环境

如果只使用 Thinker，不需要安装 CUDA、cuDNN、PyTorch、TorchVision、TorchAudio。可使用更轻量的环境：

```bash
conda create -n thinker_3x python=3.10
conda activate thinker_3x
pip install -U pip setuptools wheel
```

然后安装 Thinker 工具：

```bash
git clone https://github.com/LISTENAI/thinker.git
cd thinker/tools
sh install.sh
```

安装完成后，可直接使用 Thinker Python 工具：

```bash
tpacker --help
tvalidator --help
tprofile --help
```

如仅编译和集成 C 执行器，还需要按目标平台准备对应的 C/C++ 编译工具链。

## 四、Docker 镜像安装方式

我们将提供几种典型镜像版本供使用（暂未开放）：

- CPU 版本
- CUDA 11.2 版本
- CUDA 12.2 版本
- CUDA 12.4 版本

### 1. 安装 Docker

若未安装 Docker，请参考以下链接进行安装。若已安装，可直接进行权限验证。

- [Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Debian](https://docs.docker.com/engine/install/debian/)
- [CentOS](https://docs.docker.com/engine/install/centos/)
- [其他 Linux 发行版](https://docs.docker.com/engine/install/binaries/)

安装完成后，运行以下命令查看版本，验证 Docker 是否安装成功并具备足够权限：

```bash
docker version
```

如果出现 `Got permission denied` 权限错误，说明当前用户权限不足，可添加 Docker 用户组权限：

```bash
sudo groupadd docker
sudo gpasswd -a $USER docker
newgrp docker
docker ps
```

再次执行 `docker version`，若不再出现权限错误，即可继续下一步。

### 2. 启动 Docker 服务

```bash
sudo systemctl start docker
```

### 3. 拉取镜像并运行容器

拉取镜像：

```bash
# CPU 版本
docker pull listenai/linger:x.x.x

# GPU 版本（CUDA 11.2 / 12.2 等）
docker pull listenai/linger_gpu:x.x.x
```

运行容器：

```bash
# CPU 版本
docker container run -it listenai/linger:x.x.x /bin/bash

# GPU 版本
docker container run -it listenai/linger_gpu:x.x.x /bin/bash
```

`x.x.x` 表示具体版本号，也可默认安装最新版本。若一切正常，运行后将进入容器命令行，例如：

```bash
root@66d80f4aaf1e:/LISTENAI#
```

## 五、安装 LNN 开发套件

### 1. 安装 Linger

Linger 仅在需要量化训练或模型导出时安装。Thinker-only 场景可跳过本步骤。

```bash
git clone https://github.com/LISTENAI/linger.git
cd linger
sh install.sh
```

注意：安装过程中会在线编译 C++/CUDA 代码，耗时较长；若使用 CPU 环境或不同 CUDA 版本，请确认 PyTorch 与本机环境匹配。

### 2. 安装 Thinker

```bash
git clone https://github.com/LISTENAI/thinker.git
cd thinker/tools
sh install.sh
```

### 3. 安装验证

验证 Linger（仅完整环境需要）：

```python
>>> import linger
```

若无报错，即可认为 Linger 安装成功。

验证 Thinker：

```python
>>> import tpacker      # 资源打包工具组件
>>> import tvalidator   # 一致性校验组件
>>> import tprofile     # 性能仿真组件
```

若无报错，即可认为 Thinker 工具安装成功。也可使用命令行验证：

```bash
tpacker --help
tvalidator --help
tprofile --help
```

## 附录

### 1. Conda 环境管理常用命令

查看已创建的环境：

```bash
conda info --env
```

查看当前环境中已安装的包：

```bash
conda list
```

删除不需要的环境：

```bash
conda activate base
conda remove -n xxx --all
```

### 2. Docker 常用命令

查询容器 ID：

```bash
docker ps
```

从宿主机拷贝文件到容器：

```bash
docker cp 要拷贝的文件路径 容器名:容器内部路径
```

示例：

```bash
docker cp model 2ef7893f06bc:linger
```

从容器拷贝文件到宿主机：

```bash
docker cp 容器名:容器内部路径 宿主机路径
```

示例：

```bash
docker cp 2ef7893f06bc:/models /opt
```

终止运行的容器：

```bash
docker container kill [containID]
```

移除停止的容器：

```bash
docker container rm [containID]
```

## 补充说明

- 如果在安装过程中遇到依赖问题，可参考 `requirements.txt` 手动安装缺失包；Thinker-only 场景应优先只补充实际缺失的 Thinker 依赖，避免安装 PyTorch 相关包。
- Linger GPU 环境需要检查 `libcuda`、`libcudnn` 等系统依赖是否正确安装。
- Thinker-only 环境更适合 CI、模型打包、仿真验证和执行器集成，不依赖 NVIDIA GPU。
- 如果使用 Docker 镜像，可直接在容器内运行安装命令，减少本地环境配置问题。
