# LNN工具链开发环境搭建指南

## 一、编译工具及CUDA/CUDNN版本配置

### 1. 查看显卡驱动（仅支持NVIDIA显卡）
使用以下命令查看显卡驱动信息，确认支持的最高CUDA版本：
```bash
nvidia-smi
```

**示例输出：**  
![NVIDIA_INFO](/doc/image/nvidia_info.png)

**注意：**
- 如需更高版本的CUDA，可升级显卡驱动至最新版本。
- 如果没有出现类似信息，说明使用集成显卡或者非N卡，可使用CPU版本进行体验（训练效率非常差），直接跳至步骤二，可按照CUDA 12.2对应的配置安装其他软件和依赖包。

### 2. 安装CUDA/CUDNN版本

#### 方式一：手动安装
1. **查看当前CUDA版本：**
   ```bash
   nvcc -V
   ```
   若未安装CUDA套件，将显示类似以下信息：  
   ![NVCC](/doc/image/nvcc_none.png)

2. **安装CUDA+CUDNN对应版本：**
   - 根据显卡驱动支持的最高CUDA版本，从[官方文档](https://developer.nvidia.com/cuda-toolkit-archive)下载并安装对应版本的CUDA。
   - 验证CUDA版本：
     ```bash
     nvcc -V
     ```
   - 验证cuDNN版本（Linux系统）：
     ```bash
     cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A
     ```
     若显示版本信息，则表示安装成功。

3. **安装GCC+CMAKE+Anaconda/Miniconda：**
   - GCC版本需与CUDA版本对应，参考下图选择合适的GCC版本：  
     ![cuda_gcc](/doc/image/cuda_gcc.png)
   - CMAKE版本推荐4.2.1或更高版本。
   - 安装Anaconda，建议使用最新版本。

#### 方式二：使用Docker镜像（推荐）
- 我们提供几种典型的CUDA版本匹配的Docker镜像，预先安装了特定版本的CUDA/CUDNN/GCC/CMAKE/Anaconda/python及依赖包。用户可直接下载使用，具体使用方法请参考第三章节。

---

## 二、本地构建Python开发环境
建议使用anaconda创建虚拟环境，与本地环境隔离

### 使用Anaconda创建虚拟环境
根据Python与CUDA版本的对应关系，指定Python版本创建虚拟环境。  
例如，在CUDA 12.2下推荐使用Python 3.10.0：
```bash
conda create -n linger_thinker_3.x python==3.10.0
conda activate linger_thinker_3.x
pip install -U pip
cat requirements.txt | xargs -n 1 pip install
```

**注意：** 项目默认配置为CUDA 12.2配套软件版本。也可不创建虚拟环境直接通过pip安装依赖项，安装内容和版本参考requirements.txt


---

## 三、Docker镜像安装方式

我们将提供四种镜像版本供大家使用(暂未开放)：
- CPU版本
- CUDA 11.2版本
- CUDA 12.2版本
- CUDA 12.4版本

### 1. Docker安装
若未安装Docker工具，请参考以下链接进行安装（建议使用CentOS系统）。若已安装，则直接进行权限验证：
- [Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Debian](https://docs.docker.com/engine/install/debian/)
- [CentOS](https://docs.docker.com/engine/install/centos/)
- [其他LINUX发行版](https://docs.docker.com/engine/install/binaries/)

安装完成后，运行以下命令查看版本，验证是否安装成功并有足够权限：
```bash
docker version
```

若出现"Got permission denied"权限报错，说明当前用户权限不够，需要添加权限：
```bash
sudo groupadd docker  # 添加docker用户组
sudo gpasswd -a $USER docker   # 将登录用户加入到docker用户组中
newgrp docker     # 更新用户组
docker ps    # 测试docker命令是否可以使用sudo正常使用
```

再次执行"docker version"命令，若不再出现权限报错，继续下一步。

### 2. 启动Docker服务
```bash
sudo systemctl start docker
```

### 3. 拉取镜像并加载
1. **拉取镜像**
   - CPU版本：
     ```bash
     docker pull listenai/linger:x.x.x
     ```
   - GPU版本（CUDA 11.2）：
     ```bash
     docker pull listenai/linger_gpu:x.x.x
     ```
   - GPU版本（CUDA 12.2）：
     ```bash
     docker pull listenai/linger_gpu:x.x.x
     ```

2. **运行容器**
   - CPU版本：
     ```bash
     docker container run -it listenai/linger:x.x.x /bin/bash
     ```
   - GPU版本（CUDA 11.2）：
     ```bash
     docker container run -it listenai/linger_gpu:x.x.x /bin/bash
     ```
   - GPU版本（CUDA 12.2）：
     ```bash
     docker container run -it listenai/linger_gpu:x.x.x /bin/bash
     ```

**注意：** x.x.x表示特定版本，也可默认安装最新版本。

若一切正常，运行上述命令后，将返回一个命令行提示符：
```bash
root@66d80f4aaf1e:/LISTENAI#
```

---

## 四、安装LNN开发套件

### 1. linger安装
```bash
git clone https://github.com/LISTENAI/linger.git
cd linger && sh install.sh
```

**注意：** 安装过程中会在线编译c++/cuda代码，时间较长。

### 2. thinker安装
```bash
git clone https://github.com/LISTENAI/thinker.git
cd thinker/tools && sh install.sh
```

### 3. 安装验证
```python
Python 3.12.10 | packaged by conda-forge | (main, Apr 10 2025, 22:21:13) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import linger
>>> 
```
支持显示版本信息，即可认为安装成功。

```python
Python 3.12.10 | packaged by conda-forge | (main, Apr 10 2025, 22:21:13) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tpacker  # 资源打包工具组件
>>> import tvalidator # 一致性校验组件
>>> import tprofile # 性能仿真组件
>>> 
```
支持显示版本信息，即可认为安装成功。

---

## 附录

### 1. conda环境管理常用指令
- 查看已创建的环境：
  ```bash
  conda info --env
  ```
- 查看当前环境中已安装的包：
  ```bash
  conda list
  ```
- 删除不必要的环境：
  ```bash
  conda activate base
  conda remove -n xxx --all
  ```

### 2. docker其他常用指令
- 查询容器ID：
  ```bash
  docker ps
  ```
- 从宿主机拷贝文件到容器：
  ```bash
  docker cp 要拷贝的文件路径 容器名：容器内部路径
  ```
  示例：
  ```bash
  docker cp model 2ef7893f06bc:linger
  ```
- 从容器拷贝文件到宿主机：
  ```bash
  docker cp 容器名：容器内部路径  宿主机路径
  ```
  示例：
  ```bash
  docker cp 2ef7893f06bc:/models /opt
  ```
- 终止运行的容器：
  ```bash
  docker container kill [containID]
  ```
- 移除停止的容器：
  ```bash
  docker container rm [containID]
  ```

---

**补充说明：**
- 如果在安装过程中遇到依赖问题，可以参考`requirements.txt`文件手动安装缺失包。
- 建议在安装前检查系统依赖（如libcudnn、libcuda等）是否已正确安装。
- 如果使用Docker镜像，可以直接在容器内运行安装命令，避免本地环境配置问题。
