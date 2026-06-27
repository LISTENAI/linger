# 如何使用Docker创建linger&thinker统一开发环境
## 1、安装docker
如果未安装docker工具，请参考以下链接进行安装（建议使用Centos系统）。如果已安装则直接进行权限验证
* [Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
* [Debian](https://docs.docker.com/engine/install/debian/)
* [Centos](https://docs.docker.com/engine/install/centos/)
* [其他 LINUX 发行版](https://docs.docker.com/engine/install/binaries/)
* [Windows](https://docs.docker.com/desktop/install/windows-install/)


请安装docker-19.03及以上版本。安装完成后，运行下面的命令查看版本，验证是否安装成功并有足够权限
```Shell
$ docker version
```
如果出现"Got permission denied"权限报错，说明当前用户权限不够，需要添加权限
docker 需要用户具有 sudo 权限，但不宜直接使用root用户进行操作  
为了避免每次命令都输入sudo，可以建立 docker 用户组并把用户加入  
```Shell
$ sudo groupadd docker  # 添加docker用户组
$ sudo gpasswd -a $USER docker   # 将登陆用户加入到docker用户组中
$ newgrp docker     # 更新用户组
$ docker ps    # 测试docker命令是否可以使用sudo正常使用
```
再次执行"docker version"命令，发现不再出现"Got permission denied"权限报错，继续下一步。
可通过修改/etc/docker/daemon.json文件设置从国内服务器获取镜像文件来进行加速
## 2、启动docker服务
```Shell
$ sudo systemctl start docker # systemctl 命令的用法
```

## 3、拉取镜像并加载  
目前提供的镜像为提供linger和thinker依赖的开发环境。可以根据自己的需要选择指定的docker环境。
根据自己的需要选择对应的镜像

* 基础环境使用的是Ubuntu 20.04 LTS 

```shell
docker pull listenai/linger_thinker_cpu:1.0.0	#纯cpu镜像
```
```shell
docker pull listenai/linger_thinker_cu118:1.0.0	#cuda11.8镜像
```
若电脑不支持cuda，或者没有训练需求可以选用CPU版本。镜像的体积更小

拉取镜像成功：
```shell
...
451b59821fee: Pull complete 
aac4069a8048: Pull complete 
c6ff8e4994ee: Pull complete 
Digest: sha256:88211fd30fd0146c6189a5d2fecd0293f3f89b26409ba80a314255b02d435df2
Status: Downloaded newer image for listenai/linger_thinker_cu118:v1.0.0
docker.io/listenai/linger_thinker_cu118:v1.0.0
```

> 拉取镜像的过程比较耗时，请耐心等待。

## 4、运行linger_thinker容器
以linger_thinker(cu118版本)为例。

运行示例中，通过参数`-v `把本地主机运行路径下的`./workspace`目录，挂载到linger_thinker容器的`/root/workspace`目录，可以很方便的进行文件同步和修改，您可以根据实际需要进行挂载。（需要提前将目录创建好，windows系统同理，对应好路径即可。）可以将下载的linger、thinker、工程源码放在该目录。

```shell
docker run -it --name linger_thinker_cu118 --gpus all -v ./workspace:/root/workspace listenai/linger_thinker_cu118:v1.0.0 bash 
```

运行指令执行结果：

```shell
(linger_thinker_env) root@d5964a88f77b:/# 
```

其中d5964a88f77b表示容器ID，这表示你已经在容器里面了，返回的提示符就是容器内部的 Shell 提示符。这里进入workspace目录。这里可以下载最新版本的linger、thinker，并进行安装。

默认已经创建好了conda环境（环境的名字为：linger_thinker_env），里面已经预装了指定cuda版本的torch、torchvision。

验证nvidia驱动、CUDA。
```
# 验证nvidia驱动是否可用
nvidia-smi
# 验证cuda是否可用
nvcc -V
# 验证python环境中的torch是否可用
python -c "import torch; print('Torch版本:', torch.__version__); print('CUDA是否可用:', torch.cuda.is_available())"
```


## 5、linger_thinker_env conda环境配置
linger、thinker使用的统一conda环境（环境的名字为：linger_thinker_env）已经创建好直接激活即可。
```
conda activate linger_thinker_env
```
本镜像已经预装特定版本的 CUDA、cuDNN、GCC、CMake、Anaconda、Python 和常用依赖。直接在该环境下安装linger和thinker即可。

准备源码：

将下载好的linger源码和thinker源码放至主机的./workspace中(对应的，源码就会放置容器中的/root/workspace),也可以直接clone。

源码clone
```
# 拉取linger代码
git clone https://github.com/LISTENAI/linger.git
# 拉取thinker代码
git clone https://github.com/LISTENAI/thinker.git
```

在容器中安装linger和thinker
如容器已关闭了，使用下面命令打开容器中的终端

```bash
# 启动容器
docker start linger_thinker_cu118
# 进入容器终端
docker exec -it linger_thinker_cu118 bash
```
在容器终端中，进行linger和thinker的安装

（1）安装linger
```
# 进入到linger目录下安装linger
cd linger
# 如果环境中已经安装了torch等，可以将 requirements.txt 中的相应包注释掉
pip install -r requirements.txt
sh install.sh
```

（2）安装thinker
```
# 进入到thinker目录的tools下安装thinker
cd thinker/tools
sh install.sh
```
（3）验证安装

通过下面的脚本验证
```
# 在ligner_thinker_env中打开python
conda activate linger_thinker_env
#打开python
python
# 使用python进行验证
>>> import linger
>>> import tpacker
>>> import tvalidator
>>> import tprofile
```
若无报错，即可认为 linger 和thinker安装成功。

## 6、其它可能的操作  
以下为docker常用指令，如有其它需要，请查阅docker的指令集
* 查询容器ID
不加'-a'则显示的是正在运行中的容器。

```Shell
$ docker ps -a
```

* 从宿主机拷文件到容器里面
docker cp [要拷贝的文件路径] [容器名：要拷贝到容器里面对应的路径]
```Shell
$ docker cp model 66d80f4aaf1e:/root/workspace/  
```

* 从容器里面拷文件到宿主机
docker cp [容器名：要拷贝的文件在容器里面的路径] [要拷贝到宿主机的相应路径]
```Shell
$ docker cp 66d80f4aaf1e:/models /opt
```

* 终止运行的容器文件
```Shell
$ docker stop [containID] 
```
* 移除停止的容器文件
```Shell
$ docker rm [containID] 
```
  
* 容器内部的退出

|  方式  			 							 |  结果       									 |  再次启动  |
| ----   										| ----        								  |----   |
|exit     									|退出后,容器退出但不销毁。		|docker start 容器名/容器id|
|ctrl + D     							|退出后,容器退出但不销毁。  |docker start 容器名/容器id|
|先按 ctrl + p,再按 ctrl + q |退出后,容器后台运行，ps能查到      |docker start 容器名/容器id| 
