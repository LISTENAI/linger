## 通过源码直接安装
### x86-linux平台

推荐使用conda创建新环境，并使用如下版本的python依赖：
- python 3.7.0
- torch: 1.9.0
- torchvision: 0.10.0
- onnx: 1.7.0
- protobuf: 3.8.0

一键安装
``` sh
git clone https://github.com/LISTENAI/linger.git
cd linger
sh install.sh
```

### 其他平台
即将推出

## 验证安装
``` python
Python 3.7.3 (default, Jul 8 2020, 22:11:17)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import linger
>>> 
```

## 通过pip安装
``` shell
pip install pylinger
```

## 通过docker进行安装
您可以直接从[Docker Hub](https://hub.docker.com/)上拉取我们制作好的镜像，也可以使用我们提供的[Dockerfile](https://github.com/LISTENAI/linger/blob/main/Dockerfile)来进行镜像的制作
### 使用我们的提供的镜像
在 hub.docker.com 上注册登录后, 执行命令, 输入用户名和密码
```shell
docker login 
```

登录成功后 , 拉取已经制作好的 image 到本地。
```shell
docker pull littlezhan:linger:0.9.0
```

运行容器
```shell
docker container run -it littlezhan:linger:0.9.0 /bin/bash
```

如果一切正常，运行上面的命令以后，就会返回一个命令行提示符。
```shell
root@66d80f4aaf1e:/linger#
```

这表示你已经在容器里面了，返回的提示符就是容器内部的 Shell 提示符。能够执行命令。



### 自己制作镜像
使用本地 Dockerfile , 创建 image 文件，时间较长，请耐心等待
``` shell
docker image build -t linger:0.1.0 . (.表示当前路径)
```
从 image 文件生成容器
```shell
docker container run -it thinker:0.1.0 /bin/bash
```
这样就进入了容器中