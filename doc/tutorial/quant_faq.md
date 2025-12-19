## 通用注意事项
1. 两阶段训练：
- 浮点阶段学习率通常设高一点，定点阶段学习率设低一点

## 常见问题定位与解决
1. 环境安装问题
    问题表现：error while loading shared libraries: libmpfr.so.6: cannot open shared object file: No such file or directory
    解决方案：将系统中libmpfr.so.6拷贝到环境中，例如：cp libmpfr.so /home4/listenai/miniconda3/envs/linger3.0/lib

## 网络搭建推荐
1. 搭建网络时，若nn.Linear层后有bn层直连，推荐将此linear层改为等效的1*1的卷积，这样在最后推理实现时可以完成conv-bn的融合，加快推理效率
2. 若网络中conv-bn的直连结构较多，使用linger进行量化时，推荐使用intx的trace_layer设置，完成convbn的自动融合，导图时即只会有一个conv存在
3. 网络forward代码中推荐使用  tensor.transpose/ tensor.squeeze / tensor.unsqueeze 等写法，其等效的torch.transpose(temnsor, ...)等写法可能暂时不被linger识别，导图时会出现量化op浮点的断连情况
4. 网络定义中推荐使用 nn.Conv2d / nn.Linear/ nn.BatchNorm 等写法，其等效的在forward中直接调用的F.Conv2d等写法可能暂时不被 linger 识别，导图时会出现op未量化的断连情况