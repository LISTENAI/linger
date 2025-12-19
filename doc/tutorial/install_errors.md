
# 推荐环境安装
| Python      | PyTorch    | torchvision | CUDA Runtime | CUDA Toolkit | nvcc | GCC  | NumPy  | ONNX   | onnxruntime |
| ----------- | ---------- | ----------- | ------------ | ------------ | ---- | ---- | ------ | ------ | ----------- |
| **3.8**     | **1.9.1**  | 0.10.1      | 11.1         | 11.1         | 11.1 | 7.5  | 1.19.x | 1.10.x | 1.9.x       |
| **3.9**     | **1.12.1** | 0.13.1      | 11.6         | 11.6         | 11.6 | 8.4  | 1.21.x | 1.12.x | 1.13.x      |
| **3.10** ⭐  | **2.0.1**  | 0.15.2      | 11.7         | 11.7         | 11.7 | 9.4  | 1.23.5 | 1.14.1 | 1.15.1      |
| **3.11**    | **2.3.1**  | 0.18.1      | 12.1         | 12.1         | 12.1 | 11.3 | 1.26.x | 1.15.x | 1.17.x      |
| **3.12** ⚠️ | **2.6.0**  | 0.21.0      | 12.4         | 12.4         | 12.4 | 12.2 | 2.0.x  | 1.16.x | 1.18.x      |


# 常见错误及解决方案
* error while loading shared libraries: libmpfr.so.6: cannot open shared object file: No such file or directory
* cp libmpfr.so /home4/listenai/miniconda3/envs/linger3.0/lib
* export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home4/listenai/miniconda3/envs/linger3.0/lib
