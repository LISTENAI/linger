import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import math
import copy
import pandas as pd
import gc

# import debugpy
# print("localhost start ----------------------------")
# debugpy.listen(("localhost", 6000))
# debugpy.wait_for_client()

import linger
from linger.quant.ops.qmodule import *
from linger.config import QUANT_CONFIGS
from linger.utils import *

linger.QUANT_CONFIGS.quant_info.qat_method = QatMethod.TQT
q_configs = QUANT_CONFIGS

class Configure(Singleton):
    open_quant   = True
    quant_method = FakeQuantMethod.NATIVE
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype        = torch.float32
    seed         = 42
    input        = None # 测试算例，保证使用不同方式测试时数据都相同
    state_dict   = None # 测试模型的state_dict，保证使用不同方式测试时模型的所有参数都相同

compare_config = Configure()

def clear_all_cache():
    
    del compare_config.input
    del compare_config.state_dict
    compare_config.input = None
    compare_config.state_dict = None

    """
    清空 PyTorch 的所有缓存，包括：
    1. CUDA 显存缓存
    2. Autograd 图缓存
    3. torch.compile 生成的缓存 (Dynamo/Inductor)
    4. Python 垃圾回收
    """
    # 1. 删除无用的 Python 对象
    gc.collect()
    
    # 2. 清空 CUDA 显存缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # 清理跨进程内存
    
    # 3. 清空 Autograd 统计信息
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    
    # 4. 清空 torch.compile 相关缓存
    try:
        torch._dynamo.reset()
    except Exception:
        print("inductor cache 清理失败:", e)
    
    # try:
    #     torch._inductor.clear_cache()
    # except Exception:
    #     print("inductor cache 清理失败:", e)
    torch.cuda.synchronize()
    print("✅ 已清空所有 PyTorch 缓存")


def time_it(fn, input, repeat=200):
    for i in range(20):
        y = fn(input[i])
    torch.cuda.synchronize()
    start = time.time()
    for i in range(repeat):
        y = fn(input[i])
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) * 1000  # ms

def test(test_nums=200, fix_seq_len=True, batch_size = 40, seq_len=200, input_shape=256, output_shape=768, mode="float", is_train=True):
    if mode == "float":
        print("重新初始化了")
        input = []
        if fix_seq_len:
            for i in range(test_nums):
                input.append(torch.randn(batch_size, seq_len, input_shape).to(compare_config.device).to(compare_config.dtype))
        else:
            for i in range(test_nums):
                temp = random.randint(math.floor(seq_len * 0.9), math.ceil(seq_len * 1.1))
                input.append(torch.randn(batch_size, temp, input_shape).to(compare_config.device).to(compare_config.dtype))
        compare_config.input = copy.deepcopy(input)
        layer = nn.Sequential(torch.nn.Linear(input_shape, output_shape),
                              torch.nn.Linear(output_shape, input_shape),
                            )
        compare_config.state_dict = None
        compare_config.state_dict = copy.deepcopy(layer.state_dict())
    elif mode=="cuda":
        linger.QUANT_CONFIGS.quant_method = FakeQuantMethod.CUDA
        layer = nn.Sequential(
            QLinear(
                input_shape, 
                output_shape,
                weights_cfg=q_configs.quant_info.to_dict(), 
                activations_cfg=q_configs.quant_info.to_dict(),
                bias_cfg=q_configs.quant_info.to_dict(), 
                constrain =  q_configs.clamp_info.to_dict()   
            ),
            QLinear(
                output_shape, 
                input_shape,
                weights_cfg=q_configs.quant_info.to_dict(), 
                activations_cfg=q_configs.quant_info.to_dict(),
                bias_cfg=q_configs.quant_info.to_dict(), 
                constrain =  q_configs.clamp_info.to_dict()   
            )
        )
        layer.load_state_dict(compare_config.state_dict, strict=False)
    else:
        linger.QUANT_CONFIGS.quant_method = FakeQuantMethod.NATIVE
        layer = nn.Sequential(
            QLinear(
                input_shape, 
                output_shape,
                weights_cfg=q_configs.quant_info.to_dict(), 
                activations_cfg=q_configs.quant_info.to_dict(),
                bias_cfg=q_configs.quant_info.to_dict(), 
                constrain =  q_configs.clamp_info.to_dict()   
            ),
            QLinear(
                output_shape, 
                input_shape,
                weights_cfg=q_configs.quant_info.to_dict(), 
                activations_cfg=q_configs.quant_info.to_dict(),
                bias_cfg=q_configs.quant_info.to_dict(), 
                constrain =  q_configs.clamp_info.to_dict()   
            )
        )
        layer.load_state_dict(compare_config.state_dict, strict=False)
    layer.to(compare_config.device)
    if is_train:
        layer = layer.train()
    else:
        layer = layer.eval()

    time_ms = time_it(layer, compare_config.input, test_nums)
    print(f"test_nums:{test_nums}, fix_seq_len:{fix_seq_len}, batch_size:{batch_size}, seq_len:{seq_len}, input_shape:{input_shape}, output_shape:{output_shape}, time_ms:{time_ms}")
    # print(input[10].shape)
    output = layer(compare_config.input[10])
    torch.cuda.synchronize()

    del layer
    torch.cuda.empty_cache()
    return output, time_ms

def log_all(df, test_nums=200, fix_seq_len=True, batch_size = 40, seq_len=200, input_shape=256, output_shape=768):
    res_all = [] # 用于对比输出的一致性,仅挑选了第10个输入对比
    
    # float
    print("---------float---------")
    float_res, float_time = test(test_nums, fix_seq_len, batch_size, seq_len, input_shape, output_shape, mode="float")

    # cuda
    print("---------cuda---------")
    cuda_res, cuda_time = test(test_nums, fix_seq_len, batch_size, seq_len, input_shape, output_shape, mode="cuda")
    res_all.append(cuda_res)

    # native
    print("---------native---------")
    native_res, native_time = test(test_nums, fix_seq_len, batch_size, seq_len, input_shape, output_shape, mode="native")
    res_all.append(native_res)

    # native
    print("---------inference---------")
    infer_res, infe_time = test(test_nums, fix_seq_len, batch_size, seq_len, input_shape, output_shape, mode="cuda", is_train=False)
    # res_all.append(infer_res)

    for i in range(1, len(res_all)):
        diff_nums = (res_all[0].data != res_all[i].data).to(torch.float).sum()
        if  diff_nums !=0 :
            print(f"第{i}个和native结果不同，不同个数为：{diff_nums}")
            import pdb; pdb.set_trace()

    # df.loc[len(df)] = [test_nums, fix_seq_len, batch_size, seq_len, input_shape, output_shape, float_time, native_time, triton_time, cuda_time, compile_time]
    df.loc[len(df)] = [test_nums, fix_seq_len, batch_size, seq_len, input_shape, output_shape, float_time, cuda_time, native_time, infe_time]
    clear_all_cache()
    return df

if __name__ == "__main__":
    test_nums = 200
    fix_seq_len = False
    batch_size = [10, 20, 40]
    seq_len    = [10, 20, 40, 80, 160, 200]
    input_shape = [256, 512, 1024, 2048]
    output_shape = [256, 512, 1024, 2048]

    # batch_size = [10, 20]
    # seq_len    = [10, 20, ]
    # input_shape = [256, 512]
    # output_shape = [256, 512]

    columns = [
        "test_nums", "fix_seq_len", "batch_size", "seq_len",
        "input_shape", "output_shape", 
        "float_time",  "cuda_time","native_time", "inference", # "compile_time",  "native_time", "triton_time",
    ]

    df = pd.DataFrame(columns=columns)

    for i in batch_size:
        for j in seq_len:
            for k in input_shape:
                for g in output_shape:
                    print(f"正在计算：{i},{j},{k},{g}")
                    df = log_all(df=df, test_nums=test_nums, fix_seq_len=fix_seq_len, batch_size = i, seq_len=j, input_shape=k, output_shape=g)
    df.to_excel("/yrfs4/inference/sqtu2/LLM/code/linger3.0/my_linger/analys_linger.xlsx", index=False)
