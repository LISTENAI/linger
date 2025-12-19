import numpy as np
import struct
import math
def q_rsqrt_fixed(x_int, frac_bits=15, iterations=1):
    """
    纯整数定点版本快速倒平方根 (Fast Inverse Sqrt)
    输入:
        x_int: 定点整数（Q格式）
        frac_bits: 小数位数，比如Q15就是15
        iterations: 牛顿迭代次数 (1或2次)
    输出:
        y_int: 定点整数（Q格式），近似 1/sqrt(x)
    """
    if x_int <= 0:
        return 0  # 避免除0或负数开方

    # ===== 初始猜测 =====
    # 简化的初始估计：y0 = (1 << frac_bits) * (1/sqrt(x_real))
    # 由于我们没有浮点，这里用比例常数近似: y0 = (1 << (frac_bits*3//2)) // sqrt(x_int)
    # 实际上更快的做法是用经验公式：y0 ≈ C / x_int + B
    # 经过实验，C=0x5A82 (~0.7071 * 2^15), B=0 可得到较好初值
    # y_int = (1 << frac_bits)  # 初始猜测设为1.0
    # y_int = (1 << (frac_bits * 3 // 2)) // (x_int >> (frac_bits // 2))
    float_x = float(x_int) / (1 << frac_bits)
    f = struct.pack('f', float_x)        # float -> bytes
    i = struct.unpack('I', f)[0]       # bytes -> uint32 (unsigned int)
    # magic constant from original implementation
    i = 0x5f3759df - (i >> 1)
    # reinterpret bits as float again
    y = struct.unpack('f', struct.pack('I', i))[0]
    y_int = round(y * (1 << frac_bits))

    # Newton 迭代: y = y * (1.5 - 0.5 * x * y * y)
    threehalfs = (3 << (frac_bits - 1))  # 1.5 in Q format
    half = (1 << (frac_bits - 1))        # 0.5 in Q format

    for _ in range(iterations):
        # y^2
        y2 = (y_int * y_int) >> frac_bits
        # x * y^2
        xy2 = (x_int * y2) >> frac_bits
        # (1.5 - 0.5 * x * y^2)
        term = threehalfs - ((half * xy2) >> frac_bits)
        # y * term
        y_int = (y_int * term) >> frac_bits

    return y_int

import torch
def test_data(x, N):
    sum_x = torch.sum(x)
    sum_x2 = torch.sum(x * x)
    y = N * sum_x2 - sum_x * sum_x
    return y

# ================== 测试与验证 ==================
if __name__ == "__main__":
    x = torch.randint(1, 2, (1,))
    y1 = test_data(x, 10)
    y2 = test_data(x, 100)
    y3 = test_data(x, 1000)
    y4 = test_data(x, 10000)
    y5 = test_data(x, 100000)
    print(f"{y1},{y2},{y3},{y4},{y5}")
    # frac_bits = 15
    # # 输入一些Q15数 (对应0.25, 0.5, 1.0, 2.0)
    # x_real = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 10.0], dtype=np.float32)
    # x_int = np.round(x_real * (1 << frac_bits)).astype(np.int64)

    # results = []
    # for xi, xr in zip(x_int, x_real):
    #     y_int = q_rsqrt_fixed(xi, frac_bits=frac_bits, iterations=2)
    #     y_real = y_int / (1 << frac_bits)
    #     y_ref = 1.0 / np.sqrt(xr)
    #     err = abs(y_real - y_ref)
    #     results.append((xr, y_real, y_ref, err))

    # print(f"{'x_real':>8} {'y_fixed':>10} {'y_ref':>10} {'error':>10}")
    # for xr, y, ref, err in results:
    #     print(f"{xr:8.3f} {y:10.6f} {ref:10.6f} {err:10.6f}")
