# Linger 后量化使用方法

## 概述

后量化（Post-Quantization）是指对已经训练好的浮点模型进行量化，而不参与训练过程。本文档介绍如何使用 linger 进行后量化操作。

---

## 后量化流程

1. **（可选）应用 Cross Weight Equalization**：对模型进行权重均衡处理，使量化更友好
2. **配置量化参数**：设置激活和权重的校准方法
3. **进入校准模式并初始化量化**：使用 `linger.calibration()` 和 `linger.init()`
4. **加载权重并校准**：将浮点权重加载到量化模型中，运行前向传播收集校准数据

---

## 接口说明

### 1. （可选）Cross Weight Equalization

**接口**：`linger.quant.cross_weight_equalization.cross_weight_equalization`

```python
from linger.quant.cross_weight_equalization import cross_weight_equalization

model_cwe = cross_weight_equalization(model, dummy_input)
```

**注意事项**：
- Cross Weight Equalization 是**可选步骤**，不添加也可以进行量化
- 该技术会对模型进行以下修改：
  - 将 `ReLU6` 激活函数替换为 `ReLU`
  - 进行 bias absorb 操作
- 由于这些修改，可能对模型精度产生一定影响
- 如果量化效果不佳，可以尝试启用 CWE 来改善

### 2. 配置量化参数

**接口**：`linger.config.QuantInfo`

```python
from linger.config import QuantInfo

QuantInfo.w_calibrate_name = "histogram"  # 权重校准方法
QuantInfo.a_calibrate_name = "histogram"  # 激活校准方法
```

可选校准方法及说明：

| 方法 | 说明 |
|------|------|
| `abs_max` | 取绝对值最大值的对数作为缩放因子，简单高效，是最常用的权重校准方法 |
| `top_10` | 取绝对值前10个最大值的阈值，适合激活值分布不均匀的情况 |
| `histogram` | TensorRT直方图校准方法，通过KL散度找到最优截断阈值，精度较高但计算耗时 |
| `w_like` | 权重专用方法，通过搜索找到最佳2的幂次系数作为缩放因子 |

### 3. 进入校准模式并进行量化

**接口**：`linger.calibration(model)`、`linger.init()`

`linger.calibration()` 是上下文管理器，支持传入模型对象。进入校准模式后：
- 支持多轮校准模式，数据在各轮前向传播中累积
- 退出上下文时自动调用所有量化器的 `finalize_calibration()` 计算最终 scale

```python
with linger.calibration(model_cwe) as ctx:
    model_quant = linger.init(model_cwe, config_file=config_path)
    model_quant.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # 单轮校准：一次性送入所有校准数据
    _ = model_quant(calib_inputs)

    # 多轮校准：分批次送入校准数据，用户无感知，和单轮一样使用
    for inputs, _ in calib_loader:
        _ = model_quant(inputs)
```

**注意**：
- `linger.init()` 需要在 `linger.calibration()` 上下文管理器内部调用
- 传入 `model_cwe` 后，退出上下文时自动 finalize 所有量化器

---

## 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│              （可选）Cross Weight Equalization               │
│         对模型更友好，但会改relu6为relu，可能影响精度          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    配置量化校准方法                           │
│              QuantInfo.w_calibrate_name                     │
│              QuantInfo.a_calibrate_name                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 linger.calibration(model) 上下文             │
│                   启用多轮校准累积模式                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  linger.init() 量化初始化                     │
│                  模型算子替换为量化算子                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   加载权重到量化模型                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    执行前向传播校准                           │
│               多轮前向自动累积数据                            │
│           退出上下文时自动finalize计算scale                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 配置文件说明

linger 使用 YAML 配置文件来控制量化行为，配置项说明见下表。

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `activate_bits` | 激活数据位宽 | 8 |
| `weight_bits` | 权重量化位宽 | 8 |
| `a_calibrate_name` | 激活校准方法 | `top_10` |
| `w_calibrate_name` | 权重校准方法 | `abs_max` |
| `is_symmetry` | 是否对称量化 | `true` |
| `is_perchannel` | 是否通道级量化 | `false` |
| `platform` | 目标平台 | `venusA` |

---

## 常见问题

### Q: 量化后精度下降太多怎么办？

1. 尝试启用 Cross Weight Equalization
2. 增加校准样本数量
3. 调整校准方法（如尝试 `histogram` 代替 `abs_max`）
4. 检查模型结构是否适合量化

### Q: Cross Weight Equalization 影响精度吗？

是的，CWE 会将 `ReLU6` 替换为 `ReLU` 并进行 bias absorb，这可能对某些模型产生精度影响。建议在使用前先对比 CWE 后的浮点模型精度是否有明显下降。

### Q: 校准需要多少样本？

通常 100-500 个样本即可，具体数量取决于模型复杂度和激活值分布。建议从一个较小的数值开始尝试，如效果不佳再增加样本数量。