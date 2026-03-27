
---

### **Linger量化训练使用方法**

#### **量化训练概述**

1. **直接量化训练**：
   - 基于浮点模型进行量化训练，学习率需适当减小（通常比浮点学习率小一个数量级）。
   - 初始loss可能较大，但随着训练迭代会逐渐降低并趋于稳定。
   - 如果loss波动较大，可尝试降低学习率。

2. **两阶段量化训练**：
   - **第一阶段**：浮点约束训练
     - 调用linger的约束接口对网络进行约束，可从头训练或基于已收敛的浮点模型。
     - 约束参数包括权重和激活值，通常约束到[-8, 8]。
   - **第二阶段**：定点量化微调
     - 调用linger的量化算子替换接口，基于第一阶段的约束模型进行量化训练。
     - 该方法可实现无损量化训练。

---

#### **参考示例**

- `test/test_alexnet.py`：展示完整的浮点训练、浮点测试、约束训练、约束浮点测试和量化训练、量化效果测试流程。

---

#### **接口简介**

1. **浮点算子范围约束接口**
   ```python
   # 加载约束和量化相关配置文件
   config_file = './config.yaml'

   # 进行浮点约束训练
   model = Model()
   model = linger.constrain(model, config_file=config_file)
   ```

   通过修改配置文件来调整约束范围。

2. **量化算子替换与参数应用接口**
   ```python
   # 配置量化参数
   config_file = './config.yaml'

   # 初始化量化训练
   model = Model()
   model = linger.init(model, config_file=config_file)
   ```

3. **量化ONNX计算图导图接口**
   ```python
   with torch.no_grad():
       linger.onnx.export(model, dummy_input, "model.onnx", opset_version=12, input_names=["input"], output_names=["output"])
   ```

---

#### **量化配置文件`config.yaml`说明**

| **参数**               | **描述**                                                                 |
|------------------------|--------------------------------------------------------------------------|
| `clamp_info`           | 约束信息配置                                                             |
| `clamp_activation_value` | 激活值约束范围，默认为8（约束到[-8, 8]）                              |
| `clamp_bias_value`      | 偏置约束值，默认为`null`                                               |
| `clamp_factor_value`    | 权重动态约束参数，默认为7（约束到`weight.abs().mean() * 7`）          |
| `device`               | 设备设置，支持`cuda`                                                   |
| `dtype`                | 默认浮点数据类型，默认为`torch.float32`                               |
| `open_quant`           | 是否启用量化，默认为`true`                                             |
| `platform`             | 目标平台，默认为`venusA`，支持`arcs`、`mars`、`venusA`                 |
| `quant_info`           | 量化信息配置                                                            |
| `a_calibrate_name`     | 激活校准方法，默认为`top_10`                                           |
| `a_strategy`           | 激活量化策略，默认为`RANGE_MEAN`                                      |
| `activate_bits`        | 激活数据位宽，默认为8，支持8bit、16bit、32bit                         |
| `activation_type`      | 激活类型，默认为`none`                                                |
| `bias_bits`            | 偏置数据位宽，默认为32bit                                              |
| `is_perchannel`        | 是否启用通道级量化，默认为`false`                                      |
| `is_symmetry`          | 是否启用对称量化，默认为`true`                                         |
| `qat_method`           | QAT量化方案，默认为`MOM`，支持`MOM`和`TQT`                             |
| `round_mode`           | 舍入方式，默认为`floor_add`（floor+0.5）                               |
| `w_calibrate_name`     | 权重校准方法，默认为`abs_max`                                          |
| `w_strategy`           | 权重量化策略，默认为`RANGE_MEAN`                                      |
| `weight_bits`          | 权重位宽，默认为8bit，支持4bit、8bit                                   |
| `quant_method`         | 量化方式，默认为`NATIVE`，支持`NATIVE`、`CUDA`、`ONNX`                 |
| `seed`                 | 随机种子，默认为42                                                     |

---
