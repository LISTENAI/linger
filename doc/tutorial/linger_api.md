## linger normalize 约束训练接口

### `linger.constrain`

根据全局配置和 YAML `partN` 的局部 `clamp_info` 替换约束模块。调用会改变模型结构，之后需要重建 optimizer/scheduler。

```python
model = linger.constrain(
    model,
    config_file="configs/mix_quant.yaml",
    disable_module=None,
    disable_submodel=None,
)
```

### `linger.const_module`

在 `linger.constrain` 之后，对指定子模块及其后代递归覆盖约束范围：

```python
linger.const_module(
    model.backbone.stage2,
    c_activation_val=16,
    c_weight_val=None,
    c_bias_val=None,
    c_weight_factor=5,
)
```

`c_weight_val` 是静态 weight clamp，`c_weight_factor` 是 `mean(abs(weight)) * factor` 动态 clamp，两者应只启用一个。API 覆盖不写入 checkpoint，最终方案推荐迁移到 YAML `partN`。

## linger 量化训练接口

### `linger.init`

根据全局配置和 YAML `partN` 的局部 `quant_info/clamp_info` 替换量化模块。调用会改变模型结构，之后需要重建 optimizer/scheduler。

```python
model = linger.init(
    model,
    config_file="configs/mix_quant.yaml",
    disable_module=None,
    disable_submodel=None,
)
```

### `linger.quant_module`

在 `linger.init` 之后，对指定子模块及其后代中已存在的量化器递归覆盖位宽和 clamp：

```python
linger.quant_module(
    model.backbone.stage2,
    c_activation_val=16,
    c_weight_val=None,
    c_bias_val=None,
    c_weight_factor=5,
    data_bits=16,
    weight_bits=8,
    bias_bits=32,
    out_bits=16,
)
```

`data_bits/out_bits` 分别设置 input/output quantizer。对多输入 tensor op，`data_bits` 会应用到每个输入量化器。动态 QTensor op 需要先通过真实 forward 创建，再执行 API 覆盖。最终方案推荐使用 YAML `partN`；API 覆盖必须在恢复和导图时按相同顺序重放。

## 其他工具

- `linger.config_save_to_yaml(path)`：保存当前全局配置。mixed precision part 建议使用完整模板或 skill 生成器创建。
- `linger.disable_quant_ops(...)`：关闭指定量化 module/tensor op 注册，影响当前进程的后续模型。
- `linger.get_quant_ops_name()`：查看当前进程注册的量化 module 和 tensor op。

混合精度的完整 YAML、匹配优先级、验证和效果调优方法见[混合精度量化、局部约束与效果调优指南](mixed_precision_quantization.md)。
