
使用linger进行量化训练极其方便，只需要`import linger`，再把 linger 相关设置加到合适的代码处，就可以实现一键量化训练。
关于量化训练，我们推荐的主要将其分为两阶段：
- 第一阶段，浮点约束训练。使用linger的normalize相关接口对网络进行约束，将随机初始化的权重进行从0起步的浮点约束训练，将网络训练至收敛
- 第二阶段，定点量化微调。使用linger的init相关接口，基于第一步中得到的浮点模型进行量化训练微调，得到相比于浮点模型的无损量化训练结果

## 第一阶段，浮点约束训练
```python
# 得到初始 model
model = Model()

# 使用linger进行浮点约束设置
linger.trace_layers(model, model, dummy_input, fuse_bn=True)
linger.disable_normalize(model.last_layer)
type_modules  = (nn.Conv2d)
normalize_modules = (nn.Conv2d,nn.Linear)
linger.normalize_module(model.mid_conv, type_modules = type_modules, normalize_weight_value=16, normalize_bias_value=16, normalize_output_value=16)
model = linger.normalize_layers(model, normalize_modules = normalize_modules, normalize_weight_value=8, normalize_bias_value=8, normalize_output_value=8)

# 进行浮点网络约束训练
# 训练结束，保存浮点网络

```

## 第二阶段，定点量化微调

```python
# 得到初始 model
model = Model()

# 继承第一阶段的设置，不要进行任何改动
linger.trace_layers(model, model, dummy_input, fuse_bn=True)
linger.disable_normalize(model.last_fc)
type_modules = (nn.Conv2d)
normalize_modules = (nn.Conv2d, nn.Linear)
linger.normalize_module(model.mid_conv, type_modules = type_modules, normalize_weight_value=16, normalize_bias_value=16, normalize_output_value=16)
model = linger.normalize_layers(model, normalize_modules = normalize_modules, normalize_weight_value=8, normalize_bias_value=8, normalize_output_value=8)

# 添加linger量化训练设置
linger.disable_quant(model.last_fc)
quant_modules = (nn.Conv2d, nn.Linear)
model = linger.init(model, quant_modules = quant_modules)

# 加载第一阶段训练好的浮点约束网络参数到 model 里
# 进行量化训练

# 达到无损后，使用 torch.onnx.export 将 model 导出 onnx，将该 onnx 交给后端引擎 thinker 进行处理
with torch.no_grad():
    torch.onnx.export(model, dummy_input, "model.onnx", opset_version=12, input_names=["input"], output_names=["output"])
```