
使用linger进行量化训练极其方便，只需要`import linger`，再把 linger 相关设置加到合适的代码处，就可以实现一键量化训练。
关于量化训练，我们推荐的主要将其分为两阶段：
- 第一阶段，浮点约束训练。使用linger的constrain相关接口对网络进行约束，可以基于浮点训练好的模型也可以从头开始训练，将网络训练至收敛
- 第二阶段，定点量化微调。使用linger的init相关接口，基于第一步中得到的浮点约束模型进行量化训练微调，得到相比于浮点模型的无损量化训练结果

## 第一阶段，浮点约束训练
```python
# 得到初始 model
model = Model()

# 使用linger进行浮点约束设置
# 配置约束相关参数
linger.config_save_to_yaml('./config.yaml') # 获取默认配置
# 修改 config.yaml 改变模型配置
config_file = './config.yaml'
model = linger.constrain(model, config_file = config_file) #默认进行静态约束，weight和activation约束到[-8, 8]
# 进行浮点网络约束训练
# 训练结束，保存浮点网络

```

## 第二阶段，定点量化微调

```python
# 得到初始 model
model = Model()
# 添加linger量化训练设置
# 修改 config.yaml 改变模型配置
config_file = './config.yaml'
model = linger.init(model, config_file = config_file) #快速进行量化训练，quant_config默认配置即可，高阶配置参考'量化进阶指导'

# 加载第一阶段训练好的浮点约束网络参数到 model 里
# 进行量化训练

# 达到无损后，使用 torch.onnx.export 将 model 导出 onnx，将该 onnx 交给后端引擎 thinker 进行处理
with torch.no_grad():
    linger.onnx.export(model, dummy_input, "model.onnx", opset_version=12, input_names=["input"], output_names=["output"])
```


## config.yaml 介绍
* 基础配置
    calibration: false  # 校准开关
    clamp_info: # 约束信息配置
        clamp_activation_value: 8   # 激活约束浮点值，8代表约束到[-8, 8]
        clamp_bias_value: null      # bias约束浮点值，默认值为None
        clamp_factor_value: 7       # weight动态约束参数，默认值为7，代表约束到weight.abs().mean() * 7
        clamp_weight_value: null    # weight静态约束值，默认值为None
    device: cuda
    dtype: torch.float32    # 默认浮点数据类型
    open_quant: true
    platform: venusA        # 平台设置，目前支持arcs, mars, venusA
    quant_info: #量化信息配置
        a_calibrate_name: top_10    # 激活校准方法，默认top_10，一般不需要修改
        a_strategy: RANGE_MEAN      # 激活量化原理
        activate_bits: 8            # 激活数据位宽，默认8bit，支持8bit，16bit，32bit
        activation_type: none       # 激活类型，默认None
        bias_bits: 32               # bias数据位宽，默认32bit
        is_perchannel: false        # perchannel量化开关，暂不支持此功能
        is_symmetry: true           # 对称/非对称量化开关，目前只支持对称量化
        qat_method: MOM             # QAT量化方案，支持MOM和TQT
        round_mode: floor_add       # 舍入方式，默认支持floor+0.5
        w_calibrate_name: abs_max   # 权重校准原理，默认abs_max，一般不需要修改
        w_strategy: RANGE_MEAN      # 权重量化原理
        weight_bits: 8              # 权重位宽，默认8bit，支持4bit，8bit
    quant_method: NATIVE            # 量化/伪量化方式，默认NATIVE，支持NATIVE、CUDA、ONNX
    seed: 42

