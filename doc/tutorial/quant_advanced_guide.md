## 混合精度量化与局部约束

混合精度和局部clamp推荐通过配置文件中的`part1`、`part2`等配置段实现，便于训练、恢复和导图复现。`linger.const_module`和`linger.quant_module`也支持对子模块树进行快速覆盖，适合定位与消融实验。完整匹配规则、YAML/API示例、动态tensor op限制、验证方法和效果调优流程见[混合精度量化、局部约束与效果调优指南](mixed_precision_quantization.md)。

## TQT量化训练

TQT不只是修改`qat_method`。当前实现还需要正确选择伪量化后端、只执行一次代表性校准、在`linger.init`后创建optimizer，并在验证或导图前刷新scale。完整流程、代码和排障方式见[TQT量化训练完整指南](tqt_training.md)。

## 校准（TQT）使用方法
* 当前内置`linger.calibration()`依赖TQT的`learning_data/is_calibrate`状态，不要对MOM使用该上下文。
* TQT校准只支持一个代表性batch的一次forward，不支持循环多组数据；同一量化器重复校准会报错。
* 校准时可能创建add、bmm等小算子的module。共享module在一次forward中被重复调用时也可能触发重复校准，需要先验证模型路径。
```python
    # 修改cfg.yaml，通过a_calibrate_name和w_calibrate_name设置校准方法，推荐使用默认配置即可；
    # 量化配置
    model = linger.init(model, config_file = 'cfg.yaml')
    # 加载预训练模型
    model.load_state_dict(torch.load("./pre_train.pt"))
    model.eval()
    with torch.no_grad(), linger.calibration():  # TQT校准开关
        model(torch.load("/yrfs4/inference/sqtu2/LLM/code/linger3.0/my_linger/calibrate_input.pt")) # 走一遍前向，开始校准

    # optimizer必须在linger.init之后创建，确保learning_data进入参数组
    optimizer = build_optimizer(model)
    # 开始量化训练
```

## linger.init/constrain中'disable_module'使用方法
* 关闭model中某一类算子的量化或者约束训练；
* 'disable_module = (nn.LSTM, nn.Linear)'
* 量化示例：model = linger.init(model, config_file = 'cfg.yaml', disable_module = disable_module)
* 约束示例：model = linger.constrain(model, config_file = 'cfg.yaml', disable_module = disable_module)

## linger.init/constrain中'disable_submodel'使用方法
* 关闭model中某一层或者某一个模块的量化或者约束训练；
* disable_submodel = ('module_name*', )这些module的量化
* 量化示例：model = linger.init(model, config_file = 'cfg.yaml', disable_submodel=disable_submodel)使用, classifier为model下一级module名称，*表示匹配当前级及其子集module
* 约束示例：model = linger.constrain(model, config_file = 'cfg.yaml', disable_submodel=disable_submodel)

## linger.init中可通过yaml文件加载配置，当前配置可通过linger.config_save_to_yaml保存
## config.yaml 介绍
* 基础配置
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
