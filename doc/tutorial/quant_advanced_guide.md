## 校准（PTQ）使用方法
* 因为校准时会默认按照weight的clip配置进行weight的初始化，故暂不支持循环多组数据校准（仅支持一轮输入校准）
* 校准时会创建add、bmm等小算子的module
```python    
    # 修改cfg.yaml，通过a_calibrate_name和w_calibrate_name设置校准方法，推荐使用默认配置即可；
    # 量化配置
    model = linger.init(model, config_file = 'cfg.yaml')
    # 加载预训练模型
    model.load_state_dict("./pre_train.pt")
    with linger.calibration():  # 校准开关
        model(torch.load("/yrfs4/inference/sqtu2/LLM/code/linger3.0/my_linger/calibrate_input.pt")) # 走一遍前向，开始校准

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
    