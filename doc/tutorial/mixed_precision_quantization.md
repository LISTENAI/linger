# 混合精度量化、局部约束与效果调优指南

本文说明当前 Linger 3.x 如何按子模块配置不同的量化位宽和约束范围，以及如何用混合精度实验定位并恢复量化精度。推荐把最终方案写入 YAML 的 `part1`、`part2` 等配置段；`linger.const_module` 和 `linger.quant_module` 更适合交互式定位、消融实验或无法静态写入 YAML 的动态量化模块。

## 目录

1. [选择配置方式](#1-选择配置方式)
2. [YAML part 的匹配和覆盖规则](#2-yaml-part-的匹配和覆盖规则)
3. [准备模块名称](#3-准备模块名称)
4. [完整 mix_quant.yaml 示例](#4-完整-mix_quantyaml-示例)
5. [约束训练和量化训练接入](#5-约束训练和量化训练接入)
6. [使用 const_module 局部约束](#6-使用-const_module-局部约束)
7. [使用 quant_module 局部量化](#7-使用-quant_module-局部量化)
8. [检查实际生效配置](#8-检查实际生效配置)
9. [混合精度效果调优流程](#9-混合精度效果调优流程)
10. [恢复、导图和部署](#10-恢复导图和部署)
11. [常见问题](#11-常见问题)

## 1. 选择配置方式

| 方式 | 适用场景 | 优点 | 主要限制 |
| --- | --- | --- | --- |
| YAML `partN` | 最终训练、恢复、导图和可复现实验 | 配置可审计、可版本化，`constrain/init` 自动应用 | 需要事先知道 module name；动态创建的 tensor op 当前使用全局配置 |
| `linger.const_module` | 约束模型的快速局部 sweep | 直接按 Python 子模块对象递归覆盖 | 覆盖值不进入 YAML/checkpoint，恢复和导图必须重放 |
| `linger.quant_module` | 量化模型的快速位宽/clamp sweep，或已实例化的动态 tensor op | 可直接覆盖 input/weight/bias/output 量化器 | 覆盖值不进入 YAML/checkpoint；必须在量化模块创建后调用 |
| `disable_submodel` | 定位某个子图是否是掉点来源 | 快速构造 float island 对照 | 不是混合精度，最终部署是否支持需单独证明 |

推荐原则：

1. 先用 API 或 `disable_submodel` 做短周期定位。
2. 确认有效的位宽和 clamp 组合后，迁移到完整 YAML `partN`。
3. 用该 YAML 从模型构建开始重新跑约束、QAT、严格复载和导图。
4. 保存 YAML、代码版本、模块名称清单和指标，不只保存 checkpoint。

当 YAML 与 API 同时使用时，调用顺序决定结果：`linger.constrain/init` 先根据 YAML 创建模块，之后调用的 `const_module/quant_module` 会覆盖已创建对象的属性。最终方案应避免依赖这种隐式叠加；必须叠加时，记录并在恢复、验证和导图脚本中逐条重放。

## 2. YAML part 的匹配和覆盖规则

当前 loader 把顶层中“不是内置字段、值为 dict、且包含 `name`”的配置识别为 mixed precision part。通常命名为 `part1`、`part2`，但真正决定是否识别的是 `name` 字段。

### 2.1 名称匹配

- `name` 必须是字符串列表。
- 每个字符串按 module 完整名称或子树前缀匹配。
- `backbone.stage2` 会匹配它本身以及 `backbone.stage2.conv1` 等后代。
- part 匹配不支持 `*`、`?` 等 fnmatch 通配符；通配符只适用于 `disable_submodel`。
- 同一个 module 命中多个 part 时，YAML 中先出现的 part 生效。

因此应避免下列重叠，除非明确需要 first-match：

```yaml
part1:
  name: [backbone]

part2:
  name: [backbone.stage2]
```

上例中 `backbone.stage2` 已被 `part1` 匹配，`part2` 不会生效。更可靠的做法是让各 part 的 module 子树互不重叠。

### 2.2 quant_info 覆盖

- `weight_bits`、`activate_bits`、`bias_bits` 在 module 被量化替换时应用。
- `activate_bits` 是该 part 的通用 activation 位宽，默认影响 module 输入和输出。
- `input_bits`、`output_bits` 是 part 专用扩展字段；替换结束后再次遍历 module，分别覆盖 input/output quantizer。
- `qat_method` 当前由全局 `QUANT_CONFIGS.quant_info.qat_method` 读取，不能用 part 在不同层混用 MOM 和 TQT。
- `quant_method`、`device`、`platform` 也是全局字段，不能按 part 切换。

### 2.3 clamp_info 覆盖

`constrain` 和 `init` 都会读取 part 的 `clamp_info`：

- `clamp_activation_value`：该子树 activation 的对称约束范围。
- `clamp_weight_value`：weight 静态约束绝对值。
- `clamp_factor_value`：weight 动态范围 `mean(abs(weight)) * factor`。
- `clamp_bias_value`：bias 静态约束绝对值。

weight 静态值和动态 factor 不要同时启用。当前约束 module 中 `clamp_factor_value` 非空时优先使用动态 factor；为避免歧义，使用动态范围时把 `clamp_weight_value` 写为 `null`，使用静态范围时把 `clamp_factor_value` 写为 `null`。

### 2.4 part 必须写完整字段

当前 loader 会为每个 part 新建一份内置默认 `QuantInfo/ClampInfo`，再应用 part 中显式字段。合并到全局配置时，这些内置默认值也会覆盖全局自定义值。因此 part 中省略字段不等同于继承全局值。

例如全局 `activate_bits: 16`，part 只写 `weight_bits: 4` 时，该 part 的 `activate_bits` 可能回到内置默认 8。为保证可复现，每个 part 都应写完整的 `quant_info` 和 `clamp_info`。

## 3. 准备模块名称

module name 应来自执行完训练所需 trace/fusion、但尚未调用 `linger.constrain/init` 的同一模型结构：

```python
model = build_model().to(device)

if use_trace_layers:
    linger.QUANT_CONFIGS.device = device
    linger.trace_layers(model, *example_inputs, fuse_bn=True)

for name, module in model.named_modules():
    print(f"{name or '<root>':48s} {type(module).__name__}")
```

把输出的完整 name 写进 YAML。训练、恢复和导图必须使用相同 model code 与 trace/fusion 顺序，否则同一个 part 可能匹配到不同模块或完全失效。

建议同时检查：

- name 是否拼写、大小写完全一致；
- 目标是单层还是整棵子树；
- 两个 part 是否发生父子重叠；
- module 是否在 Linger 的约束/量化注册表中；
- 目标 module 是否因 fusion 被替换或改名。

## 4. 完整 mix_quant.yaml 示例

仓库根目录的 [`mix_quant.yaml`](../../mix_quant.yaml) 是可直接修改的完整示例。下面演示全局 8/8-bit、首层 16-bit 输出，以及一个敏感 stage 的 16-bit activation：

```yaml
calibration: false
open_quant: true
quant_method: NATIVE
device: cuda
dtype: torch.float32
seed: 42
platform: venusA

clamp_info:
  clamp_activation_value: 8
  clamp_weight_value: null
  clamp_factor_value: 7
  clamp_bias_value: null

quant_info:
  weight_bits: 8
  activate_bits: 8
  bias_bits: 32
  a_strategy: RANGE_MEAN
  w_strategy: RANGE_MEAN
  is_symmetry: true
  is_perchannel: false
  round_mode: floor_add
  activation_type: none
  qat_method: MOM
  w_calibrate_name: abs_max
  a_calibrate_name: top_10

part1:
  name:
    - backbone.conv1
  quant_info:
    weight_bits: 8
    activate_bits: 8
    input_bits: 8
    output_bits: 16
    bias_bits: 32
    a_strategy: RANGE_MEAN
    w_strategy: RANGE_MEAN
    is_symmetry: true
    is_perchannel: false
    round_mode: floor_add
    activation_type: none
    qat_method: MOM
    w_calibrate_name: abs_max
    a_calibrate_name: top_10
  clamp_info:
    clamp_activation_value: 8
    clamp_weight_value: null
    clamp_factor_value: 7
    clamp_bias_value: null

part2:
  name:
    - backbone.blocks2
    - backbone.blocks3_0
  quant_info:
    weight_bits: 8
    activate_bits: 16
    input_bits: 16
    output_bits: 16
    bias_bits: 32
    a_strategy: RANGE_MEAN
    w_strategy: RANGE_MEAN
    is_symmetry: true
    is_perchannel: false
    round_mode: floor_add
    activation_type: none
    qat_method: MOM
    w_calibrate_name: abs_max
    a_calibrate_name: top_10
  clamp_info:
    clamp_activation_value: 16
    clamp_weight_value: null
    clamp_factor_value: 7
    clamp_bias_value: null
```

示例 module name 需要替换为业务模型实际名称。高于 8-bit 的组合是否可导出和部署，必须以目标平台 op/precision 支持和 Thinker 验证为准。

也可以使用 skill 中的生成器展开完整 part 字段。`parts.json` 示例：

```json
[
  {
    "name": ["backbone.conv1"],
    "quant_info": {"input_bits": 8, "output_bits": 16},
    "clamp_info": {"clamp_activation_value": 8}
  },
  {
    "name": ["head.classifier"],
    "quant_info": {"input_bits": 16, "output_bits": 16},
    "clamp_info": {"clamp_activation_value": 16}
  }
]
```

```bash
python3 lnn-quant-training/scripts/generate_qat_config.py \
  --output configs/mix_quant.yaml \
  --platform venusA \
  --device cuda \
  --quant-method NATIVE \
  --qat-method MOM \
  --weight-bits 8 \
  --activation-bits 8 \
  --parts-json parts.json
```

生成器会把全局 `quant_info/clamp_info` 展开进每个 part，避免字段遗漏导致回退到内置默认值。

## 5. 约束训练和量化训练接入

推荐约束和 QAT 使用同一份最终 YAML，使局部 clamp 在两个阶段保持一致：

```python
config_path = "configs/mix_quant.yaml"

model = build_model().to(device)
model.load_state_dict(float_state, strict=True)

if use_trace_layers:
    linger.QUANT_CONFIGS.device = device
    linger.trace_layers(model, *example_inputs, fuse_bn=True)

# 第一阶段：按全局和 part clamp 执行约束训练。
model = linger.constrain(model, config_file=config_path)
constrain_optimizer = build_optimizer(model.parameters(), lr=constrain_lr)
train_constrained(model, constrain_optimizer)

# 第二阶段：按全局和 part 位宽/clamp 创建量化模块。
model = linger.init(model, config_file=config_path)
qat_optimizer = build_optimizer(model.parameters(), lr=qat_lr)
train_qat(model, qat_optimizer)
```

每次 `trace_layers`、`constrain` 或 `init` 改变结构后，都要重建 optimizer/scheduler。若使用 DDP/FSDP，应先完成结构替换和局部配置，再包装模型。

对 TQT，仍须遵守[TQT量化训练完整指南](tqt_training.md)中的后端、单次校准、阈值 optimizer、梯度审计和 scale refresh 规则。part 只能混合位宽和 clamp，不能混合 MOM/TQT。

## 6. 使用 const_module 局部约束

`linger.const_module` 接收一个已经完成约束替换的 module，并递归覆盖它及其后代中所有约束 module：

```python
model = linger.constrain(model, config_file="configs/base_quant.yaml")

# 对 stage2 使用较宽 activation 范围和更紧的动态 weight 范围。
linger.const_module(
    model.backbone.stage2,
    c_activation_val=16,
    c_weight_val=None,
    c_bias_val=None,
    c_weight_factor=5,
)

# 所有局部覆盖完成后再创建 optimizer。
optimizer = build_optimizer(model.parameters(), lr=constrain_lr)
```

参数语义：

| 参数 | 作用 |
| --- | --- |
| `c_activation_val` | 设置 `clamp_activation` |
| `c_weight_val` | 设置静态 `clamp_weight` |
| `c_bias_val` | 设置静态 `clamp_bias` |
| `c_weight_factor` | 设置动态 `clamp_factor` |

调用时应显式传入所有四项，避免函数默认值无意覆盖全局配置。静态 weight clamp 与动态 factor 二选一。

`const_module` 必须在 `linger.constrain` 之后调用；对原始 `nn.Module` 调用不会创建约束 module，也不会替代 `linger.constrain`。

## 7. 使用 quant_module 局部量化

`linger.quant_module` 接收一个已经完成量化替换的 module，并递归覆盖已存在量化器的位宽和 clamp：

```python
model = linger.init(model, config_file="configs/base_quant.yaml")

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

# 局部配置完成后再做 TQT 校准/MOM 首次统计和创建 optimizer。
optimizer = build_optimizer(model.parameters(), lr=qat_lr)
```

参数语义：

| 参数 | 作用 |
| --- | --- |
| `data_bits` | 子树 input quantizer 位宽，对多输入 tensor op 会覆盖每个输入量化器 |
| `weight_bits` | weight quantizer 位宽 |
| `bias_bits` | bias quantizer 位宽 |
| `out_bits` | output quantizer 位宽 |
| `c_activation_val` | output quantizer activation clamp |
| `c_weight_val` | weight quantizer 静态 clamp |
| `c_weight_factor` | weight quantizer 动态 clamp factor |
| `c_bias_val` | bias quantizer 静态 clamp |

调用时同样显式写全参数。`data_bits` 和 `out_bits` 分别对应 YAML 的 `input_bits`、`output_bits`，不是同一个边界概念。

### 动态 tensor op

Add、Mul、MatMul、BMM、Cat、GELU、Softmax 等 QTensor op 可能在第一次真实 forward 时动态创建。当前动态创建路径使用全局 `quant_info`，不会自动读取父 module 命中的 part。

需要局部覆盖这类 op 时：

1. 用代表性输入执行一次 forward，使动态量化 module 实例化。
2. 从 `model.named_modules()` 找到生成的 `_linger_qtensor...` module。
3. 对该 module 或包含它的父 module 调用 `quant_module`。
4. 重新完成必要的 MOM 统计或 TQT scale 刷新，并检查最终状态。

由于动态 module name 和创建路径依赖真实 forward，这类 API 覆盖必须在恢复和导图时使用相同输入路径重放。若目标组合可以通过全局位宽解决，优先使用全局 YAML，减少动态覆盖。

## 8. 检查实际生效配置

不要只检查 YAML 文本。量化初始化后打印每个 module 的真实量化器状态：

```python
import torch


def print_quant_config(model):
    for name, module in model.named_modules():
        fields = []

        input_quantizer = getattr(module, "input_quantizer", None)
        if isinstance(input_quantizer, torch.nn.ModuleList):
            fields.append(
                "input_bits=" + str([q.data_bits for q in input_quantizer])
            )
        elif input_quantizer is not None:
            fields.append(f"input_bits={input_quantizer.data_bits}")

        weight_quantizer = getattr(module, "weight_quantizer", None)
        if weight_quantizer is not None:
            fields.append(f"weight_bits={weight_quantizer.data_bits}")
            fields.append(f"weight_clamp={weight_quantizer.clamp_value}")
            fields.append(f"weight_factor={weight_quantizer.clamp_factor}")

        bias_quantizer = getattr(module, "bias_quantizer", None)
        if bias_quantizer is not None:
            fields.append(f"bias_bits={bias_quantizer.data_bits}")

        output_quantizer = getattr(module, "output_quantizer", None)
        if output_quantizer is not None:
            fields.append(f"output_bits={output_quantizer.data_bits}")
            fields.append(f"activation_clamp={output_quantizer.clamp_value}")

        if fields:
            print(name, type(module).__name__, ", ".join(fields))
```

约束模型可检查：

```python
for name, module in model.named_modules():
    if hasattr(module, "clamp_activation"):
        print(
            name,
            module.clamp_activation,
            module.clamp_weight,
            module.clamp_factor,
            module.clamp_bias,
        )
```

检查结果应作为实验 artifact 保存，并重点核对：

- part 中每个目标 module 都已命中；
- 未命中 module 保持全局配置；
- 重叠 part 的实际 first-match 符合预期；
- 相邻 module 的 output/input bits 没有意外冲突；
- 动态 tensor op 使用了预期位宽；
- 导出 ONNX 中的 bit/scale/platform 属性与 PyTorch module 一致。

## 9. 混合精度效果调优流程

### 9.1 先定位掉点阶段

```text
float -> constrained 掉点：优先调 clamp、fusion、BN 和结构
constrained -> QAT 掉点：优先调位宽、量化边界、统计量和 QAT 超参
QAT -> ONNX/runtime 掉点：停止盲目续训，检查 scale、图属性和 runtime
```

所有实验必须使用同一验证集、预处理、metric 和输入契约。

### 9.2 Stage 级定位

1. 建立全局 8/8-bit 基线并保存指标。
2. 每次只把一个 stage 的 input/output activation 提升到 16-bit，其他配置不变。
3. 若指标明显恢复，在该 stage 内继续二分到 block 或 layer。
4. 将不敏感层恢复到 8-bit，只保留最小高精度集合。
5. 用目标平台工具验证最终位宽组合，记录时延、模型大小和内存变化。

优先候选通常是第一层、最后一层、检测/分割 head、残差汇合、LayerNorm、Softmax、attention matmul、RNN hidden/cell 和大 projection，但必须用实测证明。

### 9.3 Clamp sweep

- activation 可围绕当前值做小范围单变量 sweep，例如 `4/8/16`。
- weight dynamic factor 可围绕默认 `7` 做小范围 sweep，例如 `5/7/9`。
- 高饱和率通常需要放宽范围；量化噪声大但几乎不饱和，可能是范围过宽、有效格点利用不足。
- 先按 stage 定位，再对少量高误差 layer 局部 override，不先修改全模型。
- 每个实验记录 constrained/QAT 指标、逐层误差、饱和率、SQNR/余弦相似度和部署成本。

### 9.4 位宽和边界

- 优先提高 activation/output bits，再判断 weight 是否同样敏感。
- 第一层输入位宽应符合真实传感器/预处理精度，不能只追求训练指标。
- 残差或 concat 前各分支范围相差较大时，先平衡结构和范围，再统一提高位宽。
- `output_bits` 会影响传给下游的 `QTensor`；下游 input quantizer 可能直接沿用上游 scale/bit，因此需要联合检查边界两侧。
- 4-bit weight 可以降低模型大小，但应先确认该 op、该平台和配套 activation/bias 精度可部署。
- 16/32-bit 在 PyTorch 中可用于定位敏感层，不代表目标硬件一定支持。

### 9.5 训练策略

- 每次位宽或 clamp 方案变化都重新初始化相应量化模型，不复用不匹配的 optimizer 状态。
- 从最佳 constrained checkpoint 开始 QAT，先降低学习率并增加 warm-up/微调周期。
- TQT 检查阈值 optimizer 和首个 backward 梯度；MOM 检查统计量初始化和 scale。
- 基础流程正确但仍略低时，再考虑 teacher distillation 或少量 feature alignment。
- 一次只改一个主要变量，至少重复关键候选以排除随机波动。

### 9.6 推荐实验表

| 实验 | 局部配置 | 目的 | 必须记录 |
| --- | --- | --- | --- |
| Baseline | 全局 8/8-bit | 建立量化损失 | float/constrained/QAT 指标 |
| Stage probe | 单 stage A16 | 找 activation 敏感区 | 指标恢复、内存变化 |
| Weight probe | 敏感 stage W16 或 W4 | 判断 weight 敏感度/压缩空间 | 精度、模型大小、平台支持 |
| Clamp probe | 单 stage `A=4/8/16` 或 factor `5/7/9` | 平衡饱和和分辨率 | 饱和率、逐层误差、指标 |
| Layer narrowing | 单 layer 高精度 | 最小化高精度范围 | 相邻边界、部署成本 |
| Disable control | 临时禁用子树量化 | 证明问题位置 | float island 前后差值 |

## 10. 恢复、导图和部署

最终 YAML 方案的恢复顺序：

```text
build model
-> 复现 trace/fusion
-> linger.constrain/init(..., 同一 mix YAML)
-> strict load matching checkpoint
-> 重建 optimizer（继续训练时）
-> warm-up/统计或 TQT scale refresh
-> eval + linger.onnx.export
```

使用 `const_module/quant_module` 覆盖时，在 `constrain/init` 后、加载和评测前按训练时同一顺序重放覆盖。checkpoint 保存 tensor 和 buffer，不会保存 Python 属性配置本身；仅加载 checkpoint 不能证明局部位宽/clamp 已恢复。

导图后运行：

```bash
python3 lnn-quant-training/scripts/inspect_quant_onnx.py \
  artifacts/model_quant.onnx \
  --format markdown \
  --fail-on-issues
```

逐层核对 ONNX 的 input/output/weight/bias bits、scale 和 platform。最后使用 Thinker/目标 runtime 验证 op 支持、memory、latency 和 golden input consistency。只有 PyTorch 指标恢复而目标平台不支持该高精度组合时，该方案不能作为最终交付。

## 11. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| part 完全未生效 | module name 不匹配或 fusion 后改名 | 从同一结构的 `named_modules()` 重新取 name |
| 更具体的 part 未生效 | 被前面的父级 part 先匹配 | 移除重叠或调整为互斥子树 |
| part 未写字段意外变为 8-bit | part 使用内置默认，而非继承全局自定义值 | 每个 part 写完整 quant/clamp 配置 |
| 写了 `*` 但没有匹配 | part 不支持 fnmatch | 使用完整 name 或父级子树前缀 |
| API 调用后恢复/导图不一致 | override 未进入 YAML/checkpoint | 重放 API，最终迁移到 YAML |
| 动态 Add/MatMul 仍是全局位宽 | tensor op 首次 forward 才创建，且创建时使用全局配置 | materialize 后对实际 module 调用 `quant_module` |
| 相邻层位宽与预期不符 | 上游 QTensor 的 bit/scale 被下游沿用 | 联合检查 upstream output 和 downstream input |
| 高位宽恢复精度但无法打包 | 目标 op/平台不支持该位宽 | 缩小高精度范围或修改量化友好结构 |
| constrain 已明显掉点 | clamp 过紧、BN/fusion 或结构问题 | 先做局部 clamp sweep，不进入 QAT 掩盖问题 |
| QAT 达标但 runtime 掉点 | 导出 scale/属性、输入契约或工具链不一致 | 进入 ONNX/Thinker first-mismatch 排查 |
