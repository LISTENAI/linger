# Linger TQT 量化训练完整指南

本文说明当前 Linger 3.x 实现中的 TQT（Trained Quantization Thresholds）训练、校准、监控、恢复和 ONNX 导出流程。示例以 8-bit 权重和激活为起点；模型结构、输入预处理、目标平台和验收指标仍应以实际业务为准。

> 本文描述的是当前仓库实现。`qat_method: TQT` 才是 TQT QAT 的选择开关；`QuantStrategy.TQT` 是另一个枚举，不能代替 `qat_method`。

## 目录

1. [TQT 的状态和量化公式](#1-tqt-的状态和量化公式)
2. [TQT 与 MOM 的区别](#2-tqt-与-mom-的区别)
3. [完整流程总览](#3-完整流程总览)
4. [准备配置文件](#4-准备配置文件)
5. [模型结构和初始化顺序](#5-模型结构和初始化顺序)
6. [执行一次校准](#6-执行一次校准)
7. [创建 optimizer](#7-创建-optimizer)
8. [训练、验证和监控](#8-训练验证和监控)
9. [刷新并固定最终 scale](#9-刷新并固定最终-scale)
10. [保存和恢复 checkpoint](#10-保存和恢复-checkpoint)
11. [量化效果优化](#11-量化效果优化)
12. [常见问题排查](#12-常见问题排查)
13. [完整 ONNX 导图流程](#13-完整-onnx-导图流程)
14. [最小端到端模板](#14-最小端到端模板)

## 1. TQT 的状态和量化公式

每个 TQT `Quantizer` 包含以下状态：

| 状态 | 类型 | 作用 |
| --- | --- | --- |
| `learning_data` | `torch.nn.Parameter` | 可学习阈值的 log2 表示，创建时初始值为 `2.1` |
| `is_calibrate` | bool buffer | 标记该量化器是否已执行内置校准 |
| `scale` | buffer | 当前 power-of-two 量化 scale，推理和导图读取该值 |

对位宽 `b`，当前 native 路径的核心计算可近似写为：

```text
threshold_log2 = learning_data
scale_exponent = round(b - 1 - threshold_log2)
scale_exponent = max(scale_exponent, 0)
scale = 2 ** scale_exponent

qmin = -(2 ** (b - 1))
qmax =  (2 ** (b - 1)) - 1
fake_quant(x) = clamp(round(x * scale), qmin, qmax) / scale
```

因此需要注意：

- `learning_data` 学习的是阈值的对数，不是最终 `scale` 本身；校准会把它初始化为近似 `log2(threshold)`。
- scale exponent 需要经过取整，所以 `learning_data` 连续变化时，`scale` 可能在若干 step 内保持不变，跨过取整边界后才跳到相邻的 2 的幂。
- 非 bias 量化器在 `bits >= 16` 时，当前实现把 scale exponent 上限限制为 20，即 `scale <= 2**20`。
- fake quant 的舍入方式受 `round_mode` 和量化器角色影响；上式用于理解 TQT 主体行为，不替代当前源码。

## 2. TQT 与 MOM 的区别

| 项目 | TQT | MOM |
| --- | --- | --- |
| 范围状态 | 可训练 `learning_data` | buffer `running_data` |
| 初始化标记 | `is_calibrate` | `is_init` |
| 范围更新 | optimizer 根据梯度更新 | 训练 forward 以 momentum 更新统计量 |
| 内置 `linger.calibration()` | 可用，且只执行一次代表性 forward | 当前不应使用 |
| optimizer 创建时机 | 必须在 `linger.init` 后 | 仍建议在结构替换后创建 |
| checkpoint 状态 | 包含 `learning_data/is_calibrate/scale` | 包含 `running_data/is_init/scale` |

MOM checkpoint 不能直接作为 TQT 训练的恢复点。MOM 转 TQT 时，应加载可匹配的业务权重，重新初始化 TQT 量化器，重新校准并重建 optimizer。

## 3. 完整流程总览

```text
确定验收指标、输入契约和目标平台
  -> 复测并保存 float baseline
  -> 审计量化友好的模型结构
  -> 可选：trace_layers 和约束训练
  -> 使用同一结构顺序重建模型
  -> linger.init(..., TQT config)
  -> 单个代表性 batch 执行一次 calibration
  -> 创建包含 learning_data 的 optimizer
  -> QAT 首个 backward 检查阈值梯度
  -> 训练并持续验证指标、阈值、scale 和饱和率
  -> 选择 best checkpoint 并严格复载
  -> 用代表性输入刷新最终 scale
  -> eval + no_grad + linger.onnx.export
  -> ONNX checker 和量化属性审计
  -> Thinker 打包及目标 runtime 一致性验证
```

在开始长时间训练前，建议先用 float checkpoint 或随机权重跑通一次最短的 `init -> forward -> export`，提前发现不支持的算子、动态控制流和输入 shape 问题。

## 4. 准备配置文件

### 4.1 NATIVE：CPU 调试和通用推荐配置

`NATIVE` 使用 PyTorch 运算构造伪量化，`learning_data` 参与 autograd。CPU 必须使用该路径；GPU 上也可使用它验证数值和阈值梯度。

```yaml
open_quant: true
quant_method: NATIVE
device: cpu
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
  qat_method: TQT
  w_calibrate_name: abs_max
  a_calibrate_name: top_10
```

在 GPU 上使用 native 路径时，将 `device` 改为 `cuda`，同时把模型和输入放到相同设备。

### 4.2 CUDA_GS：GPU TQT 训练配置

`CUDA_GS` 使用 CUDA extension，并在 backward 中显式计算 `learning_data` 梯度，适合已经编译并验证扩展的 GPU 环境。

```yaml
open_quant: true
quant_method: CUDA_GS
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
  qat_method: TQT
  w_calibrate_name: abs_max
  a_calibrate_name: top_10
```

### 4.3 后端选择注意事项

| `quant_method` | TQT 训练行为 | 建议 |
| --- | --- | --- |
| `NATIVE` | PyTorch 路径，阈值参与 autograd | CPU 必选；GPU 数值调试和训练可用 |
| `CUDA_GS` | CUDA extension 显式返回阈值梯度 | GPU TQT 训练推荐路径之一 |
| `CUDA` | 当前 backward 不返回 `learning_data` 梯度 | 不用于可学习阈值的 TQT 训练 |

`CUDA` 路径仍可能正常更新模型权重，导致 loss 看起来在下降，但 TQT 阈值没有真正学习。它只适合固定阈值或对照评估，除非后续实现已经改变。训练第一个 backward 后必须检查阈值梯度。

配置中的 `qat_method` 当前由全局 `QUANT_CONFIGS.quant_info.qat_method` 读取，不能通过 mixed precision part 在不同层混用 MOM 和 TQT。mixed precision part 可以调整层的位宽，但全模型 QAT 算法应保持一致。完整配置和调优方法见[混合精度量化、局部约束与效果调优指南](mixed_precision_quantization.md)。

## 5. 模型结构和初始化顺序

### 5.1 结构建议

- 优先使用当前项目支持列表中的 Conv、Linear、RNN、pooling、activation 和 tensor op。
- 将 Conv-BN-activation 等可融合模式保持为容易 trace 的静态结构；训练、恢复和导图使用相同 fusion 顺序。
- 避免依赖输入值的 Python 分支、运行时创建 module、不可控的动态 shape 和不稳定的极值算子。
- 残差相加和多分支汇合前尽量保持相近的动态范围。若某一分支长期饱和，先处理结构或范围，不要仅靠扩大所有层的位宽掩盖问题。
- 最后一层、第一层、输出 head、softmax 前后和大范围 embedding/RNN 状态通常更敏感，应通过逐层误差和饱和率确认后再决定是否提高位宽或禁用量化。
- 训练、验证、导图和部署必须使用一致的 normalize、padding、mask、layout、dtype 和输入范围。

### 5.2 正确初始化顺序

```python
import torch
import linger

device = torch.device("cuda")
config_path = "configs/tqt_cuda_gs.yaml"
example_inputs = (torch.randn(1, 3, 224, 224, device=device),)

model = build_model().to(device)
float_state = torch.load("checkpoints/float_best.pt", map_location=device)
model.load_state_dict(float_state, strict=True)

# 仅在训练方案需要 fusion/tracing 时执行；恢复和导图时必须复现。
linger.QUANT_CONFIGS.device = device
adapter = linger.trace_layers(model, *example_inputs, fuse_bn=True)

model = linger.init(model, config_file=config_path)
```

关键规则：

1. 先创建或加载 float/constrained 模型，再执行训练方案规定的 `trace_layers`。
2. 调用 `linger.init` 后，量化模块才会创建 `learning_data` 参数。
3. optimizer、scheduler、DDP/FSDP wrapper 等依赖参数集合的对象必须在 `linger.init` 后创建。
4. 恢复训练和导图时，必须复现相同的 model code、trace/fusion、config、`disable_module` 和 `disable_submodel`。

不要先创建 optimizer 再调用 `linger.init`，否则新增的 `learning_data` 不在 optimizer 中，阈值不会更新。

## 6. 执行一次校准

TQT 可以在训练前用一个具有代表性的 batch 初始化阈值：

```python
model.eval()
with torch.no_grad(), linger.calibration():
    model(*calibration_inputs)
model.train()
```

当前内置校准的约束如下：

- 只执行一次代表性 forward，不要循环多个 calibration batch。
- 同一量化器被第二次校准会抛出异常。共享 module 在单次 forward 中被重复调用，也可能触发该限制，应先用最小输入验证。
- 校准输入应来自真实训练/验证分布，并覆盖典型长度、幅值和 mask；不要使用全零输入或单个极端异常样本。
- activation 默认 `top_10`：元素数足够时取绝对值第 10 大值；小 tensor 使用绝对最大值。
- weight 默认 `abs_max`。当前 `w_like` 实现内部硬编码 CUDA，不建议作为通用默认值。
- RNN 有专门校准路径，当前实现对多层 RNN 有限制；先以 `num_layers == 1` 验证完整流程。

校准后检查状态：

```python
from linger.quant.quantizer import Quantizer

not_calibrated = []
for name, module in model.named_modules():
    if isinstance(module, Quantizer) and hasattr(module, "is_calibrate"):
        if not bool(module.is_calibrate.item()):
            not_calibrated.append(name)

print("uncalibrated quantizers:", not_calibrated)
```

并非所有未校准量化器都一定是错误：某些边界量化器可能未被当前输入路径触发。应结合 module 角色和实际 forward 路径判断。如果校准中途失败，不要继续使用部分量化器已校准、部分未校准的模型；应重建一份新的量化模型后重新校准。

## 7. 创建 optimizer

建议给阈值参数单独建参数组，便于使用独立学习率，并关闭 weight decay：

```python
threshold_params = []
regular_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if name.endswith("learning_data"):
        threshold_params.append(param)
    else:
        regular_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": regular_params, "lr": qat_lr},
        {"params": threshold_params, "lr": threshold_lr, "weight_decay": 0.0},
    ],
    weight_decay=weight_decay,
)

optimizer_param_ids = {
    id(param)
    for group in optimizer.param_groups
    for param in group["params"]
}
missing = [
    name
    for name, param in model.named_parameters()
    if name.endswith("learning_data") and id(param) not in optimizer_param_ids
]
assert not missing, f"TQT parameters missing from optimizer: {missing}"
```

初始 QAT 学习率可从 float 阶段约 `0.1x` 开始，但这只是起点。`threshold_lr` 应单独记录和调优；阈值震荡时先降低它，阈值完全不动时先排查梯度和 optimizer 覆盖，再考虑提高学习率。

## 8. 训练、验证和监控

### 8.1 基本训练循环

```python
for epoch in range(start_epoch, num_epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        inputs, targets = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(*inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if epoch == start_epoch and step == 0:
            audit_tqt_gradients(model)

        optimizer.step()

    # 验证前先根据最新 learning_data 刷新 scale，见第 9 节。
    refresh_tqt_scales(model, validation_example_inputs)
    metric = evaluate(model, val_loader)
    scheduler.step() if scheduler is not None else None

    if metric > best_metric:
        best_metric = metric
        save_checkpoint(...)
```

### 8.2 首个 backward 的梯度审计

```python
import torch
from linger.quant.quantizer import Quantizer

def audit_tqt_gradients(model):
    finite_grad_names = []
    missing_grad_names = []
    invalid_grad_names = []

    for name, module in model.named_modules():
        if not isinstance(module, Quantizer) or not hasattr(module, "learning_data"):
            continue
        if module.is_bias_quantizer:
            continue

        grad = module.learning_data.grad
        if grad is None:
            missing_grad_names.append(name)
        elif not torch.isfinite(grad).all():
            invalid_grad_names.append(name)
        else:
            finite_grad_names.append(name)

    if invalid_grad_names:
        raise RuntimeError(f"non-finite TQT gradients: {invalid_grad_names}")
    if not finite_grad_names:
        raise RuntimeError(
            "no non-bias TQT threshold received a finite gradient; "
            "check quant_method and the executed model path"
        )
    print("TQT gradients OK:", len(finite_grad_names))
    print("TQT gradients not produced on this path:", missing_grad_names)
```

不要要求每一个 `learning_data` 都必须有梯度：

- bias scale 通常由 input scale 与 weight scale 的乘积传入，bias 自身的 `learning_data` 可能没有有效梯度。
- 输入已经是 `QTensor` 时，下游 input quantizer 可能直接复制上游 scale，而不执行自身的阈值量化路径。
- 未被该 batch 执行的可选分支不会产生梯度。

但是，如果所有有意义的非 bias TQT 量化器都没有梯度，必须停止训练并检查是否误用了 `quant_method: CUDA`、模型路径是否实际执行，或参数是否被错误冻结。

### 8.3 每个 epoch 建议记录

- 任务 loss、主验证指标、学习率和 `threshold_lr`。
- 每个量化器的 `learning_data`、`scale`、`is_calibrate` 和位宽。
- `learning_data.grad` 是否存在、是否有限、绝对值均值和最大值。
- 输入/输出/权重的估计饱和率，以及 float-QAT 的逐层输出误差。
- float、constrained、QAT 三个阶段在同一验证集上的指标差。

`learning_data` 变化但 `scale` 暂时不变通常不是故障，因为 scale exponent 是离散的。应同时观察连续阈值、最终 scale、饱和率和任务指标。

## 9. 刷新并固定最终 scale

训练 forward 会根据当时的 `learning_data` 写入 `scale`，随后 `optimizer.step()` 更新 `learning_data`。因此一个 step 结束后，buffer 中的 `scale` 可能落后最新阈值一次。eval 路径直接使用保存的 `scale`，不会从 `learning_data` 自动重算。

在验证 best checkpoint、保存纯推理 checkpoint和导出 ONNX 前，使用代表性输入执行一次 scale refresh：

```python
import torch
from linger.quant.quantizer import Quantizer

def refresh_tqt_scales(model, example_inputs):
    model.eval()
    for module in model.modules():
        if isinstance(module, Quantizer):
            module.train()

    with torch.no_grad():
        model(*example_inputs)

    # refresh forward 后全模型必须回到 eval，供验证或导出使用。
    model.eval()
```

这里仅把 `Quantizer` 临时设为 train，使其从当前 `learning_data` 计算并写回 `scale`；父模型仍是 eval，避免 BN 统计发生变化。刷新输入必须覆盖实际导出路径。完成验证后若继续训练，再显式调用 `model.train()`。

## 10. 保存和恢复 checkpoint

### 10.1 保存

```python
payload = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler is not None else None,
    "epoch": epoch,
    "best_metric": best_metric,
    "quant_config_path": str(config_path),
    "quant_config_hash": quant_config_hash,
    "model_revision": model_revision,
}
torch.save(payload, "checkpoints/tqt_best_full.pt")
torch.save(model.state_dict(), "checkpoints/tqt_best_state_dict.pt")
```

同时保存训练代码版本、fusion/disable 参数、输入契约、随机种子和评测命令。纯 `state_dict` 便于严格导图，full checkpoint 用于恢复 optimizer 和 scheduler。

### 10.2 恢复

恢复顺序必须是：

```text
build model
-> 复现 trace/fusion
-> linger.init(..., 同一 TQT YAML)
-> strict load TQT model state
-> 创建同结构 optimizer/scheduler
-> load optimizer/scheduler state
-> 继续训练
```

```python
checkpoint = torch.load(checkpoint_path, map_location=device)

model = build_tqt_model_with_same_transform_and_config()
model.load_state_dict(checkpoint["model"], strict=True)

optimizer = build_optimizer_after_linger_init(model)
optimizer.load_state_dict(checkpoint["optimizer"])

if scheduler is not None and checkpoint["scheduler"] is not None:
    scheduler.load_state_dict(checkpoint["scheduler"])
```

不要用 `strict=False` 静默跳过 TQT 状态差异。MOM 和 TQT 的状态 key 不同；从 MOM 迁移时只能按明确规则加载可匹配的业务参数，然后重新校准 TQT 并重建 optimizer，不能加载 MOM optimizer 状态继续训练。

## 11. 量化效果优化

按以下顺序处理，且一次只改变一个主要变量：

1. **确认评测可比**：float 与 QAT 使用同一 checkpoint 业务权重、验证集、预处理、后处理、BN/eval 状态和 metric 实现。
2. **检查量化初始化**：确认校准 batch 有代表性、实际路径均被执行、`is_calibrate` 和 scale 有效，无 NaN/Inf。
3. **检查 TQT 是否真的在学习**：确认后端为 `NATIVE`/`CUDA_GS`、阈值在 optimizer 中、首个 backward 有有限梯度。
4. **分析范围和饱和率**：定位首个误差突增层；高饱和优先增大阈值或改善输入范围，量化步长过粗则考虑减小无效范围。
5. **调学习率**：先降低 QAT 主学习率，给 warm-up 和更长微调周期；阈值震荡时单独降低 `threshold_lr`，不要同时改多项超参。
6. **处理约束阶段掉点**：如果 constrained 模型已经明显掉点，先调整 activation clamp、weight clamp/factor 或结构，再进入 TQT。
7. **保护敏感层**：用逐层误差实验确认第一层、最后一层、head、残差汇合或 recurrent state 是否敏感，再对少量层提高位宽或禁用量化，并验证目标硬件支持。
8. **改善结构**：收敛多分支动态范围，替换不友好的激活/归一化/动态算子，减少异常极值；结构修改后重训 float baseline。
9. **使用蒸馏**：必要时加入 float teacher 的 logit 或 feature distillation，并分别记录任务 loss 和蒸馏 loss。
10. **重新验收部署**：高位宽或禁用层可能恢复精度，但不代表目标平台接受该图；必须重新导图、检查属性并运行目标 runtime。

不建议把切换 `w_like` 当作第一优化手段。默认先使用 weight `abs_max`、activation `top_10`，结合实际饱和率和逐层误差决定是否开发或注册新的校准方法。

## 12. 常见问题排查

| 现象 | 常见原因 | 建议检查 |
| --- | --- | --- |
| loss 下降但阈值不变 | 使用了 `CUDA` 路径；阈值不在 optimizer；参数被冻结 | 检查 YAML、optimizer 参数 id 和首个 backward 梯度 |
| `learning_data` 变化但 `scale` 不变 | scale exponent 取整后的正常离散行为 | 观察是否跨取整边界，同时看饱和率和指标 |
| eval 指标落后于训练 forward | optimizer step 后 `scale` buffer 尚未刷新 | 验证和导出前执行第 9 节 refresh |
| 校准报 `Quantizer has been calibrated` | 重复 calibration 或共享 module 被多次调用 | 只跑一次 forward；检查共享模块；失败后重建模型 |
| MOM 校准访问 `learning_data` 报错 | 对 MOM 使用了当前 TQT 专用 calibration | MOM 直接通过训练 forward 初始化统计量 |
| 一部分阈值没有梯度 | bias 派生 scale、QTensor 边界复制 scale、分支未执行 | 按量化器角色和执行路径诊断，不只统计总数 |
| CPU 上 CUDA extension 报错 | `CUDA/CUDA_GS` 需要 CUDA | CPU 配置改为 `NATIVE` |
| 恢复时 missing/unexpected keys | 模型、fusion、config、disable 设置或 QAT 方法不一致 | 完全复现结构，使用 TQT checkpoint 严格加载 |
| QAT 初始掉点很大 | 校准数据失配、float/constrained 基线本身异常、范围饱和 | 复测三阶段指标，检查逐层误差和饱和率 |
| 导出图与训练结果不一致 | scale 未刷新、输入契约不同、加载不严格或未 eval | 按第 13 节从同一 config 和 checkpoint 重建 |

## 13. 完整 ONNX 导图流程

### 13.1 导图前 Gate

必须同时满足：

- best TQT checkpoint 在固定验证集上复测达标；
- 配置、model revision、fusion、disable 设置和输入契约已归档；
- `learning_data` 和 `scale` 有限且 scale 为正；
- 使用最新阈值执行过 scale refresh；
- 导图模型处于 `eval()`，example inputs 与目标输入的数量、shape、dtype、layout 一致。

### 13.2 重建和导出

```python
import torch
import linger

device = torch.device("cuda")
config_path = "configs/tqt_cuda_gs.yaml"
onnx_path = "artifacts/model_tqt.onnx"

example_inputs = (build_realistic_example_input().to(device),)

model = build_model().to(device)

# 与训练完全一致；没有使用 trace_layers 时不要新增该步骤。
linger.QUANT_CONFIGS.device = device
adapter = linger.trace_layers(model, *example_inputs, fuse_bn=True)
model = linger.init(
    model,
    config_file=config_path,
    disable_module=TRAINING_DISABLE_MODULES,
    disable_submodel=TRAINING_DISABLE_SUBMODELS,
)

state_dict = torch.load(
    "checkpoints/tqt_best_state_dict.pt",
    map_location=device,
)
model.load_state_dict(state_dict, strict=True)

refresh_tqt_scales(model, example_inputs)
model.eval()

with torch.no_grad():
    linger.onnx.export(
        model,
        example_inputs,
        onnx_path,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )
```

多输入模型应让 `example_inputs`、`input_names` 和 `dynamic_axes` 一一对应。动态维只标记部署 runtime 真正支持的维度，不要因为 PyTorch 能导出就假设目标芯片支持任意 shape。

### 13.3 导出后验证

```bash
python3 -c "import onnx; m=onnx.load('artifacts/model_tqt.onnx'); onnx.checker.check_model(m); print('ONNX checker passed')"

python3 lnn-quant-training/scripts/inspect_quant_onnx.py \
  artifacts/model_tqt.onnx \
  --format markdown \
  --fail-on-issues
```

检查内容至少包括：

1. ONNX checker 通过，graph 输入输出名称、shape 和 dtype 正确。
2. 预期模块已转换为 Linger 量化 op，禁用层符合训练配置。
3. weight/activation/bias 位宽、scale 和 platform 属性与 YAML 一致。
4. graph 中没有意外 float fallback、未支持 op 或错误的动态轴。
5. 使用同一组 golden inputs 对比 PyTorch TQT eval 输出和部署 runtime 输出。
6. 具备 Thinker 工具链时完成 `tpacker`、memory report 和 `tvalidator`/目标 runtime 一致性验证。

普通 ONNX Runtime 通常不能执行 Linger custom-domain op，不能仅用“ONNX Runtime 是否直接运行”判断导图成败。最终部署签收以 ONNX 审计、Thinker 打包和目标 runtime 一致性为准。

## 14. 最小端到端模板

下面的模板突出 TQT 特有顺序，数据、分布式训练、AMP 和业务评测函数由项目补充：

```python
import torch
import linger

device = torch.device("cuda")
config_path = "configs/tqt_cuda_gs.yaml"

# 1. 构建并加载基线
model = build_model().to(device)
model.load_state_dict(
    torch.load("checkpoints/float_best.pt", map_location=device),
    strict=True,
)

# 2. 复现训练所需结构变换并量化
example_inputs = get_example_inputs(device)
linger.QUANT_CONFIGS.device = device
linger.trace_layers(model, *example_inputs, fuse_bn=True)
model = linger.init(model, config_file=config_path)

# 3. 单次代表性校准
calibration_inputs = get_calibration_inputs(device)
model.eval()
with torch.no_grad(), linger.calibration():
    model(*calibration_inputs)

# 4. init/calibration 后创建 optimizer
optimizer = build_tqt_optimizer(model)

# 5. QAT；首个 backward 调用 audit_tqt_gradients(model)
best_metric = train_and_validate_tqt(
    model=model,
    optimizer=optimizer,
    scale_refresh_inputs=example_inputs,
)

# 6. 严格复载 best state，刷新 scale 并保存最终状态
best_state = torch.load("checkpoints/tqt_best_state_dict.pt", map_location=device)
model.load_state_dict(best_state, strict=True)
refresh_tqt_scales(model, example_inputs)
torch.save(model.state_dict(), "checkpoints/tqt_export_state_dict.pt")

# 7. eval + no_grad 导图
model.eval()
with torch.no_grad():
    linger.onnx.export(
        model,
        example_inputs,
        "artifacts/model_tqt.onnx",
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )
```

完成训练并不等于完成交付。TQT 的完整闭环是：指标达标、阈值确实参与训练、最终 scale 已刷新、checkpoint 可严格复载、量化 ONNX 属性正确，并在目标工具链和 runtime 上通过一致性验证。
