# 量化分布分析工具使用指南

`tools/analyze_quantization.py` 用于分析 PyTorch 模型的权重、偏置和输入/输出激活分布。工具独立运行，不依赖 `linger`，分析结果以 HTML 报告和 JSON/CSV 数据保存。

## 环境准备

工具可以在任意能运行项目代码的 Python 环境中使用，不要求特定的机器、操作系统或 Conda 环境。准备一个 Python 环境，安装与模型代码匹配的 PyTorch；使用 `.npy`/`.npz` 输入文件时安装 NumPy，使用 YAML 配置文件时安装 PyYAML。

从仓库根目录执行命令，确保模型代码和它依赖的 Python 包可以被导入。例如可以使用 virtualenv、Conda 或系统 Python：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows 请运行：.venv\Scripts\activate
python -m pip install torch           # 按实际平台选择 PyTorch 安装命令
```

## `--model-factory` 是什么

`--model-factory` 指向一个**可调用对象**，用于创建模型结构，格式为 `模块:名称` 或 `文件.py:名称`。它可以是项目已有的模型构造函数，也可以是 `torch.nn.Module` 子类；不需要重新实现模型算法。

工具必须先创建模型，再把 checkpoint 中的 `state_dict` 加载进去。checkpoint 通常只保存参数，不包含如何实例化模型的代码，因此工具无法仅凭 `.pt` 文件自动推断任意模型结构。当前版本要求提供这个入口；优先复用项目已有的构造函数即可。

例如已有 `my_model.py`：

```python
import torch.nn as nn

def build_model(width=128):
    return nn.Sequential(nn.Linear(width, 10), nn.ReLU())
```

如果模型类本身可以直接实例化，也可以直接指定类名：

```bash
python tools/analyze_quantization.py \
  --model-factory my_model.py:MyNet \
  --checkpoint checkpoints/model.pt \
  --inputs artifacts/calibration.pt
```

如果项目已有 `models.py:build_model`，无需新增文件；只需把该入口传给 `--model-factory`。构造函数需要参数时，用 `--factory-kwargs` 传入 JSON：

```bash
--factory-kwargs '{"width":128,"num_classes":10}'
```

只有拿到 checkpoint、但没有对应模型代码或构造入口时，当前工具无法完成分析；需要先提供与 checkpoint 匹配的模型定义。

## 基本用法

使用校准输入分析模型：

```bash
python tools/analyze_quantization.py \
  --model-factory my_model.py:build_model \
  --checkpoint checkpoints/qat.pt \
  --float-checkpoint checkpoints/float.pt \
  --inputs artifacts/calibration.pt \
  --max-batches 32 \
  --output-dir artifacts/quant-analysis
```

`--checkpoint` 是待分析模型的权重；`--float-checkpoint` 可选，用于与浮点模型配对比较。省略 checkpoint 时分析模型工厂返回的初始权重。

没有校准文件时，可以用固定形状生成随机输入：

```bash
python tools/analyze_quantization.py \
  --model-factory my_model.py:build_model \
  --input-shape 8,3,224,224 \
  --output-dir artifacts/quant-analysis
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--config config.yaml` | 从 YAML/JSON 读取配置；命令行参数优先 |
| `--inputs FILE` | 校准输入文件 |
| `--input-shape N,N,...` | 未提供输入文件时生成随机输入 |
| `--input-dtype float32` | 随机输入的数据类型 |
| `--factory-kwargs '{"width":128}'` | 模型构造参数 |
| `--max-batches 32` | 最多处理的 batch 数 |
| `--max-samples 100000` | 激活采样上限 |
| `--histogram-bins 64` | 直方图分箱数 |
| `--device auto` | `auto`、`cpu` 或指定 CUDA 设备 |
| `--include PATTERN` | 只分析匹配的模块，可重复使用 |
| `--exclude PATTERN` | 排除匹配的模块，可重复使用 |
| `--seed 1234` | 随机输入和采样种子 |
| `--non-strict` | 允许 checkpoint 存在 missing/unexpected keys |

## 输入文件格式

`--inputs` 支持以下形式：

- `.pt`/`.pth`：Tensor、位置参数 tuple/list、关键字参数 dict，或 `{"batches": [...]}`；
- `.npy`：单个输入 Tensor；
- `.npz`：每个 key 作为一个关键字输入。

Tensor、tuple/list、dict 分别对应 `model(x)`、`model(*batch)`、`model(**batch)`。未提供 `--inputs` 时必须提供 `--input-shape`。

也可以使用配置文件，例如：

```yaml
model_factory: my_model.py:build_model
checkpoint: checkpoints/qat.pt
float_checkpoint: checkpoints/float.pt
inputs_path: artifacts/calibration.pt
output_dir: artifacts/quant-analysis
max_batches: 32
device: auto
seed: 1234
```

执行：

```bash
python tools/analyze_quantization.py --config config.yaml
```

## 输出目录

```text
report.html          # 离线 HTML 报告，直接用浏览器打开
summary.json         # 运行配置、checkpoint 和总体摘要
layer_records.json   # 每层完整统计及异常信息
layer_metrics.csv    # 便于脚本或表格工具读取
figures/*.svg        # 可独立使用的矢量图
```

重点查看 `report.html`：

- Weight、Bias、Activation 三个区域分别展示分布图；
- 每个区域只占约两行高度，其余图卡在区域内滚动查看；
- 可按层名关键词搜索；
- 可分别选择“全部 / 只看浮点 / 只看量化”；
- 图表支持悬停、键盘方向键查看分箱数据，并可导出 SVG；
- 异常记录会在图卡和异常入口中标出，可直接定位到对应层。

报告不需要启动 Web 服务，直接打开 `report.html` 即可。若某类张量没有采集到，对应区域会保留并显示提示。

## 注意事项

- checkpoint 默认严格加载。只有确认 missing/unexpected keys 符合预期时，才使用 `--non-strict`；
- 激活直方图来自有上限的采样，不能替代全量 Tensor 的精确统计；
- 分布图的浮点/量化共同分箱是基于原始分箱的区间重建估计；
- 报告中的异常是定位线索，不等同于模型精度结论；
- 不要将 checkpoint、数据集或运行日志提交到代码仓库。
