# ONNX----从pytorch 动态模型到静态模型

## 什么是ONNX
Open Neural Network Exchange（ONNX，开放神经网络交换）格式，是一个用于表示深度学习模型的标准，可使模型在不同框架之间进行转移。
ONNX为AI模型提供开源格式。 它定义了可扩展的计算图模型，以及内置运算符和标准数据类型的定义。 最初的ONNX专注于推理所需的功能。 ONNX解释计算图的可移植，它使用graph的序列化格式。 它不一定是框架选择在内部使用和操作计算的形式。 例如，如果在优化过程中操作更有效，则实现可以在存储器中以不同方式表示模型。

## 如何导出ONNX模型 
### 一个简单的入门示例
torch官方提供了从PyTorch导出onnx的接口：
```python
torch.onnx.export(model, args, f, export_params=True, verbose=False, training=TrainingMode.EVAL, input_names=None, output_names=None, aten=False, export_raw_ir=False, operator_export_type=None, opset_version=None, _retain_param_name=True, do_constant_folding=True, example_outputs=None, strip_doc_string=True, dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None, enable_onnx_checker=True, use_external_data_format=False)
```
常用参数：
- `export_params`：该参数默认为True，也就是会导出训练好的权重；若设置为False，则导出的是没有训练过的模型。
- `verbose`：默认为False，若设置为True，则会打印导出onnx时的一些日志，便于分析网络结构。
- `opset_version`：onnx op集合版本号，在linger中通常设置为11或12。
- `dynamic_axes`：可以指定哪些维度是变化的，例如当我们导出模型的时候，输入的第一个维度是batch_size，但是这个维度应该是动态变化的，可以通过该参数指定这个可以动态变化。

导出的简单示例
``` python
import torch
import torch.onnx
torch_model = Model()
# set the model to inference mode
torch_model.eval()
dummy_input = torch.randn(1,3,244,244)
torch.onnx.export(torch_model,dummy_input,"test.onnx")

```

### 使用 linger 导出 onnx
如果调用`linger.init(...)`接口后，推荐使用使用`linger.onnx.export`进行调用;

```python
import linger
.....
linger.init(...)
linger.onnx.export(...)
```

### 导出支持动态输入大小的图

``` python
linger.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
```

其中 dynamic_axes使用有几种形式:

- 仅提供索引信息
例如下例子表示 把`input_1`的`0,2,3`维作为动态输入，第`1`仍然保持固定输入，'input_2'第`0`维作为动态输入，`output`的`0,1`维作为动态输入，对于动态输入的维度，PyTorch会自动给该维度生成一个名字以替换维度信息
``` python
dynamic_axes = {'input_1':[0, 2, 3],
                  'input_2':[0],
                  'output':[0, 1]}

```

- 对于给定的索引信息，指定名字
对于`input_1`，指定动态维0、1、2的名字分别为`batch`、`width`、`height`，其他输入同理
``` python
dynamic_axes = {'input_1':{0:'batch',
                             1:'width',
                             2:'height'},
                  'input_2':{0:'batch'},
                  'output':{0:'batch',
                            1:'detections'}
```
- 将上面两者进行混用
``` python
dynamic_axes = {'input_1':[0, 2, 3],
                  'input_2':{0:'batch'},
                  'output':[0,1]}
```
### 带有可选参数的导出
例如想命名输入输出tensor名字或者比较超前的op可以加上`torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK`
``` python
import torch
import torch.onnx
torch_model = ...
# set the model to inference mode
torch_model.eval()
dummy_input = torch.randn(1,3,244,244)
linger.onnx.export(torch_model,dummy_input,"test.onnx",
                    opset_version=11,input_names=["input"],output_names=["output"],operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
```
### torch.no_grad()
torch 1.6 版本后，需要`with torch.no_grad()`,即

``` python
import torch
import torch.onnx
torch_model = ...
# set the model to inference mode
torch_model.eval()
dummy_input = torch.randn(1,3,244,244)
with torch.no_grad():
    linger.onnx.export(torch_model,dummy_input,"test.onnx",
                        opset_version=11,input_names=["input"],output_names=["output"],operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
```
`警告`：如果不使用`with torch.no_grad()`，则会报以下错误
>RuntimeError: isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/autograd/functions/utils.h":59, please report a bug to PyTorch.


