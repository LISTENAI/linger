

## 1. wb_analyse分析工具

```python
#                         原始浮点基线权重 pth          分析日志保存地址
linger.wb_analyse('data.ignore/tool_test.pt',  'data.ignore/wb_anylse.log')
#-------------------------------------------------------------------------------------
```
## or
```python

checkpoint = torch.load("best_checkpoint.pth")

checkpoint = checkpoint['state_dict']
#                  也可以传入加载后的pth     分析日志保存地址默认为./wb_analyse.log
linger.wb_analyse(checkpoint)
```

```python
'''
日志如下所示  Multiple = Max / Mean , Versu = Max / Dynamic 
+-------------------------------------------------------+--------------------+--------------------+-----------------+--------------------+-----------------+
|                       Layer_name                      |        Mean        |        Max         |     Multiple    |    Dynamic 0.99    |      Versu      |
+-------------------------------------------------------+--------------------+--------------------+-----------------+--------------------+-----------------+
|               encoder.conv1.conv.weight               |   tensor(0.8093)   |   tensor(4.0748)   |  tensor(5.0348) |   tensor(3.2437)   |  tensor(1.2562) |
|                encoder.conv1.conv.bias                |   tensor(0.1000)   |   tensor(0.1000)   |  tensor(1.0000) |   tensor(0.1000)   |    tensor(1.)   |
|                encoder.conv1.bn.weight                |   tensor(0.4724)   |   tensor(1.2380)   |  tensor(2.6208) |   tensor(1.0338)   |  tensor(1.1975) |
|                 encoder.conv1.bn.bias                 |   tensor(0.3030)   |   tensor(1.9110)   |  tensor(6.3075) |   tensor(1.5030)   |  tensor(1.2714) |
|          encoder.conv1.bn.num_batches_tracked         |  tensor(6185962)   |  tensor(6185962)   |    tensor(1.)   |  tensor(6185962)   |    tensor(1.)   |
+-------------------------------------------------------+--------------------+--------------------+-----------------+--------------------+-----------------+
'''
```

## 2、 out_analyse分析工具(初版，复杂模型可能不适用)
### 分析网络每一层的输出分布，日志形式同权重分析日志

```python
model = resnet50().cuda()
### 加载训练好的浮点checkpoint
model.load_state_dict(checkpoint)
### 给定一个网络的真实的典型输入，不要用随机数据
typical_input = torch.randn([1,3,224,224]).cuda()

with linger.Dumper() as dumper:
    # model.eval()
    dumper.analyse_layer_output(model,match_pattern="root.")   # match_pattern 可支持查看对应哪些层
    model(typical_input) #跑一遍前向
    dumper.save_out_analyse_log(save_log_path="Analyse_layer_output.log") #日志保存路径
## 此接口会在当前目录生成一个名为"Analyse_layer_output.log"的文件
```
### 根据日志中Multiple = Max / Mean , Versu = Max / Dynamic0.99 两个的数值进行分析
### ① 一般情况希望输出分布的均值和最值不要相差太大  这两个倍数供参考
### ② 当Versu大于10倍时，说明此层输出的分布最值有明显异常，对量化很不友好  ，日志中会在此层数据下面打印！！！提示
### ③ 一般推荐对于异常层来说，对其进行精细的normalize约束设置，向均值方向约束（不代表约束到均值），目的仅为抹除异常的最值即可

```python
'''
日志如下所示  Multiple = Max / Mean , Versu = Max / Dynamic 
+----------------------------+----------------+-----------------+--------------------+----------------+--------------------+
|         Layer_name         |      Mean      |       Max       | Multiple(Max/Mean) |  Dynamic 0.99  | Versu(Max/Dynamic) |
+----------------------------+----------------+-----------------+--------------------+----------------+--------------------+
|         root.conv1         | tensor(0.7991) |  tensor(4.9494) |   tensor(6.1935)   | tensor(1.6482) |   tensor(3.0028)   |
|          root.bn1          | tensor(1.1000) | tensor(11.8600) |  tensor(10.7815)   | tensor(2.5022) |   tensor(4.7399)   |
|         root.relu          | tensor(0.4383) |  tensor(7.7810) |  tensor(17.7513)   | tensor(0.8851) |   tensor(8.7912)   |
|        root.maxpool        | tensor(0.3245) |  tensor(7.7810) |  tensor(23.9802)   | tensor(0.8358) |   tensor(9.3091)   |
|    root.layer1.0.conv1     | tensor(0.7606) |  tensor(7.7810) |  tensor(10.2294)   | tensor(1.4041) |   tensor(5.5418)   |
|     root.layer1.0.bn1      | tensor(0.6418) |  tensor(4.2427) |   tensor(6.6106)   | tensor(1.5714) |   tensor(2.7000)   |
|     root.layer1.0.relu     | tensor(0.3977) |  tensor(2.7954) |   tensor(7.0291)   | tensor(0.8981) |   tensor(3.1128)   |
|    root.layer1.0.conv2     | tensor(0.1164) |  tensor(2.7954) |  tensor(24.0151)   | tensor(0.5088) |   tensor(5.4937)   |
+----------------------------+----------------+-----------------+--------------------+----------------+--------------------+
'''
```

## 3、 linger导出的onnx图中  dequant错乱 或者 图中节点有断裂，可参照下面过程操作

### torch.onnx.export提供以下选项供调试：
-   is_update_dequant = True      # 设为False，关闭添加dequant节点（&删除identity结点）的过程  
-   is_scoped_info    = True      # 设为False，关闭添加和删除节点scope name信息的过程  
-   debug_dump        = False     # 设为True，保存中间各步的onnx结果，仅供调试使用, （建议使用此选项时不要对以上两个选项做修改）


```python
dummy_input = torch.ones(1,3,224,224)  #模拟输入
with torch.no_grad():
        linger.onnx.export_debug(net, dummy_input,"export_debug.onnx",export_params=True,opset_version=12,operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,is_update_dequant = False,is_scoped_info=False,debug_dump=False)
```

## 4、当使用旧版linger导出的onnx图中仅有 dequant添加错乱情况 ，可参照下面过程修复

- conda create 新环境 安装最新版linger (方法仅供参考，保证有一个最新版的linger版本即可)
- linger.fix_dequant(ori_onnx, False)   ##原始出错的onnx模型名称 | 是否检测修复后onnxinfer能否运行(设True时需已安装onnxinfer)
- 最后将修复好的onnx保存为 后缀多了_fix.onnx

```python
##                    原始出错的onnx模型名称      | 是否检测修复后onnxinfer能否运行
linger.fix_dequant("dbpagec2_wrong.onnx",            False)
```
