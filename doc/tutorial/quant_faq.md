## 通用注意事项
1. 两阶段训练：
- 浮点阶段学习率通常设高一点，定点阶段学习率设低一点
- 浮点clamp的clamp_modules设置要和定点训练时的quant_modules保持一致，一般默认浮点做了clamp的话，定点的这层也会相应做量化
- 浮点clamp阶段，一般情况下初始loss有个明显的上升，之后再回落收敛到基线，证明clamp起到了限制作用，并且网络开始学到东西；定点阶段时，保证初始loss与浮点基线loss相差不大，且很快收敛到基线。若初始loss超出太多的话，很难收敛到基线的loss结果，即使收敛也相当于在量化阶段重新从头训练的，浮点学到的权重分布会被打乱，在某些回归任务上这样loss即使正常，实际测试效果也不会很好
2. trace_layer 只支持使用一次，多次会导致前面被覆盖，hook被清空，导致加载的融合参数
3. normalize_layer和 init 中 replace_tuple 是需要一一对应的，disable_normalize 和 disable_normalize 接口也是需要匹配调用的
4. 导图：
- 导onnx图时，需要保证 linger 的所有设置和训练时完全一致
- torch.onnx.export中，将opset_version=12,  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK 这两个选项固定下来，可避免很多报错（忘加aten选项，会报 bad node clip的错误）
5. 训练过程中不要使用 with_no_grad() 方式  linger不支持这样使用，梯度反向时会报错
6. linger 不支持混合精度量化训练，只支持全浮点网络做定点训练 


## 常见问题定位与解决
1. 导出 onnx 提示 running_x > 0 assert报错，首先确认init确实有训练，或者加载的量化模型中确实有量化参数，其次网络中有一部分并没有在forward中使用
2. 导出 onnx 提示 scale_o warning, 打开 onnx 确认对应 id 是否 dequant 添加正确及其中属性是否传递正确
3. torch 1.6 版本后，导出onnx前需要with torch.no_grad(), 如果不使用with torch.no_grad()，则会报以下错误
RuntimeError: isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/autograd/functions/utils.h":59, please report a bug to PyTorch.
4. 如果warning提示为iqcat  eval module has iqcat layer while do not match training module 的话，该问题比较复杂而且warning不影响运行，仅在导出onnx时有不同处理，一致性影响不大的话可以忽略。可以查看保存的量化pt文件中是否有iqcat的对应scale参数，如有 则可以在加载pt之前走一遍前向，将网络前向固定住，这样会自动去加载对应的iqcat参数，否则导出onnx时此处会变成浮点op
5. 如果 loss 变 nan，减小学习速率或者增大 batch_size，还有判断是否有量化输出为0值，导致某些op（如sqrt\atan2等）后面反向挂掉了
6. 在导出onnx后，不要再继续训练，此时权重等参数都变成定点值了，继续训练会报错。iqtensor.py中  zero encountered in true_divide
7. leakyrelu 不支持量化，但导出op后未添加quant节点，由于打开了inplace=True选项，导致直接对tensor做修改，继续当成了IQtensor 直传下去，导图时就会报错
8. 发现导图的onnx中scale与实际eval定位的scale值不匹配，可能由于导图前在train模式下走了前向，重新统计了running值，导致导图不一致
9. 1.9.0官方文档说明，输入的最后一个参数不能是dict 类型，不然导出onnx时，此输入强制会变成空，需要在torch.onnx.export 后改成(x, meta, { })输入才行
10. iqsigmoid使用及print显示问题：nn.sigmoid() 不会直接替换，同torch.sigmoid一样，走遍前向再print 才是iqsigmoidlayer()
11. 报错 "ConvFunction BAckward" object has no attribute 'clamp_data'，eval 模式下走backward的原因  
12. torch版本不同，导出的op也可能会有变化  
13. linger只会识别 nn.Avgpool2d()的写法，其他Adapt_avgpool的用法，会走浮点逻辑，Maxpool 同理
14. 对lstm做量化时，之前和之后的pack_padded_sequence和pad_packed_sequence 需要写全，不能用from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence 使用
   直接写全torch.nn.utils.rnn.pack_padded_sequence(_,_,_,_)   torch.nn.utils.rnn.pad_packed_sequence(_,_,_,_)，不然linger替换不了函数指针，会导致运行出错
15. 量化时学习率不能设太大，要在浮点最后训练的学习率基础上，再降一个数量级为好，不然容易出现loss爆炸变很大的情况
16. 定点训练测试时，不能只跑几个batch就去走eval测试，这时的running值还没有统计正确，会导致结果很差，推荐跑完100个batch后再做测试
17. 量化训练时加载原始浮点的checkpoint时，load_state_dict中strict参数最好不要设为false，这个选项会忽略其中key不对应的情况，可能加载的参数就没对上，后面效果自然不行
18. linger不支持其中某一部分的layer不做反向（requird_grad = False）会报如下错：
				one of the differentialted tensors does not require grad 。。。。
19. onnx导出时存在 过slice 算子之后量化tensor变为浮点tensor的情况，可能是由于slice之前的op  存在未识别/不支持量化的op，常见的情况为transpose的torch写法问题，具体情况需具体分析
20. 报错 Invoked with ：Tensor= onnx::Constant(), 'value', [4,3,2,1,], (Occurred when translating LSTMPIntFunction)
       batch_length的问题，导图时需要将其改为torch.tensor，否则还为原始的list 就会导致报错
21. 量化初始效果 loss 上升很多，首先确认是否加载checkpoint正确，尤其是设置了trace_layer后容易出现这个问题。定位：可以关闭init 仅打开trace_layer  查看loss是否与浮点相同
	若相同说明加载checkpoint正确，否则需要重新对齐浮点checkpint的key。由于trace_layer后很多key被修改，所以以往的那种for循环遍历的加载方式会导致加载错误
	推荐：可以在原始浮点用原始的方式加载完checkpoints之后，直接save一个state_dict下来，然后量化的时候，直接load保存的这个state_dict，就不用 for 循环遍历对比再加载了 
22. 测试时报错 AttributeError: Can't pickle local object 'trace_layers.<locals>.pre_hook'
	验证为多进程运行原因，改为单进程即可，一般推荐使用单卡单进程做测试，避免大部分问题
23. 量化loss不下降，确认使用的optimizer是否关联到正确的parameter上了，可能未更新权重参数，可以保存几组不同的pt数据，查看权重更新情况
24. 针对最终导出的onnx中 iqcat   iqadd    iqsigmoid，两边scale差距太大等情况，linger关闭iqtensor的相关设置  linger.SetIQTensorCat(False)
25. 针对直接量化效果理想，但实测在某些场景中不理想，可以单独统计一下这组数据的running和scale值（只跑前向,不backward），看看和最终的训练模型差距大否
26. batchnorm加入量化，loss上升非常明显，先确认是否是由于fuse_bn的原因导致，其次由于bn量化是把 (x-mean)/ var * alpha + belta  转化为 (alpha/var) x + (belta - mean/var)，定位一下 (alpha/var) 的值是否太极端
27. 关闭batchnorm层running_mean\running_var的更新，置momentum为0
主要用于浮点fine-tune或定点量化时  不想更新bn层的running_mean/var的值，因为有时浮点已经在大数据集上统计好了对应参数，fine-tune或量化时一般只过一个小数据集即可，这样可能会打乱浮点好不容易训好的分布，导致效果变差。
主要用法如下：
```python
linger.SetBnMomentumUpdate(disable = True)
```
此语句需加在 trace_layer 之后，normalize设置及量化init之前，推荐添加完在intx的所有设置后print(model) 查看bn层的momentum是否已置0，此接口会将normalizebn、NormalizeConvBN1d、normalizeconvbn2d、bnint等op的momentum置零，保证其不更新running_mean、running_var，如果bn层不做normalize、不做量化，此设置对其不起作用

28. 一些常见的函数量化选项 (默认为开启，可通过以下设置关闭)
```
# 关闭 + 的量化，网络不会再出现iqadd的量化op
linger.SetIQTensorAdd(False)
# 类似的还有 'SetIQTensorAdd' 'SetIQTensorClamp','SetIQTensorCat', 'SetIQTensorSigmoid', 'SetIQTensorTanh','SetIQTensorDiv', 'SetIQTensorMul', 'SetIQTensorSum'
linger.SetFunctionBmmQuant(True)  #此选项默认为关闭，控制torch.bmm的量化与否
linger.SetBnMomentumUpdate(True)  #此选项默认为关闭，控制bn层running_mean/running_var的值分布不更新
```

## 网络搭建推荐
1. 搭建网络时，若nn.Linear层后有bn层直连，推荐将此linear层改为等效的1*1的卷积，这样在最后推理实现时可以完成conv-bn的融合，加快推理效率
2. 若网络中conv-bn的直连结构较多，使用linger进行量化时，推荐使用intx的trace_layer设置，完成convbn的自动融合，导图时即只会有一个conv存在
3. 网络forward代码中推荐使用  tensor.transpose/ tensor.squeeze / tensor.unsqueeze 等写法，其等效的torch.transpose(temnsor, ...)等写法可能暂时不被linger识别，导图时会出现量化op浮点的断连情况
4. 网络定义中推荐使用 nn.Conv2d / nn.Linear/ nn.BatchNorm 等写法，其等效的在forward中直接调用的F.Conv2d等写法可能暂时不被 linger 识别，导图时会出现op未量化的断连情况