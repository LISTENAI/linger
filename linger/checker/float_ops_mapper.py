import torch.nn.functional as F
import torch
from linger.checker.utils import get_param,register_op
from onnx import numpy_helper
import numpy as np

@register_op(op_type="Identity")
def identity(inputs, kwargs):
    return inputs[0]

@register_op(op_type="Conv")
def conv(inputs, kwargs):
    input_length = len(inputs)
    assert input_length == 2 or input_length == 3,"Conv ops：the number of inputs is wrong, \
        expect 2 or 3, but {}, the length of List must be 2([input,weight]) or 3([input,weight,bias]) ".format(input_length)
    if input_length == 2:
        input,weights= inputs
        bias = None
    else:
        input,weights,bias = inputs
    if weights.ndim ==4:
        dilations = tuple(get_param(kwargs,'dilations'))
        groups = int(get_param(kwargs,'group'))
        pads = tuple(get_param(kwargs,"pads")[:2])
        strides = tuple(get_param(kwargs,"strides"))
        
        return F.conv2d(input,weights,bias,stride=strides,padding=pads,dilation=dilations,groups=groups)
    elif weights.ndim == 3:
        dilations = tuple(get_param(kwargs,'dilations'))
        groups = int(get_param(kwargs,'group'))
        pads = get_param(kwargs,"pads")[0]
        strides = tuple(get_param(kwargs,"strides"))
        
        return F.conv1d(input,weights,bias,stride=strides,padding=pads,dilation=dilations,groups=groups)
    else:
        assert False, "Conv ops only support Conv2d and Conv1d currently, if you want to support more, please contact cgxu2!!!"

@register_op(op_type="ConvTranspose")
def convTranspose(inputs, kwargs):
    input_length = len(inputs)
    assert input_length == 2 or input_length == 3,"ConvTranspose ops：the number of inputs is wrong, \
        expect 2 or 3, but {}, the length of List must be 2([input,weight]) or 3([input,weight,bias]) ".format(input_length)
    if input_length == 2:
        input,weights= inputs
        bias = None
    else:
        input,weights,bias = inputs
    dilations = tuple(get_param(kwargs,'dilations'))
    groups = int(get_param(kwargs,'group'))
    pads = tuple(get_param(kwargs,"pads")[:2])
    strides = tuple(get_param(kwargs,"strides"))
    out_padding = tuple(kwargs.get("output_padding",(0,0)))
    return F.conv_transpose2d(input,weights,bias,stride = strides,padding = pads, output_padding= out_padding, groups=groups,dilation=dilations)

@register_op(op_type='ATen')
def aten(inputs, kwargs):
    eps = get_param(kwargs, "eps")
    normalized_shape = get_param(kwargs,"normalized_shape")
    return F.layer_norm(inputs[0], normalized_shape, weight=inputs[1], bias=inputs[2], eps=eps)

@register_op(op_type="Abs")
def abs(inputs, kwargs):
    return torch.abs(inputs[0])

@register_op(op_type="Sin")
def sin(inputs, kwargs):
    return torch.sin(inputs[0])

@register_op(op_type="Cos")
def cos(inputs, kwargs):
    return torch.cos(inputs[0])

@register_op(op_type="Sqrt")
def sqrt(inputs, kwargs):
    return torch.sqrt(inputs[0])

@register_op(op_type='LogSoftmax')
def logsoftmax(inputs, kwargs):
    axis = get_param(kwargs, "axis")
    return torch.log_softmax(inputs[0], axis)

@register_op(op_type="Less")
def less(inputs, kwargs):
    return torch.less(inputs[0], inputs[1])

@register_op(op_type="Log")
def log(inputs, kwargs):
    return torch.log(inputs[0])

@register_op(op_type='LeakyRelu')
def leakyRelu(inputs, kwargs):
    alpha = get_param(kwargs,"alpha")
    return F.leaky_relu(inputs[0], negative_slope=alpha)

@register_op(op_type='Erf')
def erf(inputs, kwargs):
    return torch.erf(inputs[0])

@register_op(op_type='GlobalAveragePool')
def global_average_pool(inputs, kwargs):
    if len(inputs[0].shape)==4:
        return F.adaptive_avg_pool2d(inputs[0],(1,1))
    if len(inputs[0].shape)==3:
        return F.adaptive_avg_pool1d(inputs[0],(1))

@register_op(op_type='Flatten')
def flatten(inputs, kwargs):
    axis = get_param(kwargs,"axis")
    return torch.flatten(inputs[0],axis)

@register_op(op_type='Tile')
def tile(inputs, kwargs):
    return inputs[0].repeat(inputs[1].int().tolist())

@register_op(op_type='Gemm')
def gemm(inputs, kwargs):
    ret = None
    if len(inputs) ==2:
        ret = F.linear(inputs[0],inputs[1],None)
    else:
        ret = F.linear(*inputs)
    return ret

@register_op(op_type='Sub')
def sub(inputs, kwargs):
    return inputs[0] - inputs[1]  # Tensor - Scalar (Scalar,Tensor)

@register_op(op_type='Mul')
def mul(inputs, kwargs):
    return inputs[0] * inputs[1]  # Tensor - Scalar (Scalar,Tensor)

@register_op(op_type='Add')
def add(inputs, kwargs):
    return inputs[0] + inputs[1]  # Tensor - Scalar (Scalar,Tensor)

@register_op(op_type='Div')
def div(inputs, kwargs):
    return inputs[0] / inputs[1]  # Tensor - Scalar (Scalar,Tensor)

@register_op(op_type='Constant')
def constant(node):
    constant_outputs = torch.from_numpy(numpy_helper.to_array(node.attribute[0].t).copy())
    return constant_outputs

@register_op(op_type='Reshape')
def reshape(inputs,kwargs):
    inputs[1] = inputs[1].reshape(-1)
    if len(inputs[1]) < inputs[0].ndim:
        input_size = inputs[0].shape
        shape_size = [input_size[idx] if value ==0  else int(value) for idx, value in enumerate(inputs[1])]
        return inputs[0].reshape(shape_size)
    else:
        if isinstance(inputs[1],np.ndarray): 
            return inputs[0].reshape(inputs[1].astype(np.int64).tolist())
        else:
            return inputs[0].reshape(inputs[1].tolist())

@register_op(op_type='Transpose')
def transpose(inputs, kwargs):
    perm = get_param(kwargs,'perm')
    if isinstance(inputs[0], torch.Tensor):
        return inputs[0].permute(perm)
    else:
        return inputs[0].transpose(perm)

@register_op(op_type="ReduceMean")
def reducemean(inputs, kwargs):
    axes = get_param(kwargs,'axes')
    keepdims = get_param(kwargs,'keepdims')
    return inputs[0].mean(dim = axes,keepdim = bool(keepdims))

@register_op(op_type="ReduceMax")
def reducemax(inputs, kwargs):
    axes = get_param(kwargs,'axes')
    keepdims = get_param(kwargs,'keepdims')
    out,_ = torch.max(inputs[0], dim = axes[0],keepdim = bool(keepdims))
    return out

@register_op(op_type="Unsqueeze")
def unsqueeze(inputs, kwargs):
    axes = get_param(kwargs,'axes')
    if isinstance(inputs[0],torch.Tensor):
        return inputs[0].unsqueeze(dim = axes[0])
    else:
        return np.expand_dims(inputs[0],axis = axes)
    
@register_op(op_type="Concat")
def concat(inputs, kwargs):
    axis = get_param(kwargs,'axis')
    all_tensor_flag = True
    for input_single in inputs:
        if not isinstance(input_single, torch.Tensor) :
            all_tensor_flag = False
            break
    
    if not all_tensor_flag :
        new_inputs = [input_single.detach().cpu().numpy() if isinstance(input_single,torch.Tensor) else input_single for input_single in inputs]
        return np.concatenate(new_inputs,axis= axis)
    else:
        return torch.cat(inputs,dim = axis)

@register_op(op_type="Shape")
def shape(inputs, kwargs):
    return torch.tensor(list(inputs[0].shape), dtype=torch.int64)


@register_op(op_type="Gather")
def gather(inputs, kwargs):  #please refer to test_onnx_averagepool_iq samples in test_onnx_runner.py
    axis = kwargs.get('axis',0)
    if "parameter_bits" in kwargs:
        import linger
        return linger.EmbeddingInt.run_onnx_embedding(inputs, kwargs)
    if_torch = isinstance(inputs[0], torch.Tensor)
    if not isinstance(inputs[0], torch.Tensor):
        inputs[0] = torch.tensor(inputs[0]) if not isinstance(inputs[0], np.ndarray) else torch.from_numpy(inputs[0].copy())
    if not isinstance(inputs[1], torch.Tensor):
        inputs[1] = torch.tensor(inputs[1]) if not isinstance(inputs[1], np.ndarray) else torch.from_numpy(inputs[1].copy())

    if inputs[1].numel() == 1: # example : a = torch.randn([1,3,224,224]); a[:,:,2]
        slice_list = [":"]*inputs[0].ndim
        slice_list[axis] = str(inputs[1].item())
        slice_str = ','.join(slice_list)
        output = eval("inputs[0][{}]".format(slice_str))
    else:
        inputs[0] = inputs[0].transpose(0,axis)
        # output = F.embedding(torch.LongTensor(inputs[1]),inputs[0])
        output = inputs[0][inputs[1]]
        output =  output.transpose(axis, 0)
    
    if not if_torch:
        return list(output.detach().numpy()) if output.numel() > 1 else output.item()
    return output


@register_op(op_type='BatchNormalization')
def batchnormalization(inputs, kwargs):
    epsilon = get_param(kwargs, 'epsilon')
    momentum = get_param(kwargs, 'momentum')
    return F.batch_norm(inputs[0],inputs[3],inputs[4],inputs[1],inputs[2],False,momentum,epsilon)

@register_op(op_type='Slice')
def slice(inputs, kwargs):
    start = int(inputs[1].item())
    end = int(inputs[2].item())
    axes = 0
    step = 1
    if len(inputs) >3:
        axes = int(inputs[3].item())
    if len(inputs) == 5:
        step = int(inputs[4].item())

    if isinstance(inputs[0],torch.Tensor):
        slice_list = [":"]*inputs[0].ndim
        slice_list[axes] = '{}:{}:{}'.format(start,end,step)
        slice_str = ','.join(slice_list)
        output = eval("inputs[0][{}]".format(slice_str))
    else:    # process torch.size, because this type often occurs [1,3,224,224]
        slice_str = '{}:{}:{}'.format(start,end,step)
        output = np.array((eval("inputs[0][{}]".format(slice_str))))
    return output

@register_op(op_type='AveragePool')
def averagepool(inputs, kwargs):
    kernel_shape = get_param(kwargs, "kernel_shape")
    strides = get_param(kwargs, "strides")
    ceil_mode = bool(kwargs.get("ceil_mode",0))
    pads = tuple(kwargs.get("pads",[0,0,0,0]))[:2] # argument 'padding' must be tuple of ints, not str
    return F.avg_pool2d(inputs[0],kernel_size = kernel_shape,stride = strides,padding = pads,ceil_mode = ceil_mode)

@register_op(op_type="Pad")
def pad(inputs, kwargs):
    # the pads inputs in onnx is 'x1_begin,x2_begin ...,x1_end,x2_end...'
    # the pads used in F.pad is 'x4_start,x4_end....x1_start,x1_end'
    # explanation: "x4_start" refers to adding n values before the 4th dimension,"x4_end" refers to adding n values after the 4th dimension
    mode = get_param(kwargs, "mode")
    if isinstance(inputs[1],np.ndarray):
        inputs[1] = np.flip(inputs[1].reshape(2,-1),axis = 1).transpose(1,0).flatten()  # change onnx pads into F.pad format
    else:
        inputs[1] = inputs[1].reshape(2,-1).flip(dims = [1]).transpose(1,0).flatten()
    if len(inputs) ==3:  # user specified the pad_value(constant value)
        return F.pad(inputs[0],tuple(inputs[1]),mode, inputs[2].item())
    else:
        return F.pad(inputs[0],tuple(inputs[1]),mode,0)

@register_op(op_type="ConstantOfShape")
def constant_of_shape(inputs, kwargs):
    value = get_param(kwargs, 'value')
    if isinstance(inputs[0],tuple):
        return value * inputs[0][0]
    if (isinstance(inputs[0],np.ndarray) or isinstance(inputs[0], list)) and len(inputs[0]) == 0:
        return value
    if value[0]== 0.0:
        return torch.zeros(inputs[0].tolist(),dtype=torch.int64)
    elif value[0]== 1.0:
        return torch.ones(inputs[0].tolist(),dtype=torch.int64)
    return value * inputs[0]

@register_op(op_type= "Equal")
def node_equal(inputs, kwargs):
    return inputs[0]==inputs[1]


@register_op(op_type="Squeeze")
def squeeze(node_name ,inputs, kwargs):
    axes = get_param(kwargs, "axes")
    tensor = inputs[0]
    for axis in sorted(axes, reverse=True):
        tensor = tensor.squeeze(axis)
    return tensor

@register_op(op_type="GatherElements")
def gatherElements(inputs, kwargs):
    axes = get_param(kwargs, "axis")
    return inputs[0].gather(dim = axes,index = torch.from_numpy(inputs[1].copy()))

@register_op(op_type="ReduceSum")
def reduceSum(inputs, kwargs):
    axes = kwargs.get("axes",None)
    keepdim = get_param(kwargs,"keepdims")
    if isinstance(inputs[0],torch.Tensor):
        if axes is None:
            return inputs[0].sum()
        else:
            return inputs[0].sum(dim = axes ,keepdim = bool(keepdim))
    else:
        if axes is None:
            return inputs[0].sum(keepdims = bool(keepdim))
        else:
            return inputs[0].sum(axis = axes,keepdims = bool(keepdim))


@register_op(op_type="MatMul")
def matmul(inputs, kwargs):
    return torch.matmul(inputs[0],inputs[1])

@register_op(op_type="Sigmoid")
def sigmoid(inputs, kwargs):
    return torch.sigmoid(inputs[0])

@register_op(op_type="HardSigmoid")
def hardsigmoid(inputs, kwargs):
    return F.hardsigmoid(inputs[0])

@register_op(op_type="Tanh")
def tanh(inputs, kwargs):
    return torch.tanh(inputs[0])

@register_op(op_type="Range")
def range(inputs, kwargs):
    return torch.arange(inputs[0].item(),inputs[1].item(),inputs[2].item())

@register_op(op_type="Where")
def where(inputs, kwargs):
    if isinstance(inputs[0], torch.Tensor):
        return torch.where(inputs[0],inputs[1],inputs[2])
    return np.where(inputs[0],inputs[1],inputs[2])

@register_op(op_type="Expand")
def expand(inputs, kwargs):
    if (inputs[1].int()==1).all():
        return inputs[0]
    if isinstance(inputs[0], np.ndarray):
        return inputs[0] * np.ones(inputs[1],inputs[0].dtype)
    else:
        return inputs[0].expand(inputs[1].int().tolist())
        # shape1 = list(inputs[0].shape)
        # shape2 = inputs[1]
        # assert len(shape1) == len(shape2)
        # shape = [1]*len(shape1)
        # for i in range(len(shape)):
        #     if shape1[i] == 1:
        #         shape[i] = shape2[i]
        #     elif shape2[i] == 1:
        #         shape[i] = shape1[i]
        #     elif shape1[i] == shape2[i]:
        #         shape[i] = shape1[i]
        #     else:
        #         raise AttributeError
        # return inputs[0].expand(list(shape))

@register_op(op_type="Neg")
def neg(inputs, kwargs):
    return -1 * inputs[0]

@register_op(op_type="Softmax")
def softmax(inputs, kwargs):
    if "platform_quant" in kwargs:
        import linger
        return linger.SoftMaxInt.run_onnx_softmax(inputs, kwargs)
    axis = get_param(kwargs, "axis")
    return torch.softmax(inputs[0],axis)

@register_op(op_type="TopK")
def topk(inputs, kwargs):
    axis = get_param(kwargs, "axis")
    largest = bool(get_param(kwargs, "largest"))
    sorted = bool(kwargs.get("sorted",1))
    if isinstance(inputs[0], torch.Tensor):
        return inputs[0].topk(int(inputs[1]),dim = axis, largest = largest, sorted = sorted)
    elif isinstance(inputs[0], np.ndarray):
        return torch.from_numpy(inputs[0]).topk(int(inputs[1]),dim = axis, largest = largest, sorted = sorted).numpy()
    else:
        return torch.tensor(inputs[0]).topk(int(inputs[1]),dim = axis, largest = largest, sorted = sorted)

@register_op(op_type="ScatterElements")
def scatterElements(inputs, kwargs):
    if not isinstance(inputs[1], torch.Tensor) :
        inputs[1] = torch.tensor(inputs[1])
    if not isinstance(inputs[2], torch.Tensor):
        inputs[2] = torch.tensor(inputs[2])
        
    axis = get_param(kwargs, "axis")
    if isinstance(inputs[0], np.ndarray):
        return torch.from_numpy(inputs[0]).scatter(axis,inputs[1],inputs[2]).numpy()
    elif isinstance(inputs[0],torch.Tensor):
        return inputs[0].scatter(axis,inputs[1],inputs[2]).numpy()
    else:
        return torch.tensor(inputs[0]).scatter(axis,inputs[1],inputs[2]).numpy()
    

@register_op(op_type="Exp")
def exp(inputs, kwargs):
    if isinstance(inputs[0],torch.Tensor):
        return inputs[0].exp()
    else:
        return np.exp(inputs[0])


@register_op(op_type="Conv1d")
def conv1d(inputs, kwargs):
    input_length = len(inputs)
    assert input_length == 2 or input_length == 3,"Conv1d ops：the number of inputs is wrong, \
        expect 2 or 3, but {}, the length of List must be 2([input,weight]) or 3([input,weight,bias]) ".format(input_length)
    if input_length == 2:
        input,weights= inputs
        bias = None
    else:
        input,weights,bias = inputs
    dilations = tuple(get_param(kwargs,'dilations'))
    groups = int(get_param(kwargs,'group'))
    pads = get_param(kwargs,"pads")[0]
    strides = tuple(get_param(kwargs,"strides"))
    
    return F.conv1d(input,weights,bias,stride=strides,padding=pads,dilation=dilations,groups=groups)

@register_op(op_type='ReduceProd')
def reduceProd(inputs, kwargs):
    keepdims = bool(get_param(kwargs, "keepdims"))
    axes = kwargs.get('axes',None)

    if isinstance(inputs, torch.Tensor):
        if axes is None:
            return inputs[0].prod()
        else:
            return inputs[0].prod(dim = axes, keepdim=keepdims)
    else:
        inputs[0] = np.asarray(inputs[0])
        return inputs[0].prod(axis = axes,keepdims = keepdims)

@register_op(op_type="ChannelShuffle")
def channel_shuffle(inputs,kwargs):
    n,c,h,w = inputs[0].shape
    groups = kwargs.get("groups")
    return inputs[0].reshape(n,groups,c//groups, h,w).permute(0,2,1,3,4).reshape(n,c,h,w)

@register_op(op_type="ReduceL2")
def reduceL2(inputs,kwargs):
    axis = kwargs.get("axes")
    keepdim = bool(kwargs.get('keepdims'))
    return torch.norm(inputs[0],dim = axis,keepdim=keepdim)

@register_op(op_type="ArgMax")
def argmax(inputs,kwargs):
    axis = kwargs.get("axis")
    keepdim = bool(kwargs.get('keepdims'))
    return torch.argmax(inputs[0],dim = axis,keepdim=keepdim)