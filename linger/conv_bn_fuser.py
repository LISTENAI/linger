import torch
import torch.nn

from .modules import NormalizeConvBN1d, NormalizeConvBN2d
from .ops.ops_names import LINGER_AHEAD_RELU, LINGER_AHEAD_SIGMOID, LINGER_IGNORE_PAMAMTER
from .utils import Singleton, get_device


class FuseableConvBN():
    def __init__(self, conv_f, conv, bn_f, bn, root_model=None):
        self.conv_f = conv_f
        self.conv = conv
        self.bn_f = bn_f
        self.bn = bn
        self.scope_conv = None
        self.scope_bn = None
        self.root_model = None

    def set_root_model(self, root_model):
        self.root_model = root_model


class EmptyBatchNorm(torch.nn.Module):
    r"""融合后的BNmoudule占位符,没有进行任何Tensor操作

    """

    def __init__(self):
        super(EmptyBatchNorm, self).__init__()
        setattr(self, LINGER_IGNORE_PAMAMTER,
                torch.nn.Parameter(torch.zeros([1])))

    def forward(self, input):
        return input

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        pass


def fuse_conv_bn(conv, bn):
    eps = 1e-5
    c_b = getattr(conv, 'bias', None)

    b_mean = bn.running_mean.data
    b_var = bn.running_var.data
    b_w = bn.weight.data
    b_b = bn.bias.data
    sigma = 1/torch.sqrt(b_var+eps)
    alpha = b_w * sigma
    beta = b_b - b_mean * alpha
    conv.weight.data.mul_(alpha.view(-1, *([1]*(len(conv.weight.shape)-1))))
    if c_b is not None:
        conv.bias.data.mul_(alpha).add_(beta)
    else:
        conv.bias = bn.bias
        conv.bias.data.mul_(0).add_(beta)


class SingletonConvFusedBnModules(Singleton):
    fused_conv_module = {}
    fused_bn_module = {}
    _is_close_register = False

    def _close_register(self):
        self._is_close_register = True

    def _register(self, fuseable_conv_bn):
        if self._is_close_register:
            print("warning: module has initlized and linger.init may not work")
        self.fused_conv_module[fuseable_conv_bn.conv] = fuseable_conv_bn
        self.fused_bn_module[fuseable_conv_bn.bn] = fuseable_conv_bn

    def _is_registered_conv(self, conv):
        f_conv = self.fused_conv_module.get(conv)
        return f_conv

    def _is_registered_bn(self, bn):
        f_bn = self.fused_bn_module.get(bn)
        return f_bn

    def build_normalize_convbn2d_scope(self, model):
        queue = [('', '', model)]
        while len(queue) > 0:
            (node_name, scope_name, node) = queue.pop(0)
            find_fused_info = self._is_registered_conv(node)
            if find_fused_info is not None:
                find_fused_info.scope_conv = scope_name
                if find_fused_info.root_model is None:
                    find_fused_info.set_root_model(model)
                    conv_m = find_fused_info.conv
                    bn_m = find_fused_info.bn
                    conv_have_bias = False if conv_m.bias is None else True
                    clamp_conv = None
                    device = get_device(conv_m)
                    ahead_relu = getattr(
                        conv_m, LINGER_AHEAD_RELU, False)
                    if type(conv_m) == torch.nn.Conv2d:
                        clamp_conv = NormalizeConvBN2d(in_channels=conv_m.in_channels,
                                                       out_channels=conv_m.out_channels, kernel_size=conv_m.kernel_size, stride=conv_m.stride,
                                                       padding=conv_m.padding, dilation=conv_m.dilation, groups=conv_m.groups, bias=conv_have_bias, padding_mode=conv_m.padding_mode,
                                                       eps=bn_m.eps, momentum=bn_m.momentum, affine=bn_m.affine, track_running_stats=bn_m.track_running_stats,
                                                       normalize_data=None, normalize_weight=None, normalize_bias=None, ahead_relu=ahead_relu)
                    elif type(conv_m) == torch.nn.Conv1d:
                        clamp_conv = NormalizeConvBN1d(in_channels=conv_m.in_channels,
                                                       out_channels=conv_m.out_channels, kernel_size=conv_m.kernel_size, stride=conv_m.stride,
                                                       padding=conv_m.padding, dilation=conv_m.dilation, groups=conv_m.groups, bias=conv_have_bias, padding_mode=conv_m.padding_mode,
                                                       eps=bn_m.eps, momentum=bn_m.momentum, affine=bn_m.affine, track_running_stats=bn_m.track_running_stats,
                                                       normalize_data=None, normalize_weight=None, normalize_bias=None, ahead_relu=ahead_relu)
                    clamp_conv = clamp_conv.to(device)
                    setattr(find_fused_info.conv_f, node_name, clamp_conv)
                else:
                    assert find_fused_info.root_model == model
            for name, submodule in node.named_children():
                prefix = '' if scope_name == '' else scope_name+'.'
                queue.append((name, prefix+name, submodule))

    def build_empty_bn_scope(self, model):
        queue = [('', '', model)]
        while len(queue) > 0:
            (node_name, scope_name, node) = queue.pop(0)
            find_fused_info = self._is_registered_bn(node)
            if find_fused_info is not None:
                find_fused_info.scope_bn = scope_name
                if find_fused_info.root_model is None:
                    find_fused_info.set_root_model(model)
                else:
                    assert find_fused_info.root_model == model
                setattr(find_fused_info.bn_f, node_name, EmptyBatchNorm())
            for name, submodule in node.named_children():
                prefix = '' if scope_name == '' else scope_name+'.'
                queue.append((name, prefix+name, submodule))

    @staticmethod
    def get_module(model, scope):
        attr_arr = scope.split('.')
        cur_module = model
        for att in attr_arr:
            cur_module = getattr(cur_module, att, None)
        return cur_module

    def fuse_state_dicts(self, state_dict):
        for v in self.fused_conv_module.values():
            assert v.scope_conv is not None
            assert v.scope_bn is not None

            class GeneralModule():
                pass
            keys_bn = []
            atts_bn = {}
            for key_dict in state_dict.keys():
                prefix = v.scope_bn+'.'
                if key_dict.startswith(prefix):
                    attr_name = key_dict[len(prefix):]
                    attr_name = attr_name.split('.', 1)[0]
                    keys_bn.append(key_dict)
                    atts_bn[attr_name] = state_dict[key_dict]
            keys_conv = []
            atts_conv = {}
            for key_dict in state_dict.keys():
                prefix = v.scope_conv+'.'
                if key_dict.startswith(prefix):
                    attr_name = key_dict[len(prefix):]
                    attr_name = attr_name.split('.', 1)[0]
                    atts_conv[attr_name] = state_dict[key_dict]
                    keys_conv.append(key_dict)
            if LINGER_IGNORE_PAMAMTER not in atts_bn.keys():
                for att, att_dict in atts_conv.items():
                    state_dict[v.scope_conv+'.conv.'+att] = att_dict
                for att, att_dict in atts_bn.items():
                    state_dict[v.scope_conv+'.bn.'+att] = att_dict
                for key_bn_pop in keys_bn:
                    if key_bn_pop != LINGER_IGNORE_PAMAMTER:
                        state_dict.pop(key_bn_pop)
                for key_conv_pop in keys_conv:
                    state_dict.pop(key_conv_pop)

    def clear(self):
        if self._is_close_register:
            print("warning: module has initlized and linger.clear may not work")
        self.fused_conv_module.clear()
        self.fused_bn_module.clear()

    def has_fuseable_items(self):
        return len(self.fused_conv_module) > 0


class OpNodeInfo():
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.op = None
        self.scope = None

    def __str__(self):
        s = 'input:'
        for i in self.inputs:
            s += i+' '
        s += '\t output:'
        for o in self.outputs:
            s += o+' '
        s += '\t operator:' + self.op
        s += '\t scope:' + self.scope
        return s

    @staticmethod
    def parse_scope_to_path(scope_str):
        tail_name = scope_str.strip().split('/')[-1].strip()
        assert tail_name != ''
        tail_name = tail_name.replace('__module.', '', 1)
        return tail_name

    def parse_scope(self):
        return self.parse_scope_to_path(self.scope)


def _check_version(version_str, major, subor, minor):
    if '+' in version_str:
        version_str = version_str.split('+')[0]
    version_arr = version_str.split('.')
    maj = int(version_arr[0])
    sub = int(version_arr[1])
    mi = int(version_arr[2])
    if maj >= major and sub >= subor and mi >= minor:
        return True
    return False


def get_op_nodes(graph):
    nodes = []
    for n in graph.nodes():
        op_node = OpNodeInfo()
        for i in n.inputs():
            op_node.inputs.append(i.debugName())
        for o in n.outputs():
            op_node.outputs.append(o.debugName())
        op_node.op = n.kind()
        op_node.scope = n.scopeName()
        nodes.append(op_node)
    return nodes


def find_adjoin_layer(src_node_name, may_be_dst_layers, dict_input, dict_output, src_node_must_be_layers=None):
    find_nodes = []
    for dst_layer in may_be_dst_layers:

        input_tensor = dst_layer.inputs[0]
        src_node = dict_output[input_tensor]
        if src_node != None and src_node.op == src_node_name:
            input_node_set = dict_input[input_tensor]
            if len(input_node_set) == 1 and dst_layer in input_node_set:
                if src_node_must_be_layers is None:
                    find_nodes.append((src_node, dst_layer))
                elif src_node in src_node_must_be_layers:
                    find_nodes.append((src_node, dst_layer))
    target_set = set([])
    src_set = set([])
    for s, d in find_nodes:
        src_set.add(s)
        target_set.add(d)
    return find_nodes, src_set, target_set


def find_adjoin_adjoin_layer(src_node_name, mid_node_name, may_be_dst_layers, dict_input, dict_output):
    find_nodes = []
    for dst_layer in may_be_dst_layers:
        dst_input_tensor = dst_layer.inputs[0]
        mid_node = dict_output[dst_input_tensor]
        if mid_node != None and mid_node.op == mid_node_name:
            mid_input_node_set = dict_input[dst_input_tensor]
            if len(mid_input_node_set) == 1 and dst_layer in mid_input_node_set:
                mid_input_tensor = mid_node.inputs[0]
                src_node = dict_output[mid_input_tensor]
                if src_node != None and src_node.op == src_node_name:
                    src_input_node_set = dict_input[mid_input_tensor]
                    if len(src_input_node_set) == 1 and mid_node in src_input_node_set:
                        find_nodes.append((src_node, mid_node, dst_layer))
    src_set = set([])
    mid_set = set([])
    dst_set = set([])
    for (s, m, d) in find_nodes:
        src_set.add(s)
        mid_set.add(m)
        dst_set.add(d)
    return find_nodes, src_set, mid_set, dst_set


def filter_layers(node_arr, op_name):
    list_node = []
    for n in node_arr:
        if n.op == op_name:
            list_node.append(n)
    return list_node


def parse_fuseable_conv_bn(node_arr, fused_bn=True, ahead_conv_relu=True, ahead_bn_relu=True, ahead_linear_relu=True, ahead_conv_sigmoid=True, ahead_linear_sigmoid=True):
    dict_output = {}
    for n in node_arr:
        for o in n.outputs:
            dict_output[o] = n
    dict_input = {}
    for n in node_arr:
        for i in n.inputs:
            if dict_input.get(i) == None:
                dict_input[i] = set([n])
            else:
                dict_input[i].add(n)
    list_bn = filter_layers(node_arr, 'aten::batch_norm')
    list_relu = filter_layers(node_arr, 'aten::relu')
    list_sigmoid = filter_layers(node_arr, 'aten::sigmoid')
    fused_conv_sigmoid = []
    fused_linear_sigmoid = []
    fused_linear_bias_sigmoid = []
    fused_conv_bn = []
    fused_conv_bn_relu = []
    fused_conv_relu = []
    fused_bn_relu = []
    fused_linear_relu = []
    fused_linear_bias_relu = []
    if fused_bn:
        fused_conv_bn, _, _ = find_adjoin_layer(
            'aten::_convolution', list_bn, dict_input, dict_output)
        if ahead_bn_relu:
            fused_conv_bn_relu, _, _, _ = find_adjoin_adjoin_layer(
                'aten::_convolution', 'aten::batch_norm', list_relu, dict_input, dict_output)
    if ahead_conv_relu:
        fused_conv_relu, _, _ = find_adjoin_layer(
            'aten::_convolution', list_relu, dict_input, dict_output)
    if ahead_conv_sigmoid:
        fused_conv_sigmoid, _, _ = find_adjoin_layer(
            'aten::_convolution', list_sigmoid, dict_input, dict_output)
    if ahead_linear_sigmoid:
        fused_linear_bias_sigmoid, _, _, _ = find_adjoin_adjoin_layer(
            'aten::matmul', 'aten::add_', list_sigmoid, dict_input, dict_output)
        fused_linear_sigmoid, _, _ = find_adjoin_layer(
            'aten::matmul', list_sigmoid, dict_input, dict_output)
    if ahead_bn_relu:
        fused_bn_relu, _, _ = find_adjoin_layer(
            'aten::batch_norm', list_relu, dict_input, dict_output)
    if ahead_linear_relu:
        fused_linear_bias_relu, _, _, _ = find_adjoin_adjoin_layer(
            'aten::matmul', 'aten::add_', list_relu, dict_input, dict_output)
        fused_linear_relu, _, _ = find_adjoin_layer(
            'aten::matmul', list_relu, dict_input, dict_output)

    return fused_conv_bn, fused_conv_bn_relu, fused_conv_relu, fused_bn_relu, fused_linear_relu, fused_linear_bias_relu, fused_conv_sigmoid, fused_linear_sigmoid, fused_linear_bias_sigmoid


def scope_to_module(root_module, scope):
    tail_name = OpNodeInfo.parse_scope_to_path(scope)
    module_arr_name = tail_name.split('.')
    module_cur = root_module
    module_cur_name = ''
    moduel_cur_father = root_module
    str_find = ''
    for sub_att_name in module_arr_name:
        str_find += sub_att_name+"."
        moduel_cur_father = module_cur
        module_cur = getattr(module_cur, sub_att_name)
        module_cur_name = sub_att_name
        assert module_cur is not None, 'can not find '+str_find
    return (moduel_cur_father, module_cur, module_cur_name)


def FuseConvBNAheadRelu(model, *args, fused_bn=True, ahead_conv_relu=True, ahead_bn_relu=True, ahead_linear_relu=True, ahead_conv_sigmoid=True, ahead_linear_sigmoid=True):
    SingletonConvFusedBnModules().clear()
    assert _check_version(torch.__version__, 1, 5,
                          0), 'error: torch version must greater than 1.5'
    graph = torch.jit.trace(model, *args)
    node_arr = get_op_nodes(graph.inlined_graph)
    fuseable_conv_bn, fuseable_conv_bn_relu, fuseable_conv_relu, fuseable_bn_relu, fuseable_linear_relu, fuseable_linear_bias_relu, fuseable_conv_sigmoid, fuseable_linear_sigmoid, fuseable_linear_bias_sigmoid = parse_fuseable_conv_bn(
        node_arr, fused_bn, ahead_conv_relu, ahead_bn_relu, ahead_linear_relu, ahead_conv_sigmoid, ahead_linear_sigmoid)
    module_paths = []
    if fused_bn:
        if ahead_bn_relu:
            for (conv, bn, _) in fuseable_conv_bn_relu:
                _, conv_module, _ = scope_to_module(model, conv.scope)
                setattr(conv_module, LINGER_AHEAD_RELU, True)
        for (conv, bn) in fuseable_conv_bn:
            conv_module_father, conv_module, conv_module_name = scope_to_module(
                model, conv.scope)
            bn_module_father, bn_module, bn_module_name = scope_to_module(
                model, bn.scope)
            if (type(conv_module) == torch.nn.Conv2d and type(bn_module) == torch.nn.BatchNorm2d) or \
                    (type(conv_module) == torch.nn.Conv1d and type(bn_module) == torch.nn.BatchNorm1d):
                fuseableconv_bn = FuseableConvBN(
                    conv_module_father, conv_module, bn_module_father, bn_module)
                SingletonConvFusedBnModules()._register(fuseableconv_bn)
                module_paths.append((conv.parse_scope(), bn.parse_scope()))
    if ahead_conv_relu:
        for (conv, _) in fuseable_conv_relu:
            _, conv_module, _ = scope_to_module(model, conv.scope)
            setattr(conv_module, LINGER_AHEAD_RELU, True)
    if ahead_bn_relu:
        for (bn, _) in fuseable_bn_relu:
            _, bn_module, _ = scope_to_module(model, bn.scope)
            setattr(bn_module, LINGER_AHEAD_RELU, True)
    if ahead_conv_sigmoid:
        for (conv, _) in fuseable_conv_sigmoid:
            _, conv_module, _ = scope_to_module(model, conv.scope)
            setattr(conv_module, LINGER_AHEAD_SIGMOID, True)
    if ahead_linear_sigmoid:
        for (linear, _) in fuseable_linear_sigmoid:
            _, linear_module, _ = scope_to_module(model, linear.scope)
            setattr(linear_module, LINGER_AHEAD_SIGMOID, True)
        for(linear, add, _) in fuseable_linear_bias_sigmoid:
            _, linear_module, _ = scope_to_module(model, linear.scope)
            setattr(linear_module, LINGER_AHEAD_SIGMOID, True)
    if ahead_linear_relu:
        for(linear, _) in fuseable_linear_relu:
            _, linear_module, _ = scope_to_module(model, linear.scope)
            setattr(linear_module, LINGER_AHEAD_RELU, True)
        for(linear, add, _) in fuseable_linear_bias_relu:
            _, linear_module, _ = scope_to_module(model, linear.scope)
            setattr(linear_module, LINGER_AHEAD_RELU, True)
    return module_paths


def FuseBNIntoConv(model, *args):
    r"""融合BN操作到Conv里

    Args:
        model(torch.nn.Module):模型
        *args:模型的trace位置参数
        **kwargs:模型trace的keyword参数
    Example:
        >>> net1 = shufflenet_v2_x1_0(pretrained=False)
        >>> net1.load_state_dict(torch.load(dict_file))
        >>> aa = net1(input)
        >>> linger.FuseBNIntoConv(net1,dummy_input)
        >>> net2 = linger.init(net1)
        >>> net2.load_state_dict(torch.load(dict_file))
    """
    assert False, 'FuseBNIntoConv is deprecated please use linger.trace_layers(root_net, trace_net, dummy_input)'


__all__ = ['FuseBNIntoConv', 'SingletonConvFusedBnModules',
           'EmptyBatchNorm', 'FuseConvBNAheadRelu']
