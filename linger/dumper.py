import os
import re

import numpy as np
import prettytable as pt
import torch
import torch.nn as nn

from .ops import *
from .tools.weight_bias_analyse import clamp_with_dynamic

tb_all = pt.PrettyTable()

tb_all.field_names = ["Layer_name", "Mean", "Max",
                      "Multiple(Max/Mean)", "Dynamic 0.99", "Versu(Max/Dynamic)"]


def _hook_forward_anylse(module, input, output):
    if 1:  # not module.training:
        assert hasattr(module, LINGER_DUMP_NAME)
        file_path = getattr(module, LINGER_DUMP_NAME)
        if type(output) is tuple:
            if isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
                if isinstance(output[0], torch.nn.utils.rnn.PackedSequence):
                    dump_out = output[0][0]
                else:
                    dump_out = output[0]
            else:
                if type(output[1]) is tuple:
                    dump_out = torch.cat(
                        (output[0], output[1][0], output[1][1]))
                else:
                    dump_out = torch.cat(output)
        else:
            dump_out = output

        file_path = file_path[5:]  # remove "root."

        clamp_with_dynamic(dump_out.detach(), dynamic_percent=0.9,
                    layer_name=file_path, tb_all=tb_all)


def _hook_forward(module, input, output):
    if not module.training:
        assert hasattr(module, LINGER_DUMP_NAME)
        file_path = getattr(module, LINGER_DUMP_NAME)
        if type(input) is tuple:
            if len(input) == 1:
                dump_in = input[0]  # 单一输入 在此处也是tuple类型
            else:
                dump_in = input[0]  # 多个输入  只保存第一个输入
        else:
            dump_in = input
        if type(output) is tuple:
            if isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
                if isinstance(output[0], torch.nn.utils.rnn.PackedSequence):
                    dump_out = output[0][0]
                else:
                    dump_out = output[0]
            else:
                if type(output[1]) is tuple:
                    dump_out = torch.cat(
                        (output[0], output[1][0], output[1][1]))
                else:
                    dump_out = torch.cat(output)
        else:
            dump_out = output
        np.savetxt(file_path+'_input_float',
                   dump_in.detach().reshape(-1).cpu().numpy(), fmt='%f')
        np.savetxt(file_path+'_output_float',
                   dump_out.detach().reshape(-1).cpu().numpy(), fmt='%f')


def _dfs_submodules(model):
    dfs_modules = []
    stack = [('root', model)]
    while len(stack) > 0:
        (name_m, m) = stack.pop()
        children_num = 0
        for name, submodule in m.named_children():
            stack.append((name_m+'/'+name, submodule))
            children_num += 1
        if children_num == 0:
            dfs_modules.append((name_m, m))
    dfs_modules.reverse()
    return dfs_modules


class Dumper():
    def __init__(self):
        self.module_dump_quanted = []
        self.module_hooks_dump_all = []

    def __enter__(self):
        self.module_dump_quanted = []
        self.module_hooks_dump_all = []
        return self

    def __exit__(self, type, value, trace):
        self._clear_dump_quanted()
        self._clear_dump_model()

    def enable_dump_quanted(self, model: nn.Module, path: str = "./dump", match_pattern: str = ".*"):
        if model.training:
            print("error model must be eval when dump,call model.eval() before dump")
            exit(-1)
        if not os.path.exists(path):
            os.makedirs(path)
        queue = [("root", model)]
        while len(queue) > 0:
            name_m, node = queue.pop(0)
            for name, submodule in node.named_children():
                prefix = name_m + '.' + name
                queue.append((prefix, submodule))
                if isinstance(submodule, SupportQuantedIntModules) and re.match(match_pattern, prefix) is not None:
                    submodule.prefix = prefix
                    submodule.dump = True
                    submodule.path = path
                    self.module_dump_quanted.append(submodule)

    def _clear_dump_quanted(self):
        for op in self.module_dump_quanted:
            op.prefix = ''
            op.dump = False
            op.path = None
        self.module_dump_quanted = []

    def enable_dump_model(self, model: nn.Module, path: str ="./dump", match_pattern: str =".*", hook_forward=_hook_forward):
        if model.training:
            print("error model must be eval when dump,call model.eval() before dump")
            exit(-1)
        if not os.path.exists(path):
            os.makedirs(path)
        leaf_all_modules = _dfs_submodules(model)
        for name, leaf_module in leaf_all_modules:
            dump_name = name.replace('/', '.')
            if re.match(match_pattern, dump_name) is not None:
                setattr(leaf_module, LINGER_DUMP_NAME,
                        os.path.join(path, dump_name))
                hook_handle = leaf_module.register_forward_hook(_hook_forward)
                self.module_hooks_dump_all.append((leaf_module, hook_handle))

    def _clear_dump_model(self):
        for (m, hook) in self.module_hooks_dump_all:
            hook.remove()
            delattr(m, LINGER_DUMP_NAME)
        self.module_hooks_dump_all = []

    def analyse_layer_output(self, model: nn.Module, match_pattern: str =".*"):
        leaf_all_modules = _dfs_submodules(model)

        for name, leaf_module in leaf_all_modules:
            dump_name = name.replace('/', '.')
            if re.match(match_pattern, dump_name) is not None:
                setattr(leaf_module, LINGER_DUMP_NAME, dump_name)
                hook_handle = leaf_module.register_forward_hook(
                    _hook_forward_anylse)
                self.module_hooks_dump_all.append((leaf_module, hook_handle))

    def save_out_analyse_log(self, save_log_path: str ="Analyse_layer_output.log"):
        out_flie = open(save_log_path, 'w')
        out_flie.write(str(tb_all))
        out_flie.close()
