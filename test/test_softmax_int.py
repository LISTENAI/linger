import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import symbolic_trace, GraphModule
import operator

# ======================
# 定义 QAdd 模块
# ======================
class QAdd(nn.Module):
    def __init__(self):
        super(QAdd, self).__init__()
    def forward(self, x, y):
        # 这里可以放量化逻辑
        return x * y


# ======================
# 原始模型：含 add 操作
# ======================
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out = out1 + out2   # ⚠️ 这里是 add

        b1 = out.unsqueeze(1)         # (B, 1, 10)
        b2 = out.unsqueeze(2)         # (B, 10, 1)
        out = torch.bmm(b1, b2) 
        
        out = self.act(out)
        return out


# ======================
# FX Graph Transform
# ======================
def replace_add_with_qadd(gm: GraphModule) -> GraphModule:
    graph = gm.graph
    for node in list(graph.nodes):
        # 查找 add 节点（可能是 operator.add 或 Tensor.__add__）
        if node.op == "call_function" and node.target in (operator.add, torch.add, torch.ops.aten.add.Tensor):
            with graph.inserting_after(node):
                # 插入 QAdd 模块
                qadd_mod = QAdd()
                qadd_name = f"qadd_{node.name}"
                gm.add_module(qadd_name, qadd_mod)

                new_node = graph.call_module(qadd_name, args=node.args)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)
    graph.lint()
    gm.recompile()
    return gm


# ======================
# 测试
# ======================
if __name__ == "__main__":
    model = MyModel()
    traced = symbolic_trace(model)
    print("原始Graph:")
    print(traced.graph)

    x = torch.randn(4, 10)
    y = model(x)
    print("原始模型:", x, y)

    # 替换 add -> QAdd
    new_gm = replace_add_with_qadd(traced)
    print("\n替换后的Graph:")
    print(new_gm.graph)

    # 验证运行
    # x = torch.randn(4, 10)
    y = new_gm(x)
    print("\n模型输出:", x, y)
