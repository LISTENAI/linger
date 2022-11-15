
import linger
import linger.onnx
import torch
import torch.nn as nn
import numpy
import onnx



# class NetRaw_2(nn.Module):
#     def __init__(self):
#         super(NetRaw_2, self).__init__()
#         self.conv0 = nn.Conv2d(1, 100, kernel_size=(3,1), padding=(0,0), groups=1, bias=True)
#         self.bn0 = nn.BatchNorm2d(100)
#         self.relu0 = nn.ReLU()
#         self.conv1 = nn.Conv2d(100, 100, kernel_size=(1,3), padding=(0,1), groups=1, bias=True)
#         self.bn1 = nn.BatchNorm2d(100)
#         self.relu1 = nn.ReLU()
#         self.final_conv = nn.Conv2d(100, 10, 1, 1, 0)
#     def forward(self, input):
#         x = self.conv0(input)
#         x = self.bn0(x)         
#         x = self.relu0(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.final_conv(x)  #(1, 10, 1, 100) (d, b*t)
#         return x



# class NetComplicate_2(nn.Module):
#     def __init__(self):
#         super(NetComplicate_2, self).__init__()
#         self.conv0 = nn.Conv2d(1, 100, kernel_size=(3,1), padding=(0,0), groups=1, bias=True)
#         self.bn0 = nn.BatchNorm2d(100)
#         self.relu0 = nn.ReLU()
#         self.conv1 = nn.Conv2d(100, 100, kernel_size=(1,3), padding=(0,1), groups=1, bias=True)
#         self.bn1 = nn.BatchNorm2d(100)
#         self.relu1 = nn.ReLU()
#         self.final_conv = nn.Conv2d(100, 10, 1, 1, 0)
#         self.net_2 = NetRaw_2()
#     def forward(self, input):
#         x = self.conv0(input)
#         y = self.net_2(input)
#         x = self.bn0(x)         
#         x = self.relu0(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.final_conv(x)  #(1, 10, 1, 100) (d, b*t)
#         return x+y
# def test_scope_info_simple():            
#     torch.manual_seed(2)
#     torch.cuda.manual_seed_all(2)
#     aa = torch.randn(10, 1, 10, 10).cuda()
#     netraw_2 = NetRaw_2().cuda()      
#     with torch.no_grad():
#         netraw_2.eval()
#         linger.onnx.export(netraw_2, (aa), "data.ignore/test_scoped_1.onnx",export_params=True,opset_version=12, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
#     onnx_model = onnx.load("data.ignore/test_scoped_1.onnx")
#     for node in onnx_model.graph.node:
#        node_name =node.name
#       # print(node_name)
#        assert node_name.startswith('.')
#        assert len(node_name.split('.')) >1
#        assert len(node_name.split('/')) ==2

# def test_scope_info_complicate():            
#     torch.manual_seed(2)
#     torch.cuda.manual_seed_all(2)
#     aa = torch.randn(10, 1, 10, 10).cuda()
#     netcomplicate_2 = NetComplicate_2().cuda()      
#     with torch.no_grad():
#         netcomplicate_2.eval()
#         linger.onnx.export(netcomplicate_2, (aa), "data.ignore/test_scoped_2.onnx",export_params=True,opset_version=12, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
#     onnx_model = onnx.load("data.ignore/test_scoped_2.onnx")
#     for node in onnx_model.graph.node:
#        node_name =node.name
#        #print(node_name)
#        assert node_name.startswith('.')
#        assert len(node_name.split('.')) >1
#        if node.op_type !="Add":
#            assert len(node_name.split('/')) ==2
#        assert node.op_type != 'ScopedEnter'
#        assert node.op_type != 'ScopedLeave'

