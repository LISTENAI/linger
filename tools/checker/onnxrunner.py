import torch
from torch import Tensor
import onnx
from onnx import helper, numpy_helper, TensorProto
import torch.nn.functional as F
from onnx import numpy_helper
import numpy as np
from collections import deque
from .iq_ops_mapper import *
from .float_ops_mapper import *
from linger.config import QuantConfig
from linger.utils import PlatForm, quant
import os
import re
from .utils import single_node_run, if_node_run, parse_attribute_and_name, onnx_topologically_sort
from .utils import get_attribute_value
import traceback
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import math
# from typing import Literal

# DUMP_FORMAT = Literal['float', 'quantized', 'all']
DUMP_FORMAT = {'float', 'quantized', 'all'}
TRANSPARENT_OPS = {'Reshape', 'Transpose', 'Gather', 'Squueze', 'Unsqueeze', 'Slice', 'Split', 'MaxPool',\
                    'Relu', 'Clip', 'Prelu', 'Resize'}

class OnnxRunner:
    def __init__(self, path, dump = False, dump_format = 'quantized') -> None:
        super().__init__()
        assert dump_format in DUMP_FORMAT, f'args dump_format {dump_format} is invalid'
        self._dump = dump
        self._dump_fmt = dump_format
        self.__quant_op_configs = {}
        self._tensor_shapes = {}
        self._model = onnx.load(path)
        self._init_quant_op_configs()
        self._restore_quantize_nodes()
        self._load_onnx()
        self._init_dump()

    def _init_dump(self):
        if self._dump:
            if self._dump is True:
                self.__dump_dir = "data"
                self.__int_dump_path = "data/onnxrunner_int"
                self.__float_dump_path = 'data/onnxrunner_float'
            if os.path.exists(self.__dump_dir):
                os.system("rm -rf {}".format(self.__dump_dir))
            Path(self.__int_dump_path).mkdir(parents=True)
            Path(self.__float_dump_path).mkdir(parents=True)
    
    def _load_onnx(self):
        self.__input_map_dict = dict()  # input or ops output used times
        self.__tensor_dict = dict()  # storage the intermediate tensor to calculate 
        ## load initializer
        for initializer in self._model.graph.initializer:
            self.__tensor_dict[initializer.name] = torch.from_numpy(numpy_helper.to_array(initializer).copy())   # 标记
            self.__input_map_dict[initializer.name] = 0

        self._init_by_platform()

    def _restore_quantize_nodes(self):
        def resolve_input_index(node, locator_logic: dict):
            logic_type = locator_logic.get('type')
            if logic_type == 'static': return locator_logic.get('index')
            if logic_type == 'conditional':
                arg, node_arg_val = locator_logic.get('arg'), len(node.input)
                if arg == 'num_inputs':
                    for case in locator_logic.get('cases', []):
                        if 'if_equal' in case and node_arg_val == case['if_equal']: return case['index']
                        if 'if_greater_equal' in case and node_arg_val >= case['if_greater_equal']: return case['index']
            return None


        model = self._model
        graph = model.graph
        init_names = {init.name for init in graph.initializer}
        graph_inputs = [inp for inp in graph.input if inp.name not in init_names]

        consumer_map: Dict[str, List[Tuple[onnx.NodeProto, int]]] = {i.name: [] for i in graph_inputs}
        for initializer in graph.initializer: consumer_map[initializer.name] = []
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp not in consumer_map: consumer_map[inp] = []
                consumer_map[inp].append((node, i))

        nodes_to_add = []
        connections_to_rewire: Dict[str, Dict[int, str]] = {}
        processed_graph_inputs = set()
        
        print("Starting forward search from graph inputs...")
        for graph_input in graph_inputs:
            if graph_input.name in processed_graph_inputs: continue

            print(f"\nProcessing path starting from input: '{graph_input.name}'")
            queue = deque([(graph_input.name, graph_input.name)])
            visited_tensors = {graph_input.name}
            
            while queue:
                current_tensor, original_source = queue.popleft()
                consumers = consumer_map.get(current_tensor, [])
                path_fixed = False

                for consumer_node, consumer_index in consumers:
                    if consumer_node.op_type in self.__quant_op_configs:
                        print(f"  -> Path reached potential target '{consumer_node.name}' at its input index {consumer_index}.")
                        config = self.__quant_op_configs[consumer_node.op_type]
                        
                        # Dynamically check if the connection is to a quantizable slot
                        matched_quant_input_info = None
                        for quant_input_info in config['quantizable_inputs']:
                            actual_index = resolve_input_index(consumer_node, quant_input_info['locator_logic'])
                            if actual_index == consumer_index:
                                matched_quant_input_info = quant_input_info
                                break
                        
                        if matched_quant_input_info:
                            print(f"  -> SUCCESS: Connection matches the defined quantizable input '{matched_quant_input_info['name']}'.")
                            _, attrs = parse_attribute_and_name(consumer_node)
                            scale_val = attrs.get(matched_quant_input_info['scale_attr'], None)
                            zp_val = attrs.get(matched_quant_input_info['zp_attr'], (0.0))
                            data_bits = attrs.get('data_bits', 8)
                            platform = attrs.get('platform', None)
                            if scale_val is None or zp_val is None:
                                print(f"  -> ERROR: Could not extract quant params. Skipping.")
                                continue

                            quantized_output_name = f"{original_source}_quantized"
                            # import pdb; pdb.set_trace()
                            quant_node = helper.make_node('Quant', inputs=[original_source], outputs=[quantized_output_name],
                                                        name=f"{original_source}_Quant_auto", scale_x=scale_val, zeropoint=zp_val,
                                                        data_bits=data_bits, platform=platform)
                            nodes_to_add.append(quant_node)
                            for i, input in enumerate(graph.input):
                                if input == original_source:
                                    graph.input[i].type.tensor_type.elem_type = TensorProto.FLOAT
                            print(f"  -> ACTION: Scheduled insertion of Quant node for '{original_source}'.")

                            direct_consumers = consumer_map.get(original_source, [])
                            for dc_node, dc_index in direct_consumers:
                                if dc_node.name not in connections_to_rewire: connections_to_rewire[dc_node.name] = {}
                                connections_to_rewire[dc_node.name][dc_index] = quantized_output_name
                                print(f"    - Scheduled to rewire input {dc_index} of '{dc_node.name}'.")

                            processed_graph_inputs.add(original_source)
                            path_fixed = True
                            break # Break from consumers loop, this path is done
                        else:
                            print(f"  -> INFO: Connection is to a non-quantizable input slot of '{consumer_node.name}'. This path is correct.")

                    elif consumer_node.op_type in TRANSPARENT_OPS:
                        for output_tensor in consumer_node.output:
                            if output_tensor not in visited_tensors:
                                print(f"  -> Traversing through transparent op '{consumer_node.name}'...")
                                visited_tensors.add(output_tensor)
                                queue.append((output_tensor, original_source))

                if path_fixed:
                    queue.clear()

        if not nodes_to_add:
            print("\nModel analysis complete. No missing Quant nodes were detected.")

        print("\nApplying graph modifications...")
        graph.node.extend(nodes_to_add)
        for node in graph.node:
            if node.name in connections_to_rewire:
                inputs = list(node.input)
                for index, new_name in connections_to_rewire[node.name].items(): inputs[index] = new_name
                node.ClearField("input")
                node.input.extend(inputs)
                
        onnx_topologically_sort(model)

    def _init_by_platform(self):
        platform = "venus"
        
        for node in self._model.graph.node:
            for attr in node.attribute:
                if attr.name == "platform":
                    platform = attr.s.decode('utf-8')
                    break

        platform_map = {
            "venus": PlatForm.venus,
            "mars": PlatForm.mars,
            "arcs": PlatForm.arcs,
            "jupiter": PlatForm.jupiter,
            "venusA": PlatForm.venusA
        }

        if platform not in platform_map:
            raise ValueError(f"The platform {platform} is not support now")

        QUANT_CONFIGS._update_from_dict({'platform': platform})
    
    def _node_run(self, node ,inputs):
        #Note : The If node processing here, only processes the If node generated by Squeeze
        try:
            if node.op_type !="If":
                return single_node_run(node, inputs)
            else:
                return if_node_run(node, inputs, self.__tensor_dict)
        except Exception as e:  # NotImmplementedError, KeyError,ValueError
            node_name = node.name 
            # When user export onnx using operator_export_type!=ONNX, the node.name don't exist
            if node.name == "": 
                node_name = node.op_type + "_I_" + "_".join([node_input for node_input in node.input]) \
                    +"_O_"+"_".join([node_output for node_output in node.output])
                
            print("Error occured in {} , error message is {}".format(node_name, e))
            traceback.print_exc()
            exit(-1)

    def _get_input(self,node):
        ops_inputs = []
        for input in node.input:
            if len(input) == 0:  # In the floating point model, sometimes the node input is "" when the input is not input.
                ops_inputs.append(None)
            else:
                ops_inputs.append(self.__tensor_dict[input])
        return ops_inputs
    
    def _dump_output(self, node, ops_outputs):
        quant_mode = get_attribute_value(node, 'quant_mode')
        if quant_mode is None:
            quant_mode = 'floor_add'
        round_mode = StringToQuantMode(quant_mode)
        def _flatten_outputs(x):
            if isinstance(x, (tuple, list)):
                out = []
                for v in x:
                    out.extend(_flatten_outputs(v))
                return out
            return [x]
        if len(node.output) == 1:
            self.__tensor_dict[node.output[0]] = ops_outputs
            self._tensor_shapes[node.output[0]] = tuple(ops_outputs.shape)
            if self._dump:
                if self._dump_fmt == 'float' or self._dump_fmt == 'all':
                    dump_path = self.__float_dump_path+os.sep +node.output[0] +"##_float_dump.txt"
                
                    # if ops_outputs.device.type == 'cuda':
                    #     np.savetxt(dump_path, ops_outputs.flatten().cpu().numpy(),fmt="%f")
                    # else:
                    np.savetxt(dump_path, ops_outputs.detach().flatten().numpy(),fmt="%f")
                if self._dump_fmt == 'quantized' or self._dump_fmt == "all":
                    dump_path = self.__int_dump_path+os.sep +node.output[0] +"##_int_dump.txt"
                    if isinstance(ops_outputs, linger.QTensor):
                        scale = ops_outputs.scale
                        bits = ops_outputs.data_bits
                        zp = 0  # TODO: zp = ops_outputs.zero_point
                        # import pdb; pdb.set_trace() 
                        if ops_outputs.dtype == torch.float32 or ops_outputs.dtype == torch.float64:
                            q_output, _ = quant(ops_outputs, bits, scale, zp, round_mode)
                            q_output = q_output.to(torch.int32).detach().flatten().cpu().numpy()
                        else:
                            q_output = ops_outputs.detach().flatten().cpu().numpy()
                        np.savetxt(dump_path, q_output, fmt="%d")
                    else:
                        if ops_outputs.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                            q_output = ops_outputs.to(torch.int32).detach().flatten().cpu().numpy()
                            np.savetxt(dump_path, q_output, fmt="%d")
        else:
            flat_outputs = _flatten_outputs(ops_outputs)
            assert len(flat_outputs) == len(node.output), f"the output number of linger {len(flat_outputs)} is not equal to the output number of onnx {len(node.output)}"
            for output_idx, output in enumerate(node.output):
                self.__tensor_dict[output] = flat_outputs[output_idx]
                self._tensor_shapes[output] = tuple(flat_outputs[output_idx].shape)

                if self._dump and isinstance(flat_outputs[output_idx], torch.Tensor):
                    if self._dump_fmt == 'float' or self._dump_fmt == 'all':
                        dump_path = self.__float_dump_path + os.sep + node.output[output_idx] +"##_float_dump.txt"
                        np.savetxt(dump_path, flat_outputs[output_idx].detach().flatten().cpu().numpy(), fmt="%f")
                    if self._dump_fmt == 'quantized' or self._dump_fmt == "all":
                        dump_path = self.__int_dump_path + os.sep +node.output[output_idx] +"##_int_dump.txt"
                        if isinstance(flat_outputs[output_idx], linger.QTensor):
                            scale = flat_outputs[output_idx].scale
                            bits = flat_outputs[output_idx].data_bits
                            zp = 0 # TODO: zp = flat_outputs[output_idx]
                            if flat_outputs[output_idx].dtype == torch.float32 or flat_outputs[output_idx].dtype == torch.float64:
                                q_output, _ = quant(flat_outputs[output_idx], bits, scale, zp, round_mode)
                                q_output = q_output.to(torch.int).detach().flatten().cpu().numpy()
                            else:
                                q_output = flat_outputs[output_idx].detach().flatten().cpu().numpy()
                            np.savetxt(dump_path, q_output, fmt="%d")
                        else:
                            if flat_outputs[output_idx].dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                                q_output = flat_outputs[output_idx].to(torch.int32).detach().flatten().cpu().numpy()
                                np.savetxt(dump_path, q_output, fmt="%d")

    def _tensor_dict_to_list(self,data):
        ret = []
        for _,input in enumerate(self._model.graph.input):
            if input.name in data:
                ret.append(data[input.name])
        return ret

    def _traverse_input(self,data):
        # LSTMInt will combine hidden_state and cell_state into a tuple input, 
        # and the LSTMInt onnx operator needs to be input separately, so the input does not match, 
        # it needs to be processed separately, and the input hidden_state and cell_state are separated
        if type(data) !=list and type(data)!= tuple and type(data) !=Tensor and type(data)!= dict:
            raise TypeError("Input type ({}) error,must be [list,tuple,tensor,dict]!!".format(type(data)))
        if type(data) !=list and type(data)!=tuple and type(data)!=dict:
            data = [data]
        if type(data) == dict:
            data = self._tensor_dict_to_list(data)

        def _get_list_inout(torch_input, onnx_input):
            if isinstance(torch_input, tuple) or isinstance(torch_input, list):
                for ele in torch_input:
                    onnx_input = _get_list_inout(ele,onnx_input)
            else:
                onnx_input.append(torch_input)
                return onnx_input

        onnx_input = []
        torch_input = list(data)
        _get_list_inout(torch_input,onnx_input)

        onnx_input = tuple([inp if inp.device == torch.device('cpu') else inp.cpu() for inp in onnx_input])
        return onnx_input

    def get_tensor_info(self):
        return self._tensor_shapes

    def run(self, data, special_key = 'None', out_type = "list"):
        data = self._traverse_input(data)
        # In lower pytorch version , when set 'operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK' in export onnx stage,
        # the initializer is considered part of the inputs . Therefore, it needs to be dealt with separately.
        for idx,input in enumerate(self._model.graph.input):
            if idx >= len(data) :
                break
            self.__tensor_dict[input.name] = data[idx]
        
        if out_type == "dict":
            model_output = {}
        else:
            model_output = []
        for node in self._model.graph.node:   
            ops_inputs = self._get_input(node)
            ops_outputs = self._node_run(node, ops_inputs)

            # get output tensor and put it in __tensor_dict
            self._dump_output(node, ops_outputs)
            if node.output[0] == special_key and out_type == "dict":
                model_output[special_key] =  ops_outputs
        
        for output in self._model.graph.output:
            if out_type == "dict":
                model_output[output.name] = self.__tensor_dict[output.name].detach()
            else:
                model_output.append(self.__tensor_dict[output.name].detach())

        return model_output
    
    def _init_quant_op_configs(self):
        self.__quant_op_configs = {
            'AvgPool2dInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'BmmInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input_x',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'input_x_zero_point'
                    },
                    {
                        'name': 'input_y',
                        'locator_logic': {'type': 'static', 'index': 1},
                        'scale_attr': 'scale_y',
                        'zp_attr': 'input_y_zero_point'
                    }
                ]
            },
            'Conv1dInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'Conv2dInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'ConvTranspose2dInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'GRUInt': {
                'quantizable_inputs': [
                    {'name': 'sequence_input', 
                     'locator_logic': {'type': 'static', 'index': 0}, 
                     'scale_attr': 'scale_x', 
                     'zp_attr': 'x_zero_point'},
                    {'name': 'initial_hidden', 
                     'locator_logic': {'type': 'conditional', 'arg': 'num_inputs', 'cases': [{'if_equal': 7, 'index': 1}, {'if_equal': 8, 'index': 2}]}, 
                     'scale_attr': 'scale_h', 
                     'zp_attr': 'h_zero_point'},
                ]
            },
            'iqAdd': {
                'quantizable_inputs': [
                    {
                        'name': 'input_x',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'input_x_zero_point'
                    },
                    {
                        'name': 'input_y',
                        'locator_logic': {'type': 'static', 'index': 1},
                        'scale_attr': 'scale_y',
                        'zp_attr': 'input_y_zero_point'
                    }
                ]
            },
            'iqCat': {
                'quantizable_inputs': [
                    {
                        'name': 'input_0',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x_0',
                        'zp_attr': 'input_zero_point_0'
                    },
                    {
                        'name': 'input_1',
                        'locator_logic': {'type': 'static', 'index': 1},
                        'scale_attr': 'scale_x_1',
                        'zp_attr': 'input_zero_point_1'
                    },
                    {
                        'name': 'input_2',
                        'locator_logic': {'type': 'static', 'index': 2},
                        'scale_attr': 'scale_x_2',
                        'zp_attr': 'input_zero_point_2'
                    },
                    {
                        'name': 'input_3',
                        'locator_logic': {'type': 'static', 'index': 3},
                        'scale_attr': 'scale_x_3',
                        'zp_attr': 'input_zero_point_3'
                    }
                ]
            },
            'iqDiv': {
                'quantizable_inputs': [
                    {
                        'name': 'input_x',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'input_x_zero_point'
                    },
                    {
                        'name': 'input_y',
                        'locator_logic': {'type': 'static', 'index': 1},
                        'scale_attr': 'scale_y',
                        'zp_attr': 'input_y_zero_point'
                    }
                ]
            },
            'iqMul': {
                'quantizable_inputs': [
                    {
                        'name': 'input_x',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'input_x_zero_point'
                    },
                    {
                        'name': 'input_y',
                        'locator_logic': {'type': 'static', 'index': 1},
                        'scale_attr': 'scale_y',
                        'zp_attr': 'input_y_zero_point'
                    }
                ]
            },
            'iqSigmoid': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'iqSum': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'LayerNormInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'LinearInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
            'LSTMInt': {
                'quantizable_inputs': [
                    {'name': 'sequence_input', 
                     'locator_logic': {'type': 'static', 'index': 0}, 
                     'scale_attr': 'scale_i', 
                     'zp_attr': 'i_zero_point'},
                    {'name': 'initial_hidden', 
                     'locator_logic': {'type': 'conditional', 'arg': 'num_inputs', 'cases': [{'if_equal': 7, 'index': 1}, {'if_equal': 8, 'index': 2}]}, 
                     'scale_attr': 'scale_h', 
                     'zp_attr': 'h_zero_point'},
                    {'name': 'initial_cell', 
                     'locator_logic': {'type': 'conditional', 'arg': 'num_inputs', 'cases': [{'if_equal': 7, 'index': 2}, {'if_equal': 8, 'index': 3}]}, 
                     'scale_attr': 'scale_c', 
                     'zp_attr': 'c_zero_point'}
                ]
            },
            'SoftmaxInt': {
                'quantizable_inputs': [
                    {
                        'name': 'input',
                        'locator_logic': {'type': 'static', 'index': 0},
                        'scale_attr': 'scale_x',
                        'zp_attr': 'data_zero_point'
                    }
                ]
            },
        }
        
__all__=["OnnxRunner"]
