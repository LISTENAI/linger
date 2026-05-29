import torch
import torch.nn
import torch.onnx
import numpy as np
import onnx
import os
import inspect
from onnx import numpy_helper
from onnx import helper
from io import BytesIO
from pathlib import Path
from typing import Mapping

from ..config import QUANT_CONFIGS
from ..utils import QuantMode, _single, _pair, _triple

from .graph_optimizer import optimize_onnx_graph

torch_onnx_export = torch.onnx.export

QDOMAIN_NAME = 'linger'
FLOAT_ELEM_TYPE = onnx.TensorProto.FLOAT
INT8_ELEM_TYPE = onnx.TensorProto.INT8
INT16_ELEM_TYPE = onnx.TensorProto.INT16
INT32_ELEM_TYPE = onnx.TensorProto.INT32
INT64_ELEM_TYPE = onnx.TensorProto.INT64
BOOL_ELEM_TYPE = onnx.TensorProto.BOOL
LAYERNORM_NORMAL_SCALE = 2 ** 15
STANDARD_QUANT_PASSTHROUGH_OPS = {
    "Identity",
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Flatten",
    "Slice",
    "Pad",
    "Gather",
    "Expand",
    "Tile",
    "Relu",
    "Clip",
}
STANDARD_QUANT_INPUT_OPS = STANDARD_QUANT_PASSTHROUGH_OPS | {
    "Split",
    "MaxPool",
    "Shape",
    "Cast",
}
CUSTOM_QUANT_PASSTHROUGH_OPS = {
    "Squeeze",
}

def _quant_mode_from_name(mode_name):
    if isinstance(mode_name, bytes):
        mode_name = mode_name.decode("utf-8")
    if not isinstance(mode_name, str):
        return QuantMode.round
    if not hasattr(QuantMode, mode_name):
        return QuantMode.round
    return getattr(QuantMode, mode_name)


def _round_array(data, quant_mode):
    if quant_mode == QuantMode.floor_add:
        return np.floor(data + 0.5)
    if quant_mode == QuantMode.floor:
        return np.floor(data)
    if quant_mode == QuantMode.ceil:
        return np.ceil(data)
    return np.round(data)


def _int_dtype_for_bits(data_bits):
    if data_bits <= 8:
        return np.int8
    if data_bits <= 16:
        return np.int16
    if data_bits <= 32:
        return np.int32
    return np.int64


def _elem_type_for_bits(data_bits):
    if data_bits <= 8:
        return INT8_ELEM_TYPE
    if data_bits <= 16:
        return INT16_ELEM_TYPE
    if data_bits <= 32:
        return INT32_ELEM_TYPE
    return INT64_ELEM_TYPE


def quant_weight_bias(f_data, data_bits, scale, quant_mode=QuantMode.round):
    min_val = -(2**(data_bits - 1))
    max_val = (2**(data_bits - 1)) - 1

    x_scaled = _round_array(np.asarray(f_data) * scale, quant_mode)
    x_clipped = np.clip(x_scaled, min_val, max_val)
    q_val = x_clipped.astype(_int_dtype_for_bits(data_bits))
    return q_val


def _get_attr_value(node, *names):
    for attr in node.attribute:
        if attr.name in names:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            if attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            if attr.type == onnx.AttributeProto.STRING:
                return attr.s
    return None


def _get_layernorm_normal_scale(node):
    scale = _get_attr_value(node, "scale_y_normal", "scale_y_normal_f", "scale_y_norm", "scale_y_norm_f")
    return LAYERNORM_NORMAL_SCALE if scale is None else scale


def _quantize_initializer_like(graph, initializer_map, node, input_index, quantized_array, cache):
    input_name = node.input[input_index]
    cache_key = (input_name, str(quantized_array.dtype), quantized_array.tobytes())

    if cache_key in cache:
        node.input[input_index] = cache[cache_key]
        return

    current_initializer = initializer_map[input_name]
    consumers = 0
    for other_node in graph.node:
        for other_input in other_node.input:
            if other_input == input_name:
                consumers += 1

    if consumers <= 1:
        graph.initializer.remove(current_initializer)
        new_initializer = numpy_helper.from_array(quantized_array, input_name)
        graph.initializer.append(new_initializer)
        initializer_map[input_name] = new_initializer
        cache[cache_key] = input_name
        return

    new_name = f"{input_name}.quant_{input_index}"
    suffix = 0
    while new_name in initializer_map:
        suffix += 1
        new_name = f"{input_name}.quant_{input_index}_{suffix}"
    new_initializer = numpy_helper.from_array(quantized_array, new_name)
    graph.initializer.append(new_initializer)
    initializer_map[new_name] = new_initializer
    node.input[input_index] = new_name
    cache[cache_key] = new_name


def _get_direct_quant_spec_for_input(node, input_index):
    quant_mode = _quant_mode_from_name(_get_attr_value(node, "quant_mode", "quant_mode_s"))

    if 'QGRU' in node.op_type or 'QLSTM' in node.op_type:
        has_hidden = bool(_get_attr_value(node, "has_hidden", "has_hidden_i") or 0)
        has_cell = bool(_get_attr_value(node, "has_cell", "has_cell_i") or 0)
        if 'QLSTM' in node.op_type:
            state_input_count = 2 if has_hidden and has_cell else 0
        else:
            state_input_count = 1 if has_hidden else 0
        weight_ih_index = 1 + state_input_count
        weight_hh_index = weight_ih_index + 1
        bias_ih_index = weight_hh_index + 1
        bias_hh_index = bias_ih_index + 1

        scale_i = _get_attr_value(node, 'scale_x', 'scale_x_f')
        scale_h = _get_attr_value(node, 'scale_h', 'scale_h_f')
        scale_c = _get_attr_value(node, 'scale_c', 'scale_c_f')
        scale_iw = _get_attr_value(node, 'scale_iw', 'scale_iw_f')
        scale_hw = _get_attr_value(node, 'scale_hw', 'scale_hw_f')
        x_bits = _get_attr_value(node, 'x_bits', 'x_bits_i')
        x_bits = 8 if x_bits is None else int(x_bits)
        w_bits = _get_attr_value(node, 'w_bits', 'w_bits_i')
        w_bits = 8 if w_bits is None else int(w_bits)
        if input_index == 0 and scale_i is not None:
            return {"scale": scale_i, "bits": x_bits, "quant_mode": quant_mode}
        if 'QLSTM' in node.op_type and has_hidden and input_index == 1 and scale_h is not None:
            return {"scale": scale_h, "bits": x_bits, "quant_mode": QuantMode.round}
        if 'QLSTM' in node.op_type and has_hidden and has_cell and input_index == 2 and scale_c is not None:
            return {"scale": scale_c, "bits": 32, "quant_mode": QuantMode.round}
        if input_index == weight_ih_index and scale_iw is not None:
            return {"scale": scale_iw, "bits": w_bits, "quant_mode": QuantMode.round}
        if input_index == weight_hh_index and scale_hw is not None:
            return {"scale": scale_hw, "bits": w_bits, "quant_mode": QuantMode.round}
        if input_index == bias_ih_index and scale_i is not None and scale_iw is not None:
            return {"scale": scale_i * scale_iw, "bits": 32, "quant_mode": QuantMode.round}
        if input_index == bias_hh_index and scale_h is not None and scale_hw is not None:
            return {"scale": scale_h * scale_hw, "bits": 32, "quant_mode": QuantMode.round}
        return None

    scale_w = _get_attr_value(node, 'scale_w', 'scale_w_f')
    scale_x = _get_attr_value(node, 'scale_x', 'scale_x_f')
    w_bits = _get_attr_value(node, 'w_bits', 'w_bits_i')
    w_bits = 8 if w_bits is None else int(w_bits)

    if node.op_type in {"Gather", "QEmbedding"}:
        if input_index == 0 and scale_w is not None:
            return {"scale": scale_w, "bits": w_bits, "quant_mode": QuantMode.round}
        return None

    if node.op_type in {"QLayerNorm", "LayerNormInt"}:
        if input_index == 1 and scale_w is not None:
            return {"scale": scale_w, "bits": w_bits, "quant_mode": QuantMode.round}
        if input_index == 2 and scale_w is not None:
            return {
                "scale": _get_layernorm_normal_scale(node) * scale_w,
                "bits": 32,
                "quant_mode": QuantMode.round,
            }

    prefixes = []
    if input_index == 0:
        prefixes = ["x", "x_0"]
    elif input_index == 1:
        prefixes = ["y", "x_1"]
    else:
        prefixes = [f"x_{input_index}"]

    for prefix in prefixes:
        scale = _get_attr_value(node, f"scale_{prefix}", f"scale_{prefix}_f")
        bits = _get_attr_value(node, f"{prefix}_bits", f"{prefix}_bits_i")
        if scale is not None and bits is not None:
            return {"scale": scale, "bits": int(bits), "quant_mode": quant_mode}

    if input_index == 1 and scale_w is not None:
        return {"scale": scale_w, "bits": w_bits, "quant_mode": QuantMode.round}
    if input_index == 2 and scale_x is not None and scale_w is not None:
        return {"scale": scale_x * scale_w, "bits": 32, "quant_mode": QuantMode.round}

    return None


def _get_quant_node_input_spec(node, input_index):
    if node.domain != QDOMAIN_NAME or node.op_type != "Quant" or input_index != 0:
        return None

    scale = _get_attr_value(node, "scale_x", "scale_x_f")
    if scale is None:
        return None
    bits = _get_attr_value(node, "data_bits", "data_bits_i", "x_bits", "x_bits_i")
    bits = 8 if bits is None else int(bits)
    quant_mode = _quant_mode_from_name(_get_attr_value(node, "quant_mode", "quant_mode_s"))
    return {"scale": scale, "bits": bits, "quant_mode": quant_mode}


def _get_passthrough_outputs(node, input_index):
    if node.op_type in {"Identity", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Slice", "Pad", "Gather", "Clip"}:
        return list(node.output) if input_index == 0 else []
    if node.op_type in {"Expand", "Tile"}:
        return list(node.output) if input_index == 0 else []
    if node.op_type == "Split":
        return list(node.output) if input_index == 0 else []
    if node.op_type == "MaxPool":
        return [node.output[0]] if input_index == 0 and len(node.output) > 0 else []
    if node.op_type == "Where":
        return list(node.output) if input_index in {1, 2} else []
    if node.op_type == "Concat":
        return list(node.output)
    return []


def _same_quant_spec(lhs, rhs):
    if lhs is None or rhs is None:
        return lhs is rhs
    return (
        np.isclose(float(lhs["scale"]), float(rhs["scale"]), rtol=1e-6, atol=1e-8)
        and int(lhs["bits"]) == int(rhs["bits"])
        and lhs["quant_mode"] == rhs["quant_mode"]
    )


def _find_downstream_quant_spec(graph, start_node, start_input_index, max_hops=16):
    start_outputs = _get_passthrough_outputs(start_node, start_input_index)
    if not start_outputs:
        return None

    consumer_map = {}
    for node in graph.node:
        for input_index, input_name in enumerate(node.input):
            consumer_map.setdefault(input_name, []).append((node, input_index))

    queue = [(output_name, 1) for output_name in start_outputs]
    visited = set()
    found_specs = []

    while queue:
        value_name, depth = queue.pop(0)
        if depth > max_hops or value_name in visited:
            continue
        visited.add(value_name)

        for consumer_node, consumer_input_index in consumer_map.get(value_name, []):
            spec = _get_quant_node_input_spec(consumer_node, consumer_input_index)
            if spec is None:
                spec = _get_direct_quant_spec_for_input(consumer_node, consumer_input_index)
            if spec is not None:
                found_specs.append(spec)
                continue
            for output_name in _get_passthrough_outputs(consumer_node, consumer_input_index):
                queue.append((output_name, depth + 1))

    if not found_specs:
        return None

    base_spec = found_specs[0]
    for spec in found_specs[1:]:
        if not _same_quant_spec(base_spec, spec):
            return None
    return base_spec


def _get_quant_spec_for_input(graph, node, input_index):
    direct_spec = _get_direct_quant_spec_for_input(node, input_index)
    if direct_spec is not None:
        return direct_spec
    return _find_downstream_quant_spec(graph, node, input_index)


def _remove_unused_initializers(graph):
    used_names = set()
    for node in graph.node:
        used_names.update(node.input)
    used_names.update(output.name for output in graph.output)

    for initializer in list(graph.initializer):
        if initializer.name not in used_names:
            graph.initializer.remove(initializer)


def _remove_initializer_inputs(graph):
    initializer_names = {initializer.name for initializer in graph.initializer}
    kept_inputs = [value for value in graph.input if value.name not in initializer_names]
    _clear_repeated_field(graph, "input")
    graph.input.extend(kept_inputs)


def _make_tensor_state(elem_type=FLOAT_ELEM_TYPE, scale=None, bits=None):
    return {"elem_type": elem_type, "scale": scale, "bits": bits}


def _make_float_state():
    return _make_tensor_state(FLOAT_ELEM_TYPE)


def _make_quant_state(scale, bits):
    bits = 8 if bits is None else int(bits)
    return _make_tensor_state(_elem_type_for_bits(bits), float(scale), bits)


def _is_quantized_state(state):
    return state is not None and state.get("scale") is not None and state.get("bits") is not None


def _copy_state(state):
    if state is None:
        return _make_float_state()
    return dict(state)


def _existing_tensor_elem_types(graph):
    elem_types = {}
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        elem_type = value.type.tensor_type.elem_type
        if elem_type != 0:
            elem_types[value.name] = elem_type
    for initializer in graph.initializer:
        elem_types[initializer.name] = initializer.data_type
    return elem_types


def _initial_tensor_states(graph, elem_types):
    states = {}
    for name, elem_type in elem_types.items():
        states[name] = _make_tensor_state(elem_type)
    return states


def _all_value_names(graph):
    names = set()
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        if value.name:
            names.add(value.name)
    for initializer in graph.initializer:
        if initializer.name:
            names.add(initializer.name)
    for node in graph.node:
        for name in list(node.input) + list(node.output):
            if name:
                names.add(name)
    return names


def _unique_value_name(base, used_names):
    base = base or "tensor"
    name = base
    index = 0
    while name in used_names:
        index += 1
        name = f"{base}_{index}"
    used_names.add(name)
    return name


def _constant_output_state(node):
    for attr in node.attribute:
        if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
            return _make_tensor_state(attr.t.data_type)
    return _make_float_state()


def _cast_output_state(node, fallback_state):
    elem_type = fallback_state.get("elem_type", FLOAT_ELEM_TYPE) if fallback_state else FLOAT_ELEM_TYPE
    to_type = _get_attr_value(node, "to", "to_i")
    if to_type is not None:
        elem_type = int(to_type)
    return _make_tensor_state(elem_type)


def _custom_output_states(node, input_states):
    output_count = len(node.output)
    if output_count == 0:
        return []

    if node.op_type == "Dequant":
        return [_make_float_state() for _ in node.output]

    if node.op_type == "Quant":
        scale = _get_attr_value(node, "scale_x", "scale_x_f")
        bits = _get_attr_value(node, "data_bits", "data_bits_i", "x_bits", "x_bits_i")
        bits = 8 if bits is None else int(bits)
        if scale is None:
            return [_make_tensor_state(_elem_type_for_bits(bits)) for _ in node.output]
        return [_make_quant_state(scale, bits) for _ in node.output]

    if node.op_type in {"Gather", "QEmbedding"}:
        scale = _get_attr_value(node, "scale_w", "scale_w_f")
        bits = _get_attr_value(node, "w_bits", "w_bits_i")
        bits = 8 if bits is None else int(bits)
        if scale is not None:
            return [_make_quant_state(scale, bits) for _ in node.output]

    if node.op_type in CUSTOM_QUANT_PASSTHROUGH_OPS:
        first_state = input_states[0] if input_states else _make_float_state()
        return [_copy_state(first_state) for _ in node.output]

    scale = _get_attr_value(node, "scale_o", "scale_o_f", "scale_add_o", "scale_add_o_f")
    bits = _get_attr_value(node, "o_bits", "o_bits_i")
    if scale is None or bits is None:
        return [_make_float_state() for _ in node.output]

    states = [_make_quant_state(scale, bits)]
    if node.op_type in {"QGRU", "QLSTM"}:
        states.extend(_make_float_state() for _ in node.output[1:])
    else:
        states.extend(_make_quant_state(scale, bits) for _ in node.output[1:])
    return states[:output_count]


def _standard_op_accepts_quant_input(node, input_index):
    if input_index != 0 and node.op_type not in {"Equal"}:
        return False
    if node.op_type in STANDARD_QUANT_INPUT_OPS:
        return True
    if node.op_type == "Equal":
        return True
    return False


def _node_accepts_quant_input(node, input_index):
    if node.domain == QDOMAIN_NAME:
        return (
            node.op_type == "Dequant"
            or (input_index == 0 and node.op_type in CUSTOM_QUANT_PASSTHROUGH_OPS)
            or _get_direct_quant_spec_for_input(node, input_index) is not None
        )
    return _standard_op_accepts_quant_input(node, input_index)


def _infer_standard_output_states(node, input_states, elem_types):
    output_count = len(node.output)
    if output_count == 0:
        return []

    first_state = input_states[0] if input_states else _make_float_state()
    first_output_type = elem_types.get(node.output[0])

    if node.op_type in STANDARD_QUANT_PASSTHROUGH_OPS and _is_quantized_state(first_state):
        return [_copy_state(first_state) for _ in node.output]

    if node.op_type == "Split" and _is_quantized_state(first_state):
        return [_copy_state(first_state) for _ in node.output]

    if node.op_type == "MaxPool" and _is_quantized_state(first_state):
        states = [_copy_state(first_state)]
        states.extend(_make_tensor_state(INT64_ELEM_TYPE) for _ in node.output[1:])
        return states[:output_count]

    if node.op_type == "Shape":
        return [_make_tensor_state(INT64_ELEM_TYPE) for _ in node.output]

    if node.op_type == "Constant":
        return [_constant_output_state(node) for _ in node.output]

    if node.op_type == "Cast":
        return [_cast_output_state(node, first_state) for _ in node.output]

    if node.op_type == "Equal":
        return [_make_tensor_state(BOOL_ELEM_TYPE) for _ in node.output]

    if node.op_type == "If":
        return [_make_tensor_state(elem_types.get(output, FLOAT_ELEM_TYPE)) for output in node.output]

    fallback_type = first_output_type
    if fallback_type is None:
        fallback_type = first_state.get("elem_type", FLOAT_ELEM_TYPE) if first_state else FLOAT_ELEM_TYPE
    return [_make_tensor_state(elem_types.get(output, fallback_type)) for output in node.output]


def _infer_node_output_states(node, input_states, elem_types):
    if node.domain == QDOMAIN_NAME:
        return _custom_output_states(node, input_states)
    return _infer_standard_output_states(node, input_states, elem_types)


def _make_dequant_node(input_name, output_name, scale):
    return helper.make_node(
        "Dequant",
        inputs=[input_name],
        outputs=[output_name],
        domain=QDOMAIN_NAME,
        scale_o=float(scale),
    )


def _dequantize_graph_outputs(graph, states, used_names):
    appended_nodes = []
    for output in graph.output:
        output_state = states.get(output.name)
        if not _is_quantized_state(output_state):
            continue

        original_output_name = output.name
        internal_output_name = _unique_value_name(f"{original_output_name}_quant", used_names)

        producer_found = False
        for node in graph.node:
            for output_index, node_output in enumerate(node.output):
                if node_output == original_output_name:
                    node.output[output_index] = internal_output_name
                    producer_found = True
                    break
            if producer_found:
                break

        if not producer_found:
            dequant_output_name = _unique_value_name(f"{original_output_name}_dequant", used_names)
            appended_nodes.append(_make_dequant_node(original_output_name, dequant_output_name, output_state["scale"]))
            output.name = dequant_output_name
            states[dequant_output_name] = _make_float_state()
            continue

        for node in graph.node:
            for input_index, input_name in enumerate(node.input):
                if input_name == original_output_name:
                    node.input[input_index] = internal_output_name

        states[internal_output_name] = _copy_state(output_state)
        states[original_output_name] = _make_float_state()
        appended_nodes.append(_make_dequant_node(internal_output_name, original_output_name, output_state["scale"]))

    graph.node.extend(appended_nodes)


def _propagate_tensor_states(graph, insert_dequant=False):
    elem_types = _existing_tensor_elem_types(graph)
    states = _initial_tensor_states(graph, elem_types)
    used_names = _all_value_names(graph)
    new_nodes = []

    for node in list(graph.node):
        input_states = []
        if insert_dequant:
            for input_index, input_name in enumerate(node.input):
                input_state = states.get(input_name, _make_tensor_state(elem_types.get(input_name, FLOAT_ELEM_TYPE)))
                if input_name and _is_quantized_state(input_state) and not _node_accepts_quant_input(node, input_index):
                    dequant_output = _unique_value_name(f"{input_name}_dequant", used_names)
                    new_nodes.append(_make_dequant_node(input_name, dequant_output, input_state["scale"]))
                    states[dequant_output] = _make_float_state()
                    elem_types[dequant_output] = FLOAT_ELEM_TYPE
                    node.input[input_index] = dequant_output
                    input_state = states[dequant_output]
                input_states.append(input_state)
            new_nodes.append(node)
        else:
            input_states = [
                states.get(input_name, _make_tensor_state(elem_types.get(input_name, FLOAT_ELEM_TYPE)))
                for input_name in node.input
            ]

        output_states = _infer_node_output_states(node, input_states, elem_types)
        for output_name, output_state in zip(node.output, output_states):
            if not output_name:
                continue
            states[output_name] = output_state
            elem_types[output_name] = output_state["elem_type"]

    if insert_dequant:
        _clear_repeated_field(graph, "node")
        graph.node.extend(new_nodes)
        _dequantize_graph_outputs(graph, states, used_names)

    return states


def _insert_dequant_nodes(onnx_model):
    _propagate_tensor_states(onnx_model.graph, insert_dequant=True)
    return onnx_model


def _find_quant_boundary_errors(graph):
    elem_types = _existing_tensor_elem_types(graph)
    states = _initial_tensor_states(graph, elem_types)
    initializer_names = {initializer.name for initializer in graph.initializer}
    errors = []

    for node in graph.node:
        input_states = [
            states.get(input_name, _make_tensor_state(elem_types.get(input_name, FLOAT_ELEM_TYPE)))
            for input_name in node.input
        ]
        for input_index, (input_name, input_state) in enumerate(zip(node.input, input_states)):
            spec = _get_direct_quant_spec_for_input(node, input_index)
            if (
                input_name
                and spec is not None
                and input_name not in initializer_names
                and not _is_quantized_state(input_state)
            ):
                errors.append(
                    f"{node.name or node.op_type} input[{input_index}] {input_name} "
                    f"requires Quant but is not produced by a quantized tensor"
                )
            if input_name and _is_quantized_state(input_state) and not _node_accepts_quant_input(node, input_index):
                errors.append(
                    f"{node.name or node.op_type} input[{input_index}] {input_name} "
                    f"still carries quantized data without Dequant"
                )

        output_states = _infer_node_output_states(node, input_states, elem_types)
        for output_name, output_state in zip(node.output, output_states):
            if not output_name:
                continue
            states[output_name] = output_state
            elem_types[output_name] = output_state["elem_type"]

    return errors


def _fold_initializer_quant_nodes(onnx_model):
    graph = onnx_model.graph
    initializer_map = {initializer.name: initializer for initializer in graph.initializer}
    graph_output_names = {output.name for output in graph.output}
    quantized_cache = {}
    replacements = {}

    for node in graph.node:
        if node.domain != QDOMAIN_NAME or node.op_type != "Quant":
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            continue
        if node.input[0] not in initializer_map or node.output[0] in graph_output_names:
            continue
        spec = _get_quant_node_input_spec(node, 0)
        input_array = numpy_helper.to_array(initializer_map[node.input[0]])
        if input_array.dtype.kind in {"f"}:
            if spec is None or spec["scale"] is None:
                continue
            quantized_array = quant_weight_bias(
                input_array,
                spec["bits"],
                spec["scale"],
                spec["quant_mode"],
            )
            _quantize_initializer_like(
                graph,
                initializer_map,
                node,
                0,
                quantized_array,
                quantized_cache,
            )
        replacements[node.output[0]] = node.input[0]

    if not replacements:
        return onnx_model

    for node in graph.node:
        for input_index, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[input_index] = replacements[input_name]

    kept_nodes = [
        node for node in graph.node
        if not (
            node.domain == QDOMAIN_NAME
            and node.op_type == "Quant"
            and len(node.output) == 1
            and node.output[0] in replacements
        )
    ]
    _clear_repeated_field(graph, "node")
    graph.node.extend(kept_nodes)
    return onnx_model


def _shorten_tensor_names(onnx_model):
    graph = onnx_model.graph
    initializer_names = {initializer.name for initializer in graph.initializer}
    preserved_names = {value.name for value in graph.input}
    preserved_names.update(value.name for value in graph.output)
    rename_map = {}
    next_id = 0

    for node in graph.node:
        for output_name in node.output:
            if output_name in preserved_names or output_name in initializer_names:
                continue
            if output_name not in rename_map:
                rename_map[output_name] = str(next_id)
                next_id += 1

    def _rename(name):
        return rename_map.get(name, name)

    def _rename_node(node):
        output_name = next((output_name for output_name in node.output if output_name), None)
        if output_name is not None:
            node.name = f"{node.op_type}_{output_name}"

    def _rename_graph_values(target_graph):
        for node in target_graph.node:
            for index, input_name in enumerate(node.input):
                node.input[index] = _rename(input_name)
            for index, output_name in enumerate(node.output):
                node.output[index] = _rename(output_name)
            _rename_node(node)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    _rename_graph_values(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        _rename_graph_values(subgraph)

        for value in target_graph.input:
            value.name = _rename(value.name)
        for value in target_graph.output:
            value.name = _rename(value.name)
        for value in target_graph.value_info:
            value.name = _rename(value.name)

    _rename_graph_values(graph)

    return onnx_model


def _clear_repeated_field(container, field_name):
    del getattr(container, field_name)[:]


def _update_tensor_elem_type(value_info, elem_type, keep_shape):
    tensor_type = value_info.type.tensor_type
    tensor_type.elem_type = elem_type
    if not keep_shape:
        _clear_repeated_field(tensor_type.shape, "dim")


def _annotate_tensor_types(onnx_model, keep_graph_io_shape=True):
    graph = onnx_model.graph
    tensor_states = _propagate_tensor_states(graph, insert_dequant=False)
    initializer_names = {initializer.name for initializer in graph.initializer}

    existing_value_info = {value.name: value for value in graph.value_info}
    _clear_repeated_field(graph, "value_info")

    for node in graph.node:
        for output_name in node.output:
            state = tensor_states.get(output_name)
            elem_type = state["elem_type"] if state is not None else None
            if elem_type is None:
                continue
            value_info = existing_value_info.get(output_name)
            if value_info is None:
                value_info = helper.make_tensor_value_info(output_name, elem_type, None)
            _update_tensor_elem_type(value_info, elem_type, keep_shape=True)
            graph.value_info.append(value_info)

    for value in graph.input:
        if value.name in initializer_names:
            continue
        state = tensor_states.get(value.name)
        elem_type = state["elem_type"] if state is not None else None
        if elem_type is not None:
            _update_tensor_elem_type(value, elem_type, keep_shape=keep_graph_io_shape)

    for value in graph.output:
        state = tensor_states.get(value.name)
        elem_type = state["elem_type"] if state is not None else None
        if elem_type is not None:
            _update_tensor_elem_type(value, elem_type, keep_shape=keep_graph_io_shape)

    return onnx_model


def _run_model_for_export_prepare(model, args):
    with torch.no_grad():
        if isinstance(args, Mapping):
            try:
                model(args)
            except TypeError:
                model(**args)
        elif isinstance(args, (tuple, list)):
            model(*args)
        else:
            model(args)


def _is_path_like(target):
    return isinstance(target, (str, os.PathLike, Path))


def _compatible_torch_onnx_export_kwargs(export_kwargs):
    export_kwargs = dict(export_kwargs)
    try:
        signature = inspect.signature(torch_onnx_export)
    except (TypeError, ValueError):
        return export_kwargs

    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return export_kwargs

    if "use_external_data_format" in export_kwargs and "use_external_data_format" not in parameters:
        use_external_data = export_kwargs.pop("use_external_data_format")
        if "external_data" in parameters and "external_data" not in export_kwargs:
            export_kwargs["external_data"] = use_external_data

    if "external_data" in export_kwargs and "external_data" not in parameters:
        external_data = export_kwargs.pop("external_data")
        if "use_external_data_format" in parameters and "use_external_data_format" not in export_kwargs:
            export_kwargs["use_external_data_format"] = external_data

    return {key: value for key, value in export_kwargs.items() if key in parameters}


def _export_to_onnx_model(model, args, output_target, export_kwargs):
    export_kwargs = dict(export_kwargs)
    export_kwargs.setdefault("keep_initializers_as_inputs", False)
    export_kwargs = _compatible_torch_onnx_export_kwargs(export_kwargs)
    use_external_data = bool(
        export_kwargs.get("use_external_data_format", False)
        or export_kwargs.get("external_data", False)
    )

    if use_external_data:
        if not _is_path_like(output_target):
            raise ValueError("export with external data requires a filesystem path output target")
        output_path = os.fspath(output_target)
        torch_onnx_export(model, args, output_path, **export_kwargs)
        return onnx.load(output_path)

    tmp = BytesIO()
    torch_onnx_export(model, args, tmp, **export_kwargs)
    tmp.seek(0)
    return onnx.load(tmp)


def _uses_external_data(export_kwargs):
    return bool(
        export_kwargs.get("use_external_data_format", False)
        or export_kwargs.get("external_data", False)
    )


def _save_onnx_model(onnx_model, output_target, export_kwargs):
    if not _uses_external_data(export_kwargs):
        onnx.save(onnx_model, output_target)
        return

    if not _is_path_like(output_target):
        raise ValueError("export with external data requires a filesystem path output target")

    output_path = os.fspath(output_target)
    external_location = os.path.basename(output_path) + ".data"
    onnx.save_model(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_location,
        size_threshold=1024,
    )


def _validate_quantized_initializers(onnx_model):
    graph = onnx_model.graph
    initializer_map = {initializer.name: initializer for initializer in graph.initializer}
    errors = []

    for node in graph.node:
        for input_index, input_name in enumerate(node.input):
            initializer = initializer_map.get(input_name)
            if initializer is None:
                continue

            spec = _get_direct_quant_spec_for_input(node, input_index)
            if spec is None:
                continue

            expected_dtype = np.dtype(_int_dtype_for_bits(spec["bits"]))
            actual_dtype = numpy_helper.to_array(initializer).dtype
            if actual_dtype != expected_dtype:
                errors.append(
                    f"{node.name or node.op_type} input[{input_index}] "
                    f"{input_name}: expected {expected_dtype}, got {actual_dtype}"
                )

    if errors:
        preview = "\n  ".join(errors[:20])
        extra = "" if len(errors) <= 20 else f"\n  ... and {len(errors) - 20} more"
        raise RuntimeError(
            "ONNX export produced non-quantized initializer(s) for quantized inputs:\n  "
            + preview
            + extra
        )


def _validate_onnx_model(onnx_model):
    _validate_quantized_initializers(onnx_model)
    boundary_errors = _find_quant_boundary_errors(onnx_model.graph)
    if boundary_errors:
        preview = "\n  ".join(boundary_errors[:20])
        extra = "" if len(boundary_errors) <= 20 else f"\n  ... and {len(boundary_errors) - 20} more"
        raise RuntimeError(
            "ONNX export has invalid quant/dequant boundaries:\n  "
            + preview
            + extra
        )
    try:
        onnx.checker.check_model(onnx_model)
    except Exception as exc:
        raise RuntimeError(f"ONNX checker failed after linger export: {exc}") from exc


def convert_parameter_from_float_to_int(onnx_model):
    onnx_model = _fold_initializer_quant_nodes(onnx_model)
    graph = onnx_model.graph
    initializer_map = {init.name: init for init in graph.initializer}
    quantized_cache = {}

    for node in graph.node:
        for input_index, input_name in enumerate(node.input):
            if input_name not in initializer_map:
                continue
            input_array = numpy_helper.to_array(initializer_map[input_name])
            if input_array.dtype.kind not in {"f"}:
                continue
            spec = _get_quant_spec_for_input(graph, node, input_index)
            if spec is None or spec["scale"] is None:
                continue
            quantized_array = quant_weight_bias(
                input_array,
                spec["bits"],
                spec["scale"],
                spec["quant_mode"],
            )
            _quantize_initializer_like(
                graph,
                initializer_map,
                node,
                input_index,
                quantized_array,
                quantized_cache,
            )

    _remove_unused_initializers(graph)
    return onnx_model


def _finalize_onnx_graph(onnx_model):
    _remove_initializer_inputs(onnx_model.graph)
    onnx_model = _shorten_tensor_names(onnx_model)
    onnx_model = _annotate_tensor_types(onnx_model)
    _validate_onnx_model(onnx_model)
    return onnx_model


def _postprocess_onnx_model(onnx_model):
    onnx_model = optimize_onnx_graph(onnx_model)
    onnx_model = convert_parameter_from_float_to_int(onnx_model)
    onnx_model = _insert_dequant_nodes(onnx_model)
    return _finalize_onnx_graph(onnx_model)


def _ensure_state_dict_stable(model, args):
    """预执行模型，触发懒初始化，确保 state_dict keys 在 trace 前稳定。

    Transformer 等网络的位置编码模块可能在首次 forward 时动态注册 buffer，
    导致 torch.jit.trace 检测到 state_dict 变化而报错。
    通过预执行消除这一问题。
    """
    keys_before = set(model.state_dict().keys())

    training = model.training
    model.eval()
    try:
        _run_model_for_export_prepare(model, args)
    finally:
        model.train(training)

    keys_after = set(model.state_dict().keys())
    new_keys = keys_after - keys_before

    if new_keys:
        print(f"[linger.onnx.export] 预执行检测到 {len(new_keys)} 个新注册的 "
              f"buffer/parameter（可能来自位置编码等懒初始化模块）：")
        for k in sorted(new_keys):
            print(f"  + {k}")
        print("已通过预执行完成初始化，继续导出。")

    return len(new_keys) > 0

def export(model, args, f, **kwargs):
    # 0. 预执行：触发 transformer 等网络的懒初始化，防止 trace 时 state_dict 变化
    _ensure_state_dict_stable(model, args)
    
    modules = list(model.modules()) if hasattr(model, "modules") else []
    for module in modules:
        setattr(module, "_linger_prepare_onnx_export", True)
    try:
        _run_model_for_export_prepare(model, args)
    finally:
        for module in modules:
            setattr(module, "_linger_prepare_onnx_export", False)

    onnx_model = _export_to_onnx_model(model, args, f, kwargs)
    onnx_model = _postprocess_onnx_model(onnx_model)
    _save_onnx_model(onnx_model, f, kwargs)

    # inferred_model = shape_inference.infer_shapes(onnx_model)

    # 3. 覆盖保存，让最终导出的文件带有 shape
    # onnx.save(inferred_model, f)

def generate_onnx_qparam_dict(cls, input_list = False):
    # qparam_dict = {'platform_s': str(QUANT_CONFIGS.platform.name), 'quant_mode_s': str(cls.output_quantizer.round_mode.name)}
    qparam_dict = {'platform_s': str(QUANT_CONFIGS.platform.name)}
    if input_list:
        qparam_dict['x_bits_i'] = int(cls.input_quantizer[0].data_bits)
        qparam_dict['scale_x_f'] = float(cls.input_quantizer[0].scale)
        qparam_dict['is_x_qtensor_i'] = int(getattr(cls.input_quantizer[0], 'is_qtensor', False))
        if len(cls.input_quantizer) > 1:
            qparam_dict['y_bits_i'] = int(cls.input_quantizer[1].data_bits)
            qparam_dict['scale_y_f'] = float(cls.input_quantizer[1].scale)
            qparam_dict['is_y_qtensor_i'] = int(getattr(cls.input_quantizer[1], 'is_qtensor', False))
    elif hasattr(cls, "input_quantizer"):
        qparam_dict['x_bits_i'] = int(cls.input_quantizer.data_bits)
        qparam_dict['scale_x_f'] = float(cls.input_quantizer.scale)
        if hasattr(cls.input_quantizer, 'is_qtensor'):
            qparam_dict['is_input_qtensor'] = cls.input_quantizer.is_qtensor
    if hasattr(cls, 'output_quantizer'):
        qparam_dict['quant_mode_s'] = str(cls.output_quantizer.round_mode.name)
        qparam_dict['o_bits_i'] = int(cls.output_quantizer.data_bits)
        qparam_dict['scale_o_f'] = float(cls.output_quantizer.scale)
    if hasattr(cls, 'weight_quantizer') and cls.weight_quantizer is not None:
        qparam_dict['w_bits_i'] = int(cls.weight_quantizer.data_bits)
        qparam_dict['scale_w_f'] = float(cls.weight_quantizer.scale)
    qparam_dict['op_type'] = cls._get_name()

    if 'Cat' in qparam_dict['op_type']:
        qparam_dict['x_0_bits_i'] = qparam_dict['x_bits_i']
        qparam_dict['x_1_bits_i'] = qparam_dict['y_bits_i']
        qparam_dict['scale_x_0_f'] = qparam_dict['scale_x_f']
        qparam_dict['scale_x_1_f'] = qparam_dict['scale_y_f']
        qparam_dict['is_x_0_qtensor_i'] = qparam_dict.pop('is_x_qtensor_i', 0)
        qparam_dict['is_x_1_qtensor_i'] = qparam_dict.pop('is_y_qtensor_i', 0)
        qparam_dict.pop('x_bits_i', None)
        qparam_dict.pop('y_bits_i', None)
        qparam_dict.pop('scale_x_f', None)
        qparam_dict.pop('scale_y_f', None)
    elif 'Softmax' in qparam_dict['op_type']:
        qparam_dict['axis_i'] = int(cls.dim)
        qparam_dict.pop('y_bits_i', None)
        qparam_dict.pop('scale_y_f', None)
    elif 'GLU' in qparam_dict['op_type']:
        qparam_dict['dim_i'] = int(cls.dim)
        qparam_dict.pop('y_bits_i', None)
        qparam_dict.pop('scale_y_f', None)
    elif 'LayerNorm' in qparam_dict['op_type']:
        qparam_dict['scale_y_normal_f'] = float(LAYERNORM_NORMAL_SCALE)
    elif 'ConvTranspose' in qparam_dict['op_type']:
        qparam_dict['dilations_i'] = cls.dilation
        qparam_dict['kernel_shape_i'] = cls.kernel_size
        qparam_dict['pads_i'] = cls.padding * 2
        qparam_dict['strides_i'] = cls.stride
        qparam_dict['group_i'] = cls.groups
        qparam_dict['output_padding_i'] = cls.output_padding
    elif 'Conv' in qparam_dict['op_type']:
        qparam_dict['dilations_i'] = cls.dilation
        qparam_dict['kernel_shape_i'] = cls.kernel_size
        qparam_dict['pads_i'] = cls.padding * 2
        qparam_dict['strides_i'] = cls.stride
        qparam_dict['group_i'] = cls.groups
    elif 'AdaptiveAvgPool2d' in qparam_dict['op_type']:
        if tuple(_pair(cls.output_size)) != (1, 1):
            raise NotImplementedError("QAdaptiveAvgPool2d ONNX export only supports output_size=(1, 1)")
        qparam_dict['op_type'] = 'QGlobalAveragePool2d'
    elif 'AvgPool' in qparam_dict['op_type']:
        tuple_fn = _single if '1d' in qparam_dict['op_type'] else _pair
        qparam_dict['kernel_shape_i'] = tuple_fn(cls.kernel_size)
        qparam_dict['pads_i'] = tuple_fn(cls.padding) * 2
        qparam_dict['strides_i'] = tuple_fn(cls.stride)
        qparam_dict['ceil_mode_i'] = cls.ceil_mode
        qparam_dict['count_include_pad_i'] = int(cls.count_include_pad)
        if hasattr(cls, 'divisor_override'):
            qparam_dict['divisor_override_i'] = -1 if cls.divisor_override is None else int(cls.divisor_override)
    elif 'GRU' in qparam_dict['op_type'] or 'LSTM' in qparam_dict['op_type']:
        qparam_dict['input_size_i'] = int(cls.input_size)
        qparam_dict['hidden_size_i'] = int(cls.hidden_size)
        qparam_dict['num_layers_i'] = int(cls.num_layers)
        qparam_dict['batch_first_i'] = int(cls.batch_first)
        qparam_dict['go_forward_i'] = True
        qparam_dict['scale_h_f'] = float(cls.hidden_quantizer.scale)
        qparam_dict['scale_c_f'] = float(cls.cell_quantizer.scale)
        qparam_dict['w_bits_i'] = int(cls.weightih_quantizer.data_bits)
        qparam_dict['scale_iw_f'] = float(cls.weightih_quantizer.scale)
        qparam_dict['scale_hw_f'] = float(cls.weighthh_quantizer.scale)
        qparam_dict['outputs'] = 3 if 'LSTM' in qparam_dict['op_type'] else 2
        if cls.bidirectional:
            qparam_dict_r = {'platform_s': str(QUANT_CONFIGS.platform.name), 'quant_mode_s': str(cls.output_quantizer.round_mode.name)}
            qparam_dict_r['x_bits_i'] = int(cls.input_quantizer.data_bits)
            qparam_dict_r['scale_x_f'] = float(cls.input_quantizer.scale)
            qparam_dict_r['input_size_i'] = int(cls.input_size)
            qparam_dict_r['hidden_size_i'] = int(cls.hidden_size)
            qparam_dict_r['num_layers_i'] = int(cls.num_layers)
            qparam_dict_r['batch_first_i'] = int(cls.batch_first)
            qparam_dict_r['go_forward_i'] = False
            qparam_dict_r['scale_h_f'] = float(cls.hidden_reverse_quantizer.scale)
            qparam_dict_r['scale_c_f'] = float(cls.cell_quantizer.scale)
            qparam_dict_r['w_bits_i'] = int(cls.weightih_reverse_quantizer.data_bits)
            qparam_dict_r['scale_iw_f'] = float(cls.weightih_reverse_quantizer.scale)
            qparam_dict_r['scale_hw_f'] = float(cls.weighthh_reverse_quantizer.scale)
            qparam_dict_r['o_bits_i'] = int(cls.output_reverse_quantizer.data_bits)
            qparam_dict_r['scale_o_f'] = float(cls.output_reverse_quantizer.scale)
            qparam_dict_r['outputs'] = qparam_dict['outputs']
            qparam_dict['qparam_dict_r'] = qparam_dict_r
    return qparam_dict

def quantlinear(g, input, scale_x, platform, data_bits, zero_point):
    return g.op("linger::Quant", input, data_bits_i=data_bits, scale_x_f = scale_x, platform_s = platform, zero_point_i = zero_point)

def _qtensor_input_attr_keys(input_index):
    if input_index == 0:
        return [("scale_x_f", "x_bits_i", "is_x_qtensor_i"), ("scale_x_0_f", "x_0_bits_i", "is_x_0_qtensor_i")]
    if input_index == 1:
        return [("scale_y_f", "y_bits_i", "is_y_qtensor_i"), ("scale_x_1_f", "x_1_bits_i", "is_x_1_qtensor_i")]
    return [(f"scale_x_{input_index}_f", f"x_{input_index}_bits_i", f"is_x_{input_index}_qtensor_i")]


def quant_qtensor_symbolic_input(g, tensor, qparam_dict, input_index):
    is_qtensor = False
    for _, _, is_qtensor_key in _qtensor_input_attr_keys(input_index):
        is_qtensor = bool(qparam_dict.pop(is_qtensor_key, is_qtensor)) or is_qtensor

    for scale_key, bits_key, _ in _qtensor_input_attr_keys(input_index):
        if scale_key not in qparam_dict or bits_key not in qparam_dict:
            continue
        if is_qtensor:
            return tensor
        return quantlinear(
            g,
            tensor,
            float(qparam_dict[scale_key]),
            str(QUANT_CONFIGS.platform.name),
            int(qparam_dict[bits_key]),
            0,
        )
    return tensor


__all__ = ['export', 'convert_parameter_from_float_to_int', 'quant_weight_bias', 'quant_qtensor_symbolic_input']
