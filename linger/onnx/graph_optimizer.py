import re

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def _get_attr_value(node, *names):
    for attr in node.attribute:
        if attr.name in names:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            if attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            if attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            if attr.type == onnx.AttributeProto.STRING:
                return attr.s
            if attr.type == onnx.AttributeProto.TENSOR:
                return numpy_helper.to_array(attr.t)
    return None


def _is_graph_output(graph, value_name):
    return any(output.name == value_name for output in graph.output)


def _sanitize_name(name):
    name = name or "tensor"
    return re.sub(r"[^0-9A-Za-z_.]+", "_", name).strip("._") or "tensor"


def _make_initializer(name, value):
    array = np.asarray(value)
    if array.dtype == np.dtype("O"):
        raise TypeError("object dtype is not supported for ONNX initializers")
    return numpy_helper.from_array(array, name)


def _collect_initializer_map(graph):
    return {initializer.name: initializer for initializer in graph.initializer}


def _collect_constant_values(graph):
    constant_values = {}
    for initializer in graph.initializer:
        constant_values[initializer.name] = numpy_helper.to_array(initializer)
    for node in graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        tensor_value = _get_attr_value(node, "value")
        if tensor_value is not None:
            constant_values[node.output[0]] = tensor_value
    return constant_values


def _extract_static_shape_map(model):
    static_shape_map = {}
    value_infos = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    for value_info in value_infos:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        known = True
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            else:
                known = False
                break
        if known:
            static_shape_map[value_info.name] = np.asarray(dims, dtype=np.int64)
    return static_shape_map


def _constant_of_shape_dtype(node):
    for attr in node.attribute:
        if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
            tensor = numpy_helper.to_array(attr.t)
            return tensor.dtype, tensor.reshape(-1)[0]
    return np.float32, np.float32(0.0)


def _eval_constant_node(node, constant_values, static_shape_map):
    if any(input_name not in constant_values for input_name in node.input):
        if node.op_type != "Shape":
            return None
    if len(node.output) != 1:
        return None

    inputs = [constant_values[input_name] for input_name in node.input if input_name in constant_values]
    op_type = node.op_type

    try:
        if op_type == "Identity":
            return inputs[0]
        if op_type == "Cast":
            to_type = _get_attr_value(node, "to")
            if to_type is None:
                return None
            return inputs[0].astype(helper.tensor_dtype_to_np_dtype(to_type))
        if op_type == "Reshape":
            shape = np.asarray(inputs[1], dtype=np.int64).tolist()
            return np.reshape(inputs[0], shape)
        if op_type == "Transpose":
            perm = _get_attr_value(node, "perm")
            return np.transpose(inputs[0], axes=None if perm is None else tuple(perm))
        if op_type == "Unsqueeze":
            axes = _get_attr_value(node, "axes")
            if axes is None and len(inputs) > 1:
                axes = np.asarray(inputs[1], dtype=np.int64).tolist()
            if axes is None:
                return None
            result = inputs[0]
            for axis in sorted(int(axis) for axis in axes):
                result = np.expand_dims(result, axis)
            return result
        if op_type == "Squeeze":
            axes = _get_attr_value(node, "axes")
            if axes is None and len(inputs) > 1:
                axes = np.asarray(inputs[1], dtype=np.int64).tolist()
            if axes is None:
                return np.squeeze(inputs[0])
            result = inputs[0]
            for axis in sorted((int(axis) for axis in axes), reverse=True):
                result = np.squeeze(result, axis=axis)
            return result
        if op_type == "Concat":
            axis = int(_get_attr_value(node, "axis") or 0)
            return np.concatenate(inputs, axis=axis)
        if op_type == "Gather":
            axis = int(_get_attr_value(node, "axis") or 0)
            return np.take(inputs[0], np.asarray(inputs[1], dtype=np.int64), axis=axis)
        if op_type == "Slice":
            data = inputs[0]
            starts = np.asarray(inputs[1], dtype=np.int64).tolist()
            ends = np.asarray(inputs[2], dtype=np.int64).tolist()
            axes = np.asarray(inputs[3], dtype=np.int64).tolist() if len(inputs) > 3 else list(range(len(starts)))
            steps = np.asarray(inputs[4], dtype=np.int64).tolist() if len(inputs) > 4 else [1] * len(starts)
            slices = [slice(None)] * data.ndim
            for axis, start, end, step in zip(axes, starts, ends, steps):
                slices[int(axis)] = slice(int(start), int(end), int(step))
            return data[tuple(slices)]
        if op_type == "Flatten":
            axis = int(_get_attr_value(node, "axis") or 1)
            shape = inputs[0].shape
            left = int(np.prod(shape[:axis], dtype=np.int64)) if axis > 0 else 1
            right = int(np.prod(shape[axis:], dtype=np.int64))
            return np.reshape(inputs[0], (left, right))
        if op_type == "Shape":
            if node.input[0] in constant_values:
                return np.asarray(constant_values[node.input[0]].shape, dtype=np.int64)
            if node.input[0] in static_shape_map:
                return static_shape_map[node.input[0]]
            return None
        if op_type == "ConstantOfShape":
            shape = tuple(int(dim) for dim in np.asarray(inputs[0], dtype=np.int64).reshape(-1).tolist())
            dtype, fill_value = _constant_of_shape_dtype(node)
            return np.full(shape, fill_value, dtype=dtype)
        if op_type == "Expand":
            target_shape = tuple(int(dim) for dim in np.asarray(inputs[1], dtype=np.int64).reshape(-1).tolist())
            return np.broadcast_to(inputs[0], target_shape).copy()
        if op_type == "Size":
            return np.asarray(np.size(inputs[0]), dtype=np.int64)
        if op_type == "Equal":
            return np.equal(inputs[0], inputs[1])
        if op_type == "Where":
            return np.where(inputs[0], inputs[1], inputs[2])
        if op_type == "ReduceProd":
            axes = _get_attr_value(node, "axes")
            keepdims = int(_get_attr_value(node, "keepdims") if _get_attr_value(node, "keepdims") is not None else 1)
            return np.prod(inputs[0], axis=None if axes is None else tuple(axes), keepdims=bool(keepdims))
        if op_type == "ReduceSum":
            axes = _get_attr_value(node, "axes")
            keepdims = int(_get_attr_value(node, "keepdims") if _get_attr_value(node, "keepdims") is not None else 1)
            return np.sum(inputs[0], axis=None if axes is None else tuple(axes), keepdims=bool(keepdims))
        if op_type == "ReduceMean":
            axes = _get_attr_value(node, "axes")
            keepdims = int(_get_attr_value(node, "keepdims") if _get_attr_value(node, "keepdims") is not None else 1)
            return np.mean(inputs[0], axis=None if axes is None else tuple(axes), keepdims=bool(keepdims))
        if op_type == "Max":
            return np.maximum(inputs[0], inputs[1])
        if op_type == "Min":
            return np.minimum(inputs[0], inputs[1])
        if op_type == "Pow":
            return np.power(inputs[0], inputs[1])
        if op_type == "Mod":
            return np.mod(inputs[0], inputs[1])
        if op_type == "Ceil":
            return np.ceil(inputs[0])
        if op_type == "Floor":
            return np.floor(inputs[0])
        if op_type == "Relu":
            return np.maximum(inputs[0], 0)
        if op_type == "Clip":
            min_value = inputs[1] if len(inputs) > 1 else _get_attr_value(node, "min")
            max_value = inputs[2] if len(inputs) > 2 else _get_attr_value(node, "max")
            return np.clip(inputs[0], min_value, max_value)
        if op_type == "Constant":
            return None
        if op_type == "Range":
            return np.arange(inputs[0], inputs[1], inputs[2], dtype=np.result_type(inputs[0], inputs[1], inputs[2]))
        if op_type == "Tile":
            return np.tile(inputs[0], np.asarray(inputs[1], dtype=np.int64))
        if op_type == "Pad":
            pads = np.asarray(inputs[1], dtype=np.int64).tolist()
            rank = inputs[0].ndim
            begin = pads[:rank]
            end = pads[rank:]
            constant_value = inputs[2].reshape(-1)[0] if len(inputs) > 2 else 0
            return np.pad(inputs[0], list(zip(begin, end)), mode="constant", constant_values=constant_value)
        if op_type == "Split":
            axis = int(_get_attr_value(node, "axis") or 0)
            split = _get_attr_value(node, "split")
            if split is None and len(inputs) > 1:
                split = np.asarray(inputs[1], dtype=np.int64).tolist()
            if split is None:
                return None
            outputs = np.split(inputs[0], np.cumsum(split)[:-1], axis=axis)
            if len(node.output) == len(outputs):
                return outputs
            return np.asarray(inputs[0].shape, dtype=np.int64)
        if op_type == "Add":
            return np.add(inputs[0], inputs[1])
        if op_type == "Sub":
            return np.subtract(inputs[0], inputs[1])
        if op_type == "Mul":
            return np.multiply(inputs[0], inputs[1])
        if op_type == "Div":
            return np.divide(inputs[0], inputs[1])
        if op_type == "Neg":
            return np.negative(inputs[0])
        if op_type == "Abs":
            return np.abs(inputs[0])
    except Exception:
        return None

    return None


def _replace_node_output_uses(graph, old_name, new_name):
    for node in graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name


def _lift_constant_nodes_to_initializers(model):
    graph = model.graph
    initializer_map = _collect_initializer_map(graph)
    removable_nodes = []
    changed = False

    for node in graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        if _is_graph_output(graph, node.output[0]):
            continue
        tensor_value = _get_attr_value(node, "value")
        if tensor_value is None:
            continue
        if node.output[0] not in initializer_map:
            graph.initializer.append(_make_initializer(node.output[0], tensor_value))
            initializer_map[node.output[0]] = graph.initializer[-1]
        removable_nodes.append(node)
        changed = True

    for node in removable_nodes:
        graph.node.remove(node)
    return changed


def _fold_constant_subgraphs(model):
    graph = model.graph
    initializer_map = _collect_initializer_map(graph)
    constant_values = _collect_constant_values(graph)
    static_shape_map = _extract_static_shape_map(model)
    removable_nodes = []
    changed = False

    for node in list(graph.node):
        if node.op_type == "Constant":
            continue
        if any(_is_graph_output(graph, output_name) for output_name in node.output):
            continue
        folded = _eval_constant_node(node, constant_values, static_shape_map)
        if folded is None:
            continue
        if len(node.output) == 1:
            folded = [folded]
        for output_name, output_value in zip(node.output, folded):
            if output_name in initializer_map:
                graph.initializer.remove(initializer_map[output_name])
            graph.initializer.append(_make_initializer(output_name, output_value))
            initializer_map[output_name] = graph.initializer[-1]
            constant_values[output_name] = output_value
        removable_nodes.append(node)
        changed = True

    for node in removable_nodes:
        graph.node.remove(node)
    return changed


def _eliminate_identity_nodes(model):
    graph = model.graph
    changed = False

    for node in list(graph.node):
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        input_name = node.input[0]
        output_name = node.output[0]

        if _is_graph_output(graph, output_name):
            for other in graph.node:
                for index, other_output in enumerate(other.output):
                    if other_output == input_name:
                        other.output[index] = output_name
            for other in graph.node:
                if other is node:
                    continue
                for index, other_input in enumerate(other.input):
                    if other_input == input_name:
                        other.input[index] = output_name
        else:
            _replace_node_output_uses(graph, output_name, input_name)

        graph.node.remove(node)
        changed = True

    return changed


def _canonicalize_parameter_initializer_names(model):
    graph = model.graph
    initializer_map = _collect_initializer_map(graph)
    if not initializer_map:
        return False

    consumer_count = {}
    for node in graph.node:
        for input_name in node.input:
            consumer_count[input_name] = consumer_count.get(input_name, 0) + 1

    changed = False
    existing_names = {initializer.name for initializer in graph.initializer}

    for node in graph.node:
        scale_w = _get_attr_value(node, "scale_w", "scale_w_f")
        scale_x = _get_attr_value(node, "scale_x", "scale_x_f")
        role_specs = []

        if scale_w is not None and len(node.input) > 1 and node.input[1] in initializer_map:
            role_specs.append((1, "weight"))
        if scale_w is not None and scale_x is not None and len(node.input) > 2 and node.input[2] in initializer_map:
            role_specs.append((2, "bias"))

        for input_index, suffix in role_specs:
            old_name = node.input[input_index]
            if old_name.endswith(suffix):
                continue

            base_name = _sanitize_name(node.name or (node.output[0] if node.output else node.op_type))
            new_name = f"{base_name}.{suffix}"
            dedup_index = 0
            while new_name in existing_names and new_name != old_name:
                dedup_index += 1
                new_name = f"{base_name}_{dedup_index}.{suffix}"

            old_initializer = initializer_map.get(old_name)
            if old_initializer is None:
                continue

            if consumer_count.get(old_name, 0) <= 1:
                old_initializer.name = new_name
                initializer_map[new_name] = old_initializer
                initializer_map.pop(old_name, None)
            else:
                new_initializer = _make_initializer(new_name, numpy_helper.to_array(old_initializer))
                graph.initializer.append(new_initializer)
                initializer_map[new_name] = new_initializer
                consumer_count[old_name] -= 1

            node.input[input_index] = new_name
            existing_names.add(new_name)
            changed = True

    return changed


def _remove_unused_initializers(model):
    graph = model.graph
    used_names = set()
    for node in graph.node:
        used_names.update(node.input)
    used_names.update(value.name for value in graph.output)
    changed = False

    for initializer in list(graph.initializer):
        if initializer.name in used_names:
            continue
        graph.initializer.remove(initializer)
        changed = True

    return changed


def _remove_dead_nodes(model):
    graph = model.graph
    graph_outputs = {output.name for output in graph.output}
    changed = False

    while True:
        consumer_names = set(graph_outputs)
        for node in graph.node:
            consumer_names.update(node.input)

        removable = [
            node for node in graph.node
            if node.op_type not in {"If", "Loop", "Scan"} and all(output not in consumer_names for output in node.output)
        ]
        if not removable:
            break
        for node in removable:
            graph.node.remove(node)
        changed = True

    return changed


def _collect_external_value_refs(graph):
    defined_names = {value.name for value in graph.input}
    defined_names.update(initializer.name for initializer in graph.initializer)
    for node in graph.node:
        defined_names.update(output_name for output_name in node.output if output_name)

    refs = set()
    for node in graph.node:
        for input_name in node.input:
            if input_name and input_name not in defined_names:
                refs.add(input_name)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                refs.update(ref for ref in _collect_external_value_refs(attr.g) if ref not in defined_names)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    refs.update(ref for ref in _collect_external_value_refs(subgraph) if ref not in defined_names)
    return refs


def _node_dependency_inputs(node):
    input_names = set(input_name for input_name in node.input if input_name)
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            input_names.update(_collect_external_value_refs(attr.g))
        elif attr.type == onnx.AttributeProto.GRAPHS:
            for subgraph in attr.graphs:
                input_names.update(_collect_external_value_refs(subgraph))
    return input_names


def _sort_graph_topologically(graph):
    nodes = list(graph.node)
    if len(nodes) < 2:
        return False

    output_to_index = {}
    for index, node in enumerate(nodes):
        for output_name in node.output:
            if output_name:
                output_to_index[output_name] = index

    dependencies = [set() for _ in nodes]
    dependents = [set() for _ in nodes]
    for index, node in enumerate(nodes):
        for input_name in _node_dependency_inputs(node):
            producer_index = output_to_index.get(input_name)
            if producer_index is None or producer_index == index:
                continue
            dependencies[index].add(producer_index)
            dependents[producer_index].add(index)

    ready = [index for index, deps in enumerate(dependencies) if not deps]
    sorted_indices = []

    while ready:
        index = ready.pop(0)
        sorted_indices.append(index)
        for dependent_index in sorted(dependents[index]):
            dependencies[dependent_index].discard(index)
            if not dependencies[dependent_index] and dependent_index not in ready and dependent_index not in sorted_indices:
                ready.append(dependent_index)

    if len(sorted_indices) != len(nodes):
        return False

    if sorted_indices == list(range(len(nodes))):
        return False

    del graph.node[:]
    graph.node.extend(nodes[index] for index in sorted_indices)
    return True


def _sort_graphs_topologically(graph):
    changed = False
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                changed |= _sort_graphs_topologically(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    changed |= _sort_graphs_topologically(subgraph)
    changed |= _sort_graph_topologically(graph)
    return changed


def optimize_onnx_graph(model, infer_shapes=True, max_passes=8):
    if infer_shapes:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass

    for _ in range(max_passes):
        round_changed = False
        if infer_shapes:
            try:
                model = shape_inference.infer_shapes(model)
            except Exception:
                pass
        round_changed |= _lift_constant_nodes_to_initializers(model)
        round_changed |= _fold_constant_subgraphs(model)
        round_changed |= _eliminate_identity_nodes(model)
        round_changed |= _canonicalize_parameter_initializer_names(model)
        round_changed |= _remove_dead_nodes(model)
        round_changed |= _remove_unused_initializers(model)
        round_changed |= _sort_graphs_topologically(model.graph)
        if not round_changed:
            break

    if infer_shapes:
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass
    _sort_graphs_topologically(model.graph)

    return model


__all__ = ["optimize_onnx_graph"]
