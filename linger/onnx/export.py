from io import BytesIO

import torch
import torch.nn
import torch.onnx

import onnx

from .scope import build_global_scope, build_onnx_scope_info
from .update_dequant import parser_dequant

torch_onnx_export = torch.onnx.export


def export(model, args, f, export_params=True, verbose=False, training=False,
           input_names=None, output_names=None, aten=False, export_raw_ir=False,
           operator_export_type=None, opset_version=12, _retain_param_name=True,
           do_constant_folding=True, example_outputs=None, strip_doc_string=True,
           dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None,
           enable_onnx_checker=True, use_external_data_format=False, is_update_dequant=True,
           is_input_quant=False, is_scoped_info=False, debug_dump=False):
    print('change onnx export to linger export')
    if is_scoped_info:
        scopes_info = {}
        scopes_info = build_global_scope(model)

    if is_update_dequant:
        if isinstance(args, tuple):
            model.eval()
            model(*args)
            args = list(args)
            args = tuple([arg if not isinstance(arg, float)
                         else torch.tensor(arg) for arg in args])
        else:
            model.eval()
            model(args)
        tmp = BytesIO()
        torch_onnx_export(model, args, tmp, export_params, verbose, training, input_names, output_names, aten, export_raw_ir,
                          operator_export_type, opset_version, _retain_param_name, do_constant_folding, example_outputs, strip_doc_string,
                          dynamic_axes, keep_initializers_as_inputs, custom_opsets, enable_onnx_checker, use_external_data_format)
        tmp.seek(0)
        onnx_model = onnx.load(tmp)

        if debug_dump:
            onnx.save(onnx_model, "debug_onnx_torch_export.onnx")

        if is_scoped_info:
            onnx_model = build_onnx_scope_info(onnx_model)
            if debug_dump:
                onnx.save(onnx_model, "debug_onnx_scoped_info.onnx")
        
        onnx_model = parser_dequant(onnx_model, is_input_quant)
        if debug_dump:
            onnx.save(onnx_model, "debug_onnx_update_dequant.onnx")

        return onnx.save(onnx_model, f)
    else:
        print("Error:is_update_dequant is not support now")

    if is_scoped_info:
        for _, enter_hook, leave_hook in scopes_info.values():
            enter_hook.remove()
            leave_hook.remove()
        scopes_info.clear()


__all__ = ['export']
