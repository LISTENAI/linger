from ..ops.ops_names import (LINGER_FUNCINT_BMM_COUNTER,
                             LINGER_IQTENSOR_LAYER_COUNTER)

self_module = []


def get_current_module():
    if len(self_module) > 0:
        return self_module[-1]
    else:
        return None


def hook_pre_forward(module, input):
    setattr(module, LINGER_IQTENSOR_LAYER_COUNTER, 0)
    setattr(module, LINGER_FUNCINT_BMM_COUNTER, 0)
    self_module.append(module)


def hook_forward(module, input, output):
    cur = self_module.pop()
    assert cur == module
    setattr(module, LINGER_IQTENSOR_LAYER_COUNTER, 0)
    setattr(module, LINGER_FUNCINT_BMM_COUNTER, 0)


__all__ = ['get_current_module', 'hook_pre_forward', 'hook_forward']
