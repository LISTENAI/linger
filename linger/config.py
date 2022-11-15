from enum import Enum

from .utils import PlatFormQuant, Singleton


class LayerFusion(Enum):
    LayerFusionConvBN = 1


class LayerQuantPredictor(Enum):
    LayerPredictorConvRelu = 1
    LayerPredictorBNRelu = 2


class Configure(Singleton):
    class PlatFormQuantConfig():
        platform_quant = PlatFormQuant.luna_quant

    class IQTensorConfig():
        iqmul = True
        iqadd = True
        iqcat = True
        iqclamp = True
        iqsigmoid = True
        iqdiv = True
        iqsum = True
        iqtanh = True
        softmaxint = True
        logsoftmaxint = True
        iqvar = True

    class FunctionConfig():
        linear = True
        bmm = False
        channel_shuffle = False

    class IQCat2AddZeroConfig():
        iqadd2addzero = False

    class BnMomentumUpdateConfig():
        disable = False

    PlatFormQuant = PlatFormQuantConfig()
    IQTensor = IQTensorConfig()
    FunctionQuant = FunctionConfig()
    IQCat2AddZero = IQCat2AddZeroConfig()
    BnMomentumUpdate = BnMomentumUpdateConfig()

config = Configure()

def SetPlatFormQuant(platform_quant=PlatFormQuant.luna_quant):
    r"""设置是否采用luna_quant量化方式

    Args:
        platform_quant(PlatForm.luna_quant):采用何种硬件量化方式，默认luna量化方式

    """
    config.PlatFormQuant.platform_quant = platform_quant



def SetIQTensorMul(enable):
    r"""设置是否启用iqadd功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.IQTensor.iqmul = enable


def SetIQTensorDiv(enable):
    r"""设置是否启用iqdiv功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.IQTensor.iqdiv = enable


def SetIQTensorAdd(enable):
    r"""设置是否启用iqadd功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.IQTensor.iqadd = enable


def SetIQTensorSum(enable):
    r"""设置是否启用iqdiv功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.IQTensor.iqsum = enable


def SetIQTensorCat(enable):
    r"""设置是否启用iqcat功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.IQTensor.iqcat = enable


def SetIQTensorSigmoid(enable):
    r"""设置是否启用iqsigmoid功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.IQTensor.iqsigmoid = enable

def SetIQTensorSoftmax(enable):
    r"""设置是否启用softmaxInt功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.IQTensor.softmaxint = enable

def SetIQTensorLogSoftmax(enable):
    r"""设置是否启用LogSoftmax功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.IQTensor.logsoftmaxint = enable

def SetIQTensorTanh(enable):
    r"""设置是否启用iqtanh功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.IQTensor.iqtanh = enable


def SetIQTensorClamp(enable):
    r"""设置是否启用iqclamp功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.IQTensor.iqclamp = enable

def SetIQTensorVar(enable):
    r"""设置是否启用iqVar功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.IQTensor.iqvar = enable


def SetFunctionLinearQuant(enable):
    r"""设置是否启用F.linear量化功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.FunctionQuant.linear = enable


def SetFunctionBmmQuant(enable):
    r"""设置是否启用torch.bmm量化功能,默认启用

    Args:
        enable(bool):是否启用

    """
    config.FunctionQuant.bmm = enable

def SetFunctionChannelShuffleQuant(enable):
    config.FunctionQuant.channel_shuffle = enable


def SetCastorBiasInt16(bias_int16=True):
    config.CastorBiasInt16.bias_int16 = bias_int16


def SetBnMomentumUpdate(disable=True):
    config.BnMomentumUpdate.disable = disable


def SetIQCat2AddZero(enable=True):
    config.IQCat2AddZero.iqadd2addzero = enable


__all__ = ['config', 'SetPlatFormQuant', 'SetIQTensorAdd', 'SetFunctionLinearQuant', 'SetFunctionBmmQuant', 'SetIQTensorClamp',
           'SetIQTensorCat', 'SetIQTensorSigmoid', 'SetIQTensorSoftmax', 'SetIQTensorLogSoftmax', 'SetIQTensorTanh', 'SetIQTensorDiv', 'SetIQTensorMul', 'SetIQTensorSum',
           'LayerFusion', 'LayerQuantPredictor', 'SetIQCat2AddZero', 'SetBnMomentumUpdate', 'SetIQTensorVar', 'SetFunctionChannelShuffleQuant']
