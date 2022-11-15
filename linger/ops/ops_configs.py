import torch.nn as nn

from ..modules import *
from .avgpool2d_int import AvgPool2dInt
from .batchnorm_int import BatchNormInt
from .bmm_int import BmmInt
from .conv1d_int import Conv1dInt
from .conv_int import Conv2dInt
from .convtranspose_int import ConvTranspose2dInt
from .embedding_int import EmbeddingInt
from .gru_int import GRUInt
from .iqtensor import iqAddLayer, iqDivLayer, iqMulLayer, iqSumLayer
from .linear_int import LinearInt
from .linger_functional import (iqCatLayer, iqClampLayer, iqSigmoidLayer,
                                iqTanhLayer, softmaxInt, logsoftmaxInt)
from .lstm_int import LSTMInt
from .relu6_int import ReLU6Int
from .layernorm_int import LayerNormInt

DefaultQuantIntXOP = (nn.BatchNorm2d, nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.AvgPool2d, nn.Conv1d, NormalizeConvBN1d, NormalizeConvBN2d,
                      NormalizeConv2d, NormalizeConv1d, NormalizeConvTranspose2d, NormalizeLinear, NormalizeFastLSTM, NormalizeFastGRU,nn.ReLU6, nn.GRU, nn.LSTM)

SupportQuantTorchModules = [nn.BatchNorm2d, nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.GRU, nn.LSTM, nn.AvgPool2d, nn.Conv1d,  NormalizeConvBN1d, NormalizeConvBN2d, NormalizeConv2d,
                            NormalizeConv1d, NormalizeConvTranspose2d, NormalizeLinear, nn.ReLU6, NormalizeFastLSTM, NormalizeFastGRU, NormalizeBatchNorm2d, NormalizeLayerNorm, nn.Embedding, nn.Upsample, NormalizeEmbedding, nn.LayerNorm]
SupportQuantedIntModules = (BatchNormInt, LinearInt, Conv2dInt, ConvTranspose2dInt, GRUInt, LSTMInt, AvgPool2dInt, Conv1dInt, ReLU6Int, BmmInt,
                            iqAddLayer, iqMulLayer, iqDivLayer, iqSumLayer, iqCatLayer, iqSigmoidLayer, iqTanhLayer, iqClampLayer, EmbeddingInt, LayerNormInt, softmaxInt, logsoftmaxInt)
