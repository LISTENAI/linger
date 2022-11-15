#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from .normalize_layernorm import NormalizeLayerNorm
from .normalize_fastLSTM import NormalizeFastLSTM
from .normalize_fastGRU import NormalizeFastGRU
import torch.nn as nn

from .normalize_conv1d import NormalizeConv1d
from .normalize_conv2d import NormalizeConv2d
from .normalize_convbn1d import NormalizeConvBN1d
from .normalize_convbn2d import NormalizeConvBN2d
from .normalize_convTranspose2d import NormalizeConvTranspose2d
from .normalize_linear import NormalizeLinear

from.normalize_batchnorm2d import NormalizeBatchNorm2d


DefaultNormalizeIntXModule = (nn.Conv2d, nn.Linear, nn.ConvTranspose2d, NormalizeConvBN2d,
                              nn.Conv1d, NormalizeConvBN1d, nn.BatchNorm2d, nn.GRU, nn.LSTM)
SupportNormalizeTorchModules = [nn.Conv2d, nn.Linear, nn.ConvTranspose2d, NormalizeConvBN2d,
                                nn.Conv1d, NormalizeConvBN1d, nn.BatchNorm2d, nn.GRU, nn.LSTM, nn.Embedding]
SupportNormalizeIntXModules = (NormalizeConv2d, NormalizeLinear, NormalizeConvTranspose2d, NormalizeConvBN2d,
                               NormalizeConv1d, NormalizeConvBN1d, NormalizeBatchNorm2d, NormalizeFastGRU, NormalizeFastLSTM, NormalizeLayerNorm)
