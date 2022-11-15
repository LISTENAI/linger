import os
import sys

cur_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cur_dir, "lib"))

from .__version import __version__, version_info
from .config import *
from .conv_bn_fuser import EmptyBatchNorm, FuseBNIntoConv, FuseConvBNAheadRelu
from .dumper import Dumper
from .initialize import (DefaultQuantIntXOP, disable_quant, init, quant_module,
                         quant_module_by_type, quant_tensor)
from .layer_normalizer import (disable_normalize, normalize_layers,
                               normalize_module)
from .layer_tracer import trace_layers
from .modules import (NormalizeBatchNorm2d, NormalizeConv1d, NormalizeConv2d,
                      NormalizeConvBN1d, NormalizeConvBN2d,
                      NormalizeConvTranspose2d, NormalizeEmbedding,
                      NormalizeFastGRU, NormalizeFastLSTM, NormalizeLayerNorm,
                      NormalizeLinear)
from .onnx import parser_dequant
from .ops import *
from .quant import *
from .tools import fix_dequant, onnx_quant, wb_analyse
from .utils import *

name = "linger"
