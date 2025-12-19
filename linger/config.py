import torch
from enum import Enum
from .utils import *
import os
import yaml

# class Singleton:
#     _instance = None  # 保存唯一实例
#     def __new__(cls, *args, **kwargs):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#         return cls._instance
#     def __init__(self):
#         # 只初始化一次
#         if not hasattr(self, "_initialized"):
#             self._initialized = True


def str_to_dtype(s: str):
    """将字符串转回 torch.dtype，如 'torch.float32' -> torch.float32"""
    if not isinstance(s, str) or not s.startswith("torch."):
        raise ValueError(f"Invalid dtype string: {s}")
    attr_name = s[6:]  # 去掉 "torch."
    if hasattr(torch, attr_name):
        return getattr(torch, attr_name)
    else:
        raise ValueError(f"Unknown torch dtype: {s}")

class QuantInfo():
    def __init__(self):
        self.weight_bits            = 8
        self.activate_bits          = 8
        self.bias_bits              = 32
        self.a_strategy             = QuantStrategy.RANGE_MEAN
        self.w_strategy             = QuantStrategy.RANGE_MEAN
        self.is_symmetry            = True
        self.is_perchannel          = False
        self.round_mode             = QuantMode.floor_add
        self.activation_type        = ActivationType.none
        self.qat_method             = QatMethod.MOM
        self.w_calibrate_name       = "abs_max"
        self.a_calibrate_name       = "top_10"
    
    def to_dict(self):
        return self.__dict__
    
    def to_save_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith('_') or k.startswith('to'):
                continue
            # 处理嵌套配置（如 quant_info）
            if isinstance(v, Enum):
                result[k] = v.name
            else:
                result[k] = v
        return result

    def _update_from_dict(self, data: dict):
        """从字典更新当前配置（支持Enum）"""
        for key, value in data.items():
            if not hasattr(self, key):
                continue  # 忽略未知字段
            
            current = getattr(self, key)
            
            # 如果当前是 torch.dtype，且 value 是字符串 → 转换
            if isinstance(current, Enum) and isinstance(value, str):
                enum_cls = type(current)
                if hasattr(enum_cls, value):
                    setattr(self, key, getattr(enum_cls, value))
                else:
                    print(f"⚠️ 无效枚举值: {value}")
            else:
                setattr(self, key, value)




class ClampInfo():
    def __init__(self):
        self.clamp_weight_value     = None
        self.clamp_bias_value       = None
        self.clamp_activation_value = 8
        self.clamp_factor_value     = 7  # for dyn clip
    def to_dict(self):
        return self.__dict__

    def to_save_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith('_') or k.startswith('to'):
                continue
            # 处理嵌套配置（如 quant_info）
            if isinstance(v, Enum):
                result[k] = v.name
            else:
                result[k] = v
        return result

class QuantConfig(Singleton):
    open_quant   = True
    quant_method = FakeQuantMethod.NATIVE
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype        = torch.float32
    seed         = 42
    calibration  = False
    platform     = PlatForm.venusA

    clamp_info   = ClampInfo()
    quant_info   = QuantInfo()

    @classmethod
    def _load_from_yaml(cls, config_path: str):
        """
        从 YAML 文件加载配置，并覆盖当前实例的属性
        支持嵌套字段(如 quant_info.weight_bits)
        """
        if not os.path.exists(config_path):
            raise ValueError(f"配置文件 {config_path} 不存在")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except:
            raise ValueError(f"加载配置失败: {config_path}")

        # 设置属性
        cls._update_from_dict(config_data)

    @classmethod
    def _save_to_yaml(cls, config_path: str):
        """
        将当前配置保存到 YAML 文件
        """
        config_dict = cls._to_save_dict()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, allow_unicode=True)

    # 因为config写在了class里，没有通过__init__函数初始化，这里获得__dict__时需要cls
    @classmethod
    def _to_save_dict(cls):
        """将整个配置转换为字典，包括嵌套对象"""
        result = {}
        for k, v in cls.__dict__.items():
            if k.startswith('_'):
                continue
            # 处理嵌套配置（如 quant_info）
            if isinstance(v, (QuantInfo, ClampInfo)):
                result[k] = v.to_save_dict()
            # 处理 torch.dtype
            elif isinstance(v, torch.dtype):
                result[k] = str(v)
            elif isinstance(v, torch.device):
                result[k] = str(v)
            # 处理 Enum（如果你有）
            elif isinstance(v, Enum):
                result[k] = v.name
            else:
                result[k] = v
        return result

    @classmethod
    def _update_from_dict(cls, data: dict):
        """从字典更新当前配置（支持 dtype 和 Enum）"""
        for key, value in data.items():
            if not hasattr(cls, key):
                continue  # 忽略未知字段
            
            current = getattr(cls, key)
            
            # 如果当前是 torch.dtype，且 value 是字符串 → 转换
            if isinstance(value, str) and value.startswith("torch."):
                try:
                    setattr(cls, key, str_to_dtype(value))
                except ValueError as e:
                    print(f"⚠️ 跳过无效 dtype: {e}")
            
            elif isinstance(current, torch.device) and isinstance(value, str):
                try:
                    setattr(cls, key, torch.device(value))
                except Exception as e:
                    print(f"⚠️ 无效 device: {value}, error: {e}")
            # 如果当前是 Enum，且 value 是字符串 → 转换
            elif isinstance(current, Enum) and isinstance(value, str):
                enum_cls = type(current)
                if hasattr(enum_cls, value):
                    setattr(cls, key, getattr(enum_cls, value))
                else:
                    print(f"⚠️ 无效枚举值: {value}")
            
            # 如果是嵌套对象（如 quant_info），在_set_nested_attr函数里处理，当前步跳过
            elif isinstance(current, ClampInfo):
                if isinstance(value, dict):
                    current.__dict__.update(value)
            elif isinstance(current, QuantInfo):
                if isinstance(value, dict): # 自定义更新QuantInfo，支持enum类型
                    current._update_from_dict(value)
            
            # 其他普通字段直接赋值
            else:
                setattr(cls, key, value)
    
QUANT_CONFIGS = QuantConfig()
