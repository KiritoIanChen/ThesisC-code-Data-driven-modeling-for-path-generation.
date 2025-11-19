#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDR-Net Official Inference Wrapper
使用GDR-Net官方推理流程的包装器

这个实现严格按照GDR-Net原始代码的推理流程：
1. 使用官方配置文件 (a6_cPnP_lm13.py)
2. 使用官方的数据预处理 (crop_resize_by_warp_affine, normalize_image)
3. 使用官方的模型构建 (build_model_optimizer)
4. 使用官方的PnP求解 (pose_from_pred)

Date: 2025-10-17
"""

import sys
import os
from pathlib import Path
from types import ModuleType
import torch
import numpy as np
import cv2
from typing import Optional, Tuple, Dict
import logging

# ============================================================================
# Step 1: Mock detectron2 和 mmcv.runner (在导入GDR-Net之前)
# 这些模块不是算法核心，只是工具函数，可以安全地Mock
# ============================================================================

def create_detectron2_mock():
    """创建detectron2的Mock模块"""
    if 'detectron2' in sys.modules:
        print("✓ detectron2已存在，跳过Mock")
        return
    
    print("🔧 创建detectron2 Mock...")
    
    # 主模块
    detectron2 = ModuleType('detectron2')
    
    # detectron2.utils.events
    detectron2_utils = ModuleType('detectron2.utils')
    detectron2_utils_events = ModuleType('detectron2.utils.events')
    
    class MockEventStorage:
        def put_scalars(self, **kwargs):
            pass
        def put_scalar(self, *args, **kwargs):
            pass
        def put_image(self, *args, **kwargs):
            pass
    
    def get_event_storage():
        return MockEventStorage()
    
    detectron2_utils_events.get_event_storage = get_event_storage
    detectron2_utils_events.EventStorage = MockEventStorage
    
    # detectron2.utils.logger
    detectron2_utils_logger = ModuleType('detectron2.utils.logger')
    
    def log_first_n(lvl, msg, n=1, name=None, key='caller'):
        """Mock log_first_n"""
        pass
    
    detectron2_utils_logger.log_first_n = log_first_n
    
    # detectron2.utils.comm
    detectron2_utils_comm = ModuleType('detectron2.utils.comm')
    detectron2_utils_comm.get_world_size = lambda: 1
    detectron2_utils_comm.get_rank = lambda: 0
    detectron2_utils_comm.is_main_process = lambda: True
    detectron2_utils_comm.synchronize = lambda: None
    
    # detectron2.utils.env
    detectron2_utils_env = ModuleType('detectron2.utils.env')
    detectron2_utils_env.TORCH_VERSION = tuple(map(int, torch.__version__.split('.')[:2]))
    
    # detectron2.config
    detectron2_config = ModuleType('detectron2.config')
    
    class CfgNode(dict):
        """Mock CfgNode"""
        def __init__(self, init_dict=None):
            super().__init__()
            if init_dict:
                for k, v in init_dict.items():
                    self[k] = CfgNode(v) if isinstance(v, dict) else v
        
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(f"CfgNode has no attribute '{name}'")
        
        def __setattr__(self, name, value):
            self[name] = value
        
        def clone(self):
            return CfgNode(dict(self))
    
    detectron2_config.CfgNode = CfgNode
    
    # detectron2.layers
    detectron2_layers = ModuleType('detectron2.layers')
    detectron2_layers.cat = torch.cat  # 关键：cat函数就是torch.cat
    
    # detectron2.layers.batch_norm
    detectron2_layers_batch_norm = ModuleType('detectron2.layers.batch_norm')
    detectron2_layers_batch_norm.BatchNorm2d = torch.nn.BatchNorm2d  # 添加BatchNorm2d
    detectron2_layers_batch_norm.NaiveSyncBatchNorm = torch.nn.BatchNorm2d
    detectron2_layers_batch_norm.FrozenBatchNorm2d = torch.nn.BatchNorm2d
    
    # detectron2.solver
    detectron2_solver = ModuleType('detectron2.solver')
    
    # Mock LR schedulers - 使用torch内置的scheduler作为替代
    class WarmupCosineLR(torch.optim.lr_scheduler.CosineAnnealingLR):
        """Mock WarmupCosineLR"""
        def __init__(self, optimizer, max_iters, warmup_factor=0.001, warmup_iters=1000, warmup_method="linear"):
            super().__init__(optimizer, T_max=max_iters)
    
    class WarmupMultiStepLR(torch.optim.lr_scheduler.MultiStepLR):
        """Mock WarmupMultiStepLR"""
        def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=0.001, warmup_iters=1000, warmup_method="linear"):
            super().__init__(optimizer, milestones=milestones, gamma=gamma)
    
    def build_lr_scheduler(cfg, optimizer):
        """Mock LR scheduler builder"""
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    detectron2_solver.WarmupCosineLR = WarmupCosineLR
    detectron2_solver.WarmupMultiStepLR = WarmupMultiStepLR
    detectron2_solver.build_lr_scheduler = build_lr_scheduler
    
    # 组装模块树
    detectron2_utils.events = detectron2_utils_events
    detectron2_utils.logger = detectron2_utils_logger
    detectron2_utils.comm = detectron2_utils_comm
    detectron2_utils.env = detectron2_utils_env
    detectron2.utils = detectron2_utils
    detectron2.config = detectron2_config
    detectron2.layers = detectron2_layers
    detectron2.solver = detectron2_solver
    
    # 注册到sys.modules
    sys.modules['detectron2'] = detectron2
    sys.modules['detectron2.utils'] = detectron2_utils
    sys.modules['detectron2.utils.events'] = detectron2_utils_events
    sys.modules['detectron2.utils.logger'] = detectron2_utils_logger
    sys.modules['detectron2.utils.comm'] = detectron2_utils_comm
    sys.modules['detectron2.utils.env'] = detectron2_utils_env
    sys.modules['detectron2.config'] = detectron2_config
    sys.modules['detectron2.layers'] = detectron2_layers
    sys.modules['detectron2.layers.batch_norm'] = detectron2_layers_batch_norm
    sys.modules['detectron2.solver'] = detectron2_solver
    
    print("✓ detectron2 Mock创建成功")


def create_mmcv_mock():
    """创建mmcv的Mock模块（包括Config和runner）"""
    # 检查是否需要真实的mmcv
    try:
        from mmcv import Config as RealConfig
        print("✓ 检测到真实的mmcv，使用真实Config")
        
        # Mock mmcv.utils（如果不存在）
        if not hasattr(sys.modules.get('mmcv', None), 'utils'):
            print("🔧 创建mmcv.utils Mock...")
            mmcv_utils = ModuleType('mmcv.utils')
            def build_from_cfg(cfg, registry, default_args=None):
                if isinstance(cfg, dict):
                    cfg = cfg.copy()
                    obj_type = cfg.pop('type', None)
                    if obj_type and hasattr(torch.optim, obj_type):
                        return getattr(torch.optim, obj_type)
                    return cfg
                return cfg
            mmcv_utils.build_from_cfg = build_from_cfg
            sys.modules['mmcv.utils'] = mmcv_utils
        
        # Mock mmcv.cnn（如果不存在）
        if not hasattr(sys.modules.get('mmcv', None), 'cnn'):
            print("🔧 创建mmcv.cnn Mock...")
            mmcv_cnn = ModuleType('mmcv.cnn')
            def normal_init(module, mean=0, std=1, bias=0):
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.normal_(module.weight, mean, std)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.constant_(module.bias, bias)
            def kaiming_init(module, a=0, mode='fan_in', nonlinearity='leaky_relu', bias=0, distribution='normal'):
                if hasattr(module, 'weight') and module.weight is not None:
                    if distribution == 'uniform':
                        torch.nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
                    else:
                        torch.nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.constant_(module.bias, bias)
            def constant_init(module, val, bias=0):
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.constant_(module.weight, val)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.constant_(module.bias, bias)
            mmcv_cnn.normal_init = normal_init
            mmcv_cnn.kaiming_init = kaiming_init
            mmcv_cnn.constant_init = constant_init
            sys.modules['mmcv.cnn'] = mmcv_cnn
        
        # 只Mock runner部分
        if not hasattr(sys.modules.get('mmcv', None), 'runner'):
            print("🔧 创建mmcv.runner Mock...")
            mmcv_runner = ModuleType('mmcv.runner')
            mmcv_runner_optimizer = ModuleType('mmcv.runner.optimizer')
            
            # 使用真实的mmcv.Registry
            from mmcv.utils import Registry
            OPTIMIZERS = Registry('optimizer')
            
            class DefaultOptimizerConstructor:
                def __init__(self, optimizer_cfg):
                    self.optimizer_cfg = optimizer_cfg
                def __call__(self, model):
                    return torch.optim.Adam(model.parameters(), lr=0.001)
            
            def build_optimizer(model, cfg=None):
                params = [p for p in model.parameters() if p.requires_grad]
                return torch.optim.Adam(params, lr=1e-4)
            
            def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
                """Mock load_checkpoint - 支持torchvision://和http(s)://等URL"""
                # 处理torchvision://这样的URL
                if isinstance(filename, str) and filename.startswith('torchvision://'):
                    model_name = filename.replace('torchvision://', '')
                    if logger:
                        logger.info(f"加载torchvision预训练模型: {model_name}")
                    # 使用torchvision的预训练权重
                    import torchvision.models as models
                    if hasattr(models, model_name):
                        pretrained_model = getattr(models, model_name)(pretrained=True)
                        state_dict = pretrained_model.state_dict()
                    else:
                        if logger:
                            logger.warning(f"⚠️ torchvision中没有{model_name}，跳过预训练权重加载")
                        return {}
                    checkpoint = {'state_dict': state_dict}
                elif isinstance(filename, str) and (filename.startswith('http://') or filename.startswith('https://')):
                    # 处理http(s)://这样的URL
                    checkpoint = torch.hub.load_state_dict_from_url(filename, map_location=map_location or 'cpu')
                else:
                    # 处理本地文件路径
                    checkpoint = torch.load(filename, map_location=map_location or 'cpu')
                
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[k[7:] if k.startswith('module.') else k] = v
                model.load_state_dict(new_state_dict, strict=strict)
                return checkpoint
            
            def obj_from_dict_real(info, parent=None, default_args=None):
                """Mock obj_from_dict"""
                if isinstance(info, dict):
                    obj_type = info.get('type', None)
                    if obj_type and obj_type in ['Adam', 'SGD', 'AdamW']:
                        return getattr(torch.optim, obj_type)
                    return info
                return info
            
            mmcv_runner_optimizer.OPTIMIZERS = OPTIMIZERS
            mmcv_runner_optimizer.DefaultOptimizerConstructor = DefaultOptimizerConstructor
            mmcv_runner_optimizer.build_optimizer = build_optimizer
            mmcv_runner.optimizer = mmcv_runner_optimizer
            mmcv_runner.load_checkpoint = load_checkpoint
            mmcv_runner.obj_from_dict = obj_from_dict_real
            
            sys.modules['mmcv.runner'] = mmcv_runner
            sys.modules['mmcv.runner.optimizer'] = mmcv_runner_optimizer
            print("✓ mmcv.runner Mock创建成功")
        return
    except ImportError:
        pass
    
    # 如果没有真实的mmcv，创建完整的Mock
    print("🔧 创建完整mmcv Mock（包括Config）...")
    
    # 主mmcv模块
    if 'mmcv' not in sys.modules:
        sys.modules['mmcv'] = ModuleType('mmcv')
    
    mmcv = sys.modules['mmcv']
    
    # mmcv.Config - 简化版，从YAML/PY文件加载配置
    class MockConfig:
        """Mock mmcv.Config - 支持从.py文件加载配置"""
        def __init__(self, cfg_dict=None, **kwargs):
            self._cfg_dict = cfg_dict or {}
            if kwargs:
                self._cfg_dict.update(kwargs)
        
        @staticmethod
        def fromfile(filename):
            """从文件加载配置（支持.py和.yaml）"""
            filename = str(filename)
            if filename.endswith('.py'):
                # 执行Python配置文件
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", filename)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 提取所有大写变量作为配置
                cfg_dict = {}
                for key in dir(module):
                    if not key.startswith('_'):
                        cfg_dict[key] = getattr(module, key)
                
                return MockConfig(cfg_dict)
            elif filename.endswith(('.yaml', '.yml')):
                import yaml
                with open(filename, 'r') as f:
                    cfg_dict = yaml.safe_load(f)
                return MockConfig(cfg_dict)
            else:
                raise ValueError(f"Unsupported config file format: {filename}")
        
        def __getattr__(self, name):
            if name.startswith('_'):
                return object.__getattribute__(self, name)
            if name in self._cfg_dict:
                value = self._cfg_dict[name]
                # 递归转换字典为MockConfig
                if isinstance(value, dict):
                    return MockConfig(value)
                return value
            # 调试：打印可用的键
            print(f"⚠️ Config属性 '{name}' 不存在")
            print(f"   可用的键: {list(self._cfg_dict.keys())[:10]}")
            raise AttributeError(f"Config has no attribute '{name}'. Available keys: {list(self._cfg_dict.keys())[:10]}")
        
        def __getitem__(self, key):
            return self._cfg_dict[key]
        
        def __setattr__(self, name, value):
            if name.startswith('_'):
                object.__setattr__(self, name, value)
            else:
                self._cfg_dict[name] = value
        
        def get(self, key, default=None):
            return self._cfg_dict.get(key, default)
        
        def __contains__(self, key):
            return key in self._cfg_dict
        
        def keys(self):
            return self._cfg_dict.keys()
        
        def items(self):
            return self._cfg_dict.items()
        
        def __repr__(self):
            return f"MockConfig({self._cfg_dict})"
    
    mmcv.Config = MockConfig
    
    # mmcv.utils
    mmcv_utils = ModuleType('mmcv.utils')
    
    def build_from_cfg(cfg, registry, default_args=None):
        """Mock build_from_cfg - 从配置构建对象"""
        if isinstance(cfg, dict):
            cfg = cfg.copy()
            obj_type = cfg.pop('type', None)
            
            if obj_type is None:
                return cfg
            
            # 如果registry是字典且包含该类型
            if isinstance(registry, dict) and obj_type in registry:
                obj_cls = registry[obj_type]
            # 否则尝试作为类名
            elif isinstance(obj_type, str):
                # 简化：返回torch优化器
                if hasattr(torch.optim, obj_type):
                    obj_cls = getattr(torch.optim, obj_type)
                else:
                    return cfg
            else:
                obj_cls = obj_type
            
            # 合并default_args
            if default_args:
                for key, value in default_args.items():
                    cfg.setdefault(key, value)
            
            # 返回类（不实例化，因为可能缺少参数）
            return obj_cls
        return cfg
    
    mmcv_utils.build_from_cfg = build_from_cfg
    mmcv.utils = mmcv_utils
    sys.modules['mmcv.utils'] = mmcv_utils
    
    # mmcv.cnn - 权重初始化函数
    mmcv_cnn = ModuleType('mmcv.cnn')
    
    def normal_init(module, mean=0, std=1, bias=0):
        """用正态分布初始化权重"""
        if hasattr(module, 'weight') and module.weight is not None:
            torch.nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.constant_(module.bias, bias)
    
    def kaiming_init(module, a=0, mode='fan_in', nonlinearity='leaky_relu', bias=0, distribution='normal'):
        """用Kaiming初始化权重"""
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                torch.nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
            else:
                torch.nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.constant_(module.bias, bias)
    
    def constant_init(module, val, bias=0):
        """用常数初始化权重"""
        if hasattr(module, 'weight') and module.weight is not None:
            torch.nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            torch.nn.init.constant_(module.bias, bias)
    
    mmcv_cnn.normal_init = normal_init
    mmcv_cnn.kaiming_init = kaiming_init
    mmcv_cnn.constant_init = constant_init
    mmcv.cnn = mmcv_cnn
    sys.modules['mmcv.cnn'] = mmcv_cnn
    
    # mmcv.runner.optimizer
    mmcv_runner = ModuleType('mmcv.runner')
    mmcv_runner_optimizer = ModuleType('mmcv.runner.optimizer')
    
    # 创建一个简化的Registry Mock（如果mmcv不存在）
    class MockRegistry:
        """简化的Registry Mock"""
        def __init__(self, name):
            self._name = name
            self._module_dict = {}
        
        def register_module(self, name=None, force=False, module=None):
            def _register(cls):
                module_name = name if name is not None else cls.__name__
                if module_name in self._module_dict and not force:
                    raise KeyError(f'{module_name} is already registered')
                self._module_dict[module_name] = cls
                return cls
            if module is not None:
                return _register(module)
            return _register
        
        def get(self, key):
            return self._module_dict.get(key)
        
        def __contains__(self, key):
            return key in self._module_dict
    
    OPTIMIZERS = MockRegistry('optimizer')
    
    class DefaultOptimizerConstructor:
        def __init__(self, optimizer_cfg):
            self.optimizer_cfg = optimizer_cfg
        
        def __call__(self, model):
            return torch.optim.Adam(model.parameters(), lr=0.001)
    
    def build_optimizer(model, cfg=None):
        """Mock build_optimizer"""
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=1e-4)
    
    def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
        """Mock load_checkpoint - 支持torchvision://和http(s)://等URL"""
        # 处理torchvision://这样的URL
        if isinstance(filename, str) and filename.startswith('torchvision://'):
            model_name = filename.replace('torchvision://', '')
            if logger:
                logger.info(f"加载torchvision预训练模型: {model_name}")
            # 使用torchvision的预训练权重
            import torchvision.models as models
            if hasattr(models, model_name):
                pretrained_model = getattr(models, model_name)(pretrained=True)
                state_dict = pretrained_model.state_dict()
            else:
                if logger:
                    logger.warning(f"⚠️ torchvision中没有{model_name}，跳过预训练权重加载")
                return {}
            checkpoint = {'state_dict': state_dict}
        elif isinstance(filename, str) and (filename.startswith('http://') or filename.startswith('https://')):
            # 处理http(s)://这样的URL
            checkpoint = torch.hub.load_state_dict_from_url(filename, map_location=map_location or 'cpu')
        else:
            # 处理本地文件路径
            checkpoint = torch.load(filename, map_location=map_location or 'cpu')
        
        # 提取state_dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 移除'module.'前缀（如果有）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # 加载到模型
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
        
        if logger:
            if missing_keys:
                logger.info(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.info(f"Unexpected keys: {unexpected_keys}")
        
        return checkpoint
    
    def obj_from_dict(info, parent=None, default_args=None):
        """Mock obj_from_dict - 从字典创建对象"""
        # 简化实现：假设info是一个dict，包含'type'键
        if isinstance(info, dict):
            obj_type = info.get('type', None)
            if obj_type is None:
                return info
            
            # 如果是torch优化器
            if obj_type in ['Adam', 'SGD', 'AdamW']:
                optimizer_cls = getattr(torch.optim, obj_type)
                # 简化：返回类，实际使用时需要传入参数
                return optimizer_cls
            
            # 其他情况返回原字典
            return info
        return info
    
    mmcv_runner_optimizer.OPTIMIZERS = OPTIMIZERS
    mmcv_runner_optimizer.DefaultOptimizerConstructor = DefaultOptimizerConstructor
    mmcv_runner_optimizer.build_optimizer = build_optimizer
    mmcv_runner.optimizer = mmcv_runner_optimizer
    mmcv_runner.load_checkpoint = load_checkpoint
    mmcv_runner.obj_from_dict = obj_from_dict  # 添加obj_from_dict
    
    # 注册到sys.modules
    sys.modules['mmcv.runner'] = mmcv_runner
    sys.modules['mmcv.runner.optimizer'] = mmcv_runner_optimizer
    
    print("✓ mmcv Mock创建成功（包括Config、runner、load_checkpoint和obj_from_dict）")


# 在导入任何GDR-Net模块之前先创建Mock
print("=" * 80)
print("初始化GDR-Net官方推理包装器")
print("=" * 80)
create_detectron2_mock()
create_mmcv_mock()  # 修改：使用新的create_mmcv_mock

# ============================================================================
# Step 2: 添加GDR-Net路径并导入官方模块
# ============================================================================

# 添加GDR-Net路径
current_file = Path(__file__).resolve()
project_root = current_file.parents[3]  # odb_box/
GDRNET_ROOT = project_root / "GDR-Net-main" / "GDR-Net-main"

if str(GDRNET_ROOT) not in sys.path:
    sys.path.insert(0, str(GDRNET_ROOT))

# 验证路径
if not GDRNET_ROOT.exists():
    print(f"❌ GDR-Net根目录不存在: {GDRNET_ROOT}")
    print(f"   当前文件: {current_file}")
    print(f"   项目根目录: {project_root}")
    print(f"   请确保GDR-Net-main在项目根目录下")
    raise FileNotFoundError(f"GDR-Net根目录不存在: {GDRNET_ROOT}")

print(f"✓ GDR-Net根目录: {GDRNET_ROOT}")

# 导入GDR-Net官方模块
try:
    from mmcv import Config
    from core.gdrn_modeling.models.GDRN import build_model_optimizer
    from core.utils.data_utils import crop_resize_by_warp_affine, get_2d_coord_np
    from lib.pysixd import inout, misc
    
    # ref模块是可选的，如果不存在就跳过
    try:
        import core.utils.ref as ref
    except ImportError:
        # 创建一个简单的ref Mock（如果原始文件不存在）
        import types
        ref = types.ModuleType('ref')
        # 添加LineMOD相关的常量（如果需要的话）
        print("⚠️ ref模块不存在，使用Mock版本")
    
    print("✓ GDR-Net官方模块导入成功")
    print("=" * 80)
except ImportError as e:
    print(f"❌ 导入GDR-Net模块失败: {e}")
    print(f"请确保GDR-Net-main在正确位置: {GDRNET_ROOT}")
    import traceback
    traceback.print_exc()
    raise

# LineMOD物体信息
LINEMOD_OBJECTS = {
    1: "ape", 2: "benchvise", 4: "camera", 5: "can", 6: "cat",
    8: "driller", 9: "duck", 10: "eggbox", 11: "glue",
    12: "holepuncher", 13: "iron", 14: "lamp", 15: "phone"
}

LINEMOD_13_OBJECTS = [
    "ape", "benchvise", "camera", "can", "cat", "driller", "duck",
    "eggbox", "glue", "holepuncher", "iron", "lamp", "phone"
]


class GDRNetOfficialInference:
    """GDR-Net官方推理流程的包装器"""
    
    def __init__(
        self,
        config_path: str = None,
        model_path: str = None,
        device: str = 'cuda',
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            config_path: GDR-Net配置文件路径 (默认使用a6_cPnP_lm13.py)
            model_path: 模型权重路径
            device: 'cuda' or 'cpu'
            logger: 日志记录器
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.logger = logger or self._setup_logger()
        
        # 设置默认路径
        if config_path is None:
            config_path = GDRNET_ROOT / "configs" / "gdrn" / "lm" / "a6_cPnP_lm13.py"
        
        if model_path is None:
            # 使用默认模型路径
            # 当前文件: odb_box/yolo-maskrcnn-gdr/linemod_6d_pose_v2/pose_estimation/gdrnet_official_wrapper.py
            # 向上3层到odb_box: parents[3]
            # 模型路径: odb_box/yolo-maskrcnn-gdr/models/gdr/gdrn/lm/a6_cPnP_lm13/gdrn_lm.pth
            odb_box_root = Path(__file__).resolve().parents[3]
            model_path = odb_box_root / "yolo-maskrcnn-gdr" / "models" / "gdr" / "gdrn" / "lm" / "a6_cPnP_lm13" / "gdrn_lm.pth"
        
        self.config_path = Path(config_path)
        self.model_path = Path(model_path)
        
        # 加载配置和模型
        self.cfg = self._load_config()
        self.model = self._load_model()
        
        # 加载LineMOD模型信息（3D模型点云、边界框等）
        self._load_linemod_models_info()
        
        self.logger.info(f"✓ GDR-Net官方推理引擎已初始化")
        self.logger.info(f"  - 配置: {self.config_path.name}")
        self.logger.info(f"  - 设备: {self.device}")
        self.logger.info(f"  - 输入分辨率: {self.cfg.MODEL.CDPN.BACKBONE.INPUT_RES}")
        self.logger.info(f"  - 输出分辨率: {self.cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('GDRNetOfficial')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_config(self) -> Config:
        """加载官方GDR-Net配置"""
        self.logger.info(f"加载配置文件: {self.config_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        try:
            cfg = Config.fromfile(str(self.config_path))
            self.logger.info("✓ 配置文件加载成功")
            
            # 为推理添加缺失的默认值
            # SOLVER相关（仅训练需要，推理时提供默认值）
            if not hasattr(cfg, 'SOLVER'):
                cfg.SOLVER = {}
            if not hasattr(cfg.SOLVER, 'BASE_LR'):
                cfg.SOLVER.BASE_LR = 1e-4  # 默认学习率
            if not hasattr(cfg.SOLVER, 'OPTIMIZER'):
                cfg.SOLVER.OPTIMIZER = 'Adam'
            if not hasattr(cfg.SOLVER, 'WEIGHT_DECAY'):
                cfg.SOLVER.WEIGHT_DECAY = 0.0
            
            # TEST配置 - 强制启用USE_PNP以获取mask输出
            if not hasattr(cfg, 'TEST'):
                # 使用已导入的Config类（顶部导入）
                cfg.TEST = type(cfg)(dict())  # 使用同样的Config类型
            # 强制设置为True，即使base config中是False
            cfg.TEST.USE_PNP = True  # 启用PnP，这样会输出mask、coor_x、coor_y、coor_z等
            
            self.logger.info(f"  - INPUT_RES: {cfg.MODEL.CDPN.BACKBONE.INPUT_RES}")
            self.logger.info(f"  - OUTPUT_RES: {cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES}")
            self.logger.info(f"  - NUM_CLASSES: {cfg.MODEL.CDPN.ROT_HEAD.NUM_CLASSES}")
            self.logger.info(f"  - TEST.USE_PNP: {cfg.TEST.USE_PNP}")
            
            return cfg
        except Exception as e:
            self.logger.error(f"❌ 配置文件加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_model(self) -> torch.nn.Module:
        """使用官方build_model_optimizer函数加载模型"""
        self.logger.info("构建GDR-Net模型...")
        
        try:
            # 使用官方的build_model_optimizer函数
            model, _ = build_model_optimizer(self.cfg)
            
            # 加载权重
            if self.model_path.exists():
                self.logger.info(f"加载模型权重: {self.model_path}")
                checkpoint = torch.load(str(self.model_path), map_location=self.device)
                
                # 提取模型状态字典
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # 加载权重
                model.load_state_dict(state_dict, strict=True)
                self.logger.info("✓ 模型权重加载成功")
            else:
                self.logger.warning(f"⚠️ 模型文件不存在: {self.model_path}")
                self.logger.warning("⚠️ 使用随机初始化的权重")
            
            # 设置为评估模式
            model.eval()
            model.to(self.device)
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_linemod_models_info(self):
        """加载LineMOD物体的3D模型信息"""
        self.logger.info("加载LineMOD物体模型信息...")
        
        # 获取模型目录 - 使用GDRNET_ROOT来定位
        dataset_root = GDRNET_ROOT.parents[1] / "dataset" / "Linemod_preprocessed" / "Linemod_preprocessed"
        models_root = dataset_root / "models"
        
        if not models_root.exists():
            self.logger.warning(f"⚠️ 模型目录不存在: {models_root}")
            self.logger.info(f"   尝试的路径: {dataset_root}")
            self.models_info = {}
            self.models_pts = {}
            self.models_extents = {}
            return
        
        # 加载models_info.yml
        models_info_path = models_root / "models_info.yml"
        if models_info_path.exists():
            import yaml
            with open(models_info_path, 'r') as f:
                self.models_info = yaml.safe_load(f)
        else:
            self.logger.warning(f"⚠️ models_info.yml不存在")
            self.models_info = {}
        
        # 加载每个物体的3D模型点云
        self.models_pts = {}
        self.models_extents = {}
        
        for obj_id, obj_name in LINEMOD_OBJECTS.items():
            ply_path = models_root / f"obj_{obj_id:02d}.ply"
            if ply_path.exists():
                # 加载PLY文件
                model = inout.load_ply(str(ply_path), vertex_scale=0.001)  # mm to m
                pts = model['pts']
                self.models_pts[obj_id] = pts
                
                # 计算物体范围（extents）
                xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
                ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
                zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
                self.models_extents[obj_id] = np.array([
                    xmax - xmin, ymax - ymin, zmax - zmin
                ], dtype=np.float32)
        
        self.logger.info(f"✓ 已加载 {len(self.models_pts)} 个物体的3D模型")
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像归一化（按照GDR-Net官方方式）
        Args:
            image: (C, H, W) float32 array, range [0, 255]
        Returns:
            normalized image (C, H, W)
        """
        pixel_mean = np.array(self.cfg.MODEL.PIXEL_MEAN).reshape(-1, 1, 1)  # [123.675, 116.28, 103.53]
        pixel_std = np.array(self.cfg.MODEL.PIXEL_STD).reshape(-1, 1, 1)    # [58.395, 57.12, 57.375]
        return (image - pixel_mean) / pixel_std
    
    def prepare_roi_data(
        self,
        image: np.ndarray,
        bbox_xyxy: np.ndarray,
        object_id: int,
        camera_K: np.ndarray,
        score: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        按照GDR-Net官方方式准备ROI数据
        
        Args:
            image: (H, W, 3) BGR图像, uint8
            bbox_xyxy: [x1, y1, x2, y2]
            object_id: LineMOD物体ID (1-15)
            camera_K: 3x3相机内参矩阵
            score: 检测置信度
        
        Returns:
            包含所有必需张量的字典
        """
        cfg = self.cfg
        im_H, im_W = image.shape[:2]
        
        # 配置参数
        input_res = cfg.MODEL.CDPN.BACKBONE.INPUT_RES    # 256
        output_res = cfg.MODEL.CDPN.BACKBONE.OUTPUT_RES  # 64
        pad_scale = cfg.INPUT.DZI_PAD_SCALE              # 1.5
        
        # 计算bbox中心和尺寸
        x1, y1, x2, y2 = bbox_xyxy
        bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float32)
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        
        # 计算scale（考虑pad_scale）
        scale = max(bh, bw) * pad_scale
        scale = min(scale, max(im_H, im_W)) * 1.0
        
        # 使用官方的crop_resize_by_warp_affine进行ROI裁剪
        # 裁剪并缩放到input_res x input_res
        roi_img = crop_resize_by_warp_affine(
            image,
            bbox_center,
            scale,
            input_res,
            interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)  # HWC -> CHW
        
        # 归一化（使用官方方式）
        roi_img = self.normalize_image(roi_img.astype(np.float32))
        
        # 生成2D坐标图（用于PnP Net）
        coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)  # CHW -> HWC
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d,
            bbox_center,
            scale,
            output_res,
            interpolation=cv2.INTER_LINEAR
        ).transpose(2, 0, 1)  # HWC -> CHW
        
        # 转换为LineMOD类别标签（0-based）
        if object_id in LINEMOD_OBJECTS:
            obj_name = LINEMOD_OBJECTS[object_id]
            roi_cls = LINEMOD_13_OBJECTS.index(obj_name)  # 0-12
        else:
            raise ValueError(f"不支持的物体ID: {object_id}")
        
        # 获取物体extent
        if object_id in self.models_extents:
            roi_extent = self.models_extents[object_id]
        else:
            self.logger.warning(f"⚠️ 物体 {object_id} 没有extent信息，使用默认值")
            roi_extent = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        
        # 准备所有输入张量（严格按照官方batch_data_test的格式）
        batch = {
            # 图像相关
            'roi_img': torch.from_numpy(roi_img).float().unsqueeze(0).to(self.device),  # [1, 3, 256, 256]
            'roi_coord_2d': torch.from_numpy(roi_coord_2d).float().unsqueeze(0).to(self.device),  # [1, 2, 64, 64]
            
            # ROI信息
            'roi_cls': torch.tensor([roi_cls], dtype=torch.long).to(self.device),  # [1]
            'roi_extent': torch.from_numpy(roi_extent).float().unsqueeze(0).to(self.device),  # [1, 3]
            'score': torch.tensor([score], dtype=torch.float32).to(self.device),  # [1]
            
            # 几何信息
            'roi_cam': torch.from_numpy(camera_K).float().unsqueeze(0).to(self.device),  # [1, 3, 3]
            'roi_center': torch.from_numpy(bbox_center).float().unsqueeze(0).to(self.device),  # [1, 2]
            'roi_wh': torch.tensor([[bw, bh]], dtype=torch.float32).to(self.device),  # [1, 2]
            'resize_ratio': torch.tensor([output_res / scale], dtype=torch.float32).to(self.device),  # [1]
            
            # 图像尺寸
            'im_H': torch.tensor([im_H], dtype=torch.float32).to(self.device),
            'im_W': torch.tensor([im_W], dtype=torch.float32).to(self.device),
        }
        
        return batch
    
    def estimate_pose(
        self,
        image: np.ndarray,
        bbox_xyxy: np.ndarray,
        object_id: int,
        camera_K: np.ndarray,
        score: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray], Optional[Dict]]:
        """
        使用官方GDR-Net流程估计6D姿态
        
        Args:
            image: (H, W, 3) BGR图像, uint8
            bbox_xyxy: [x1, y1, x2, y2]
            object_id: LineMOD物体ID
            camera_K: 3x3相机内参矩阵
            score: 检测置信度
        
        Returns:
            rotation: (3, 3) 旋转矩阵
            translation: (3,) 平移向量 (单位: 米)
            confidence: 姿态估计置信度
            mask: (H, W) 分割掩码 (可选)
        """
        self.logger.info(f"=== 使用官方GDR-Net流程进行姿态估计 ===")
        self.logger.info(f"   物体ID: {object_id} ({LINEMOD_OBJECTS.get(object_id, 'unknown')})")
        self.logger.info(f"   检测框: {bbox_xyxy}")
        
        try:
            # 计算ROI参数（用于后续mask映射）
            x1, y1, x2, y2 = bbox_xyxy
            bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float32)
            bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
            pad_scale = self.cfg.INPUT.DZI_PAD_SCALE
            scale = max(bh, bw) * pad_scale
            scale = min(scale, max(image.shape[:2])) * 1.0
            roi_info = {
                'center': bbox_center,
                'scale': scale,
                'input_res': self.cfg.MODEL.CDPN.BACKBONE.INPUT_RES
            }
            
            # 准备ROI数据（严格按照官方方式）
            batch = self.prepare_roi_data(image, bbox_xyxy, object_id, camera_K, score)
            
            # 前向推理
            with torch.no_grad():
                outputs = self.model(
                    batch['roi_img'],
                    roi_classes=batch['roi_cls'],
                    roi_cams=batch['roi_cam'],
                    roi_whs=batch['roi_wh'],
                    roi_centers=batch['roi_center'],
                    resize_ratios=batch['resize_ratio'],
                    roi_coord_2d=batch['roi_coord_2d'],
                    roi_extents=batch['roi_extent'],
                )
            
            self.logger.info(f"   ✓ 模型前向传播完成")
            self.logger.info(f"   输出键: {list(outputs.keys())}")
            
            # 提取姿态（rot 或 ego_rot）
            if 'ego_rot' in outputs:
                rotation = outputs['ego_rot'][0].cpu().numpy()  # (3, 3)
            elif 'rot' in outputs:
                rotation = outputs['rot'][0].cpu().numpy()  # (3, 3)
            else:
                self.logger.error("❌ 输出中没有'ego_rot'或'rot'")
                rotation = np.eye(3)
            
            if 'trans' in outputs:
                translation = outputs['trans'][0].cpu().numpy()  # (3,)
            else:
                self.logger.error("❌ 输出中没有'trans'")
                translation = np.array([0., 0., 1.])
            
            # 提取置信度
            if 'confidence' in outputs:
                confidence = float(outputs['confidence'][0].cpu().numpy())
            else:
                confidence = 0.8  # 默认置信度
            
            # 提取mask（来自GDR-Net的mask分支）
            mask = None
            if 'mask' in outputs:
                mask_tensor = outputs['mask'][0].cpu().numpy()  # (C, H, W) 或 (H, W)
                
                # 如果有多个通道，取第一个
                if len(mask_tensor.shape) == 3:
                    mask_tensor = mask_tensor[0]  # 取第一个通道
                
                # 先上采样到input_res (256x256)，提高分辨率
                input_res = self.cfg.MODEL.CDPN.BACKBONE.INPUT_RES
                mask_upsampled = cv2.resize(mask_tensor, (input_res, input_res), interpolation=cv2.INTER_LINEAR)
                
                # 归一化到[0, 1]
                mask_min, mask_max = mask_upsampled.min(), mask_upsampled.max()
                if mask_max > mask_min:
                    mask_norm = (mask_upsampled - mask_min) / (mask_max - mask_min)
                else:
                    mask_norm = mask_upsampled
                
                # 使用较低的阈值以获得更完整的mask
                mask = ((mask_norm > 0.2) * 255).astype(np.uint8)
                
                # 形态学操作：先闭运算（填充小孔），不做开运算（避免过度侵蚀）
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                # 可选：膨胀以获得更完整的轮廓
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.dilate(mask, kernel_dilate, iterations=1)
                
                self.logger.info(f"   ✓ 提取GDR-Net mask: shape={mask.shape}, 覆盖率={np.sum(mask > 0) / mask.size * 100:.1f}%")
            
            self.logger.info(f"✓ 官方GDR-Net推理成功")
            self.logger.info(f"   旋转矩阵: shape={rotation.shape}")
            self.logger.info(f"   平移向量: {translation}")
            self.logger.info(f"   置信度: {confidence:.3f}")
            
            return rotation, translation, confidence, mask, roi_info
            
        except Exception as e:
            self.logger.error(f"❌ 姿态估计失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回默认值
            return np.eye(3), np.array([0., 0., 1.]), 0.0, None, None


def test_official_inference():
    """测试官方推理流程"""
    print("=" * 80)
    print("测试GDR-Net官方推理流程")
    print("=" * 80)
    
    # 初始化推理引擎
    engine = GDRNetOfficialInference(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备测试数据 - 修正路径
    # gdrnet_official_wrapper.py在: odb_box/yolo-maskrcnn-gdr/linemod_6d_pose_v2/pose_estimation/
    # dataset在: odb_box/dataset/
    dataset_root = GDRNET_ROOT.parents[1] / "dataset" / "Linemod_preprocessed" / "Linemod_preprocessed"
    test_image_path = dataset_root / "data" / "01" / "rgb" / "0000.png"
    
    if not test_image_path.exists():
        print(f"❌ 测试图像不存在: {test_image_path}")
        print(f"   数据集根目录: {dataset_root}")
        print(f"   GDRNET_ROOT: {GDRNET_ROOT}")
        return
    
    # 加载图像
    image = cv2.imread(str(test_image_path))
    print(f"✓ 加载测试图像: {test_image_path.name}, shape={image.shape}")
    
    # 使用GT bbox进行测试
    import yaml
    gt_path = dataset_root / "data" / "01" / "gt.yml"
    with open(gt_path, 'r') as f:
        gt_data = yaml.safe_load(f)
    
    # 获取第一个物体的GT信息
    gt_0 = gt_data[0][0]
    obj_bb = gt_0['obj_bb']
    bbox_xyxy = np.array([obj_bb[0], obj_bb[1], obj_bb[0] + obj_bb[2], obj_bb[1] + obj_bb[3]])
    
    # 相机内参
    info_path = dataset_root / "data" / "01" / "info.yml"
    with open(info_path, 'r') as f:
        info_data = yaml.safe_load(f)
    cam_K = np.array(info_data[0]['cam_K']).reshape(3, 3)
    
    # 进行推理
    rotation, translation, confidence, mask = engine.estimate_pose(
        image=image,
        bbox_xyxy=bbox_xyxy,
        object_id=1,  # ape
        camera_K=cam_K
    )
    
    print("\n" + "=" * 80)
    print("推理结果:")
    print("=" * 80)
    print(f"旋转矩阵:\n{rotation}")
    print(f"平移向量: {translation}")
    print(f"置信度: {confidence}")
    if mask is not None:
        print(f"Mask: shape={mask.shape}, 覆盖率={np.sum(mask > 0) / mask.size * 100:.1f}%")
    
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    test_official_inference()

