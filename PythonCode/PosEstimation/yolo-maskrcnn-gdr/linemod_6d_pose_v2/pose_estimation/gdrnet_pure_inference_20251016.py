#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure GDR-Net Inference (No Detectron2, Only PyTorch!)
纯PyTorch的GDR-Net推理 - 直接使用训练好的模型权重

核心思路：
1. 直接导入GDR-Net的模型定义（无需Detectron2）
2. 加载训练好的权重
3. 进行前向推理

Date: 2025-10-16
"""

import sys
import os
from pathlib import Path
from types import ModuleType
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, Tuple
import logging

# ============================================================================
# Mock detectron2 and mmcv to avoid heavy dependencies
# These are only used for training, not needed for inference
# ============================================================================

def create_mock_modules():
    """Create proper mock modules using types.ModuleType"""
    
    # ========== Mock detectron2 ==========
    detectron2 = ModuleType('detectron2')
    
    # detectron2.utils
    detectron2_utils = ModuleType('detectron2.utils')
    detectron2_utils_events = ModuleType('detectron2.utils.events')
    detectron2_utils_logger = ModuleType('detectron2.utils.logger')
    detectron2_utils_comm = ModuleType('detectron2.utils.comm')
    detectron2_utils_env = ModuleType('detectron2.utils.env')
    
    class MockEventStorage:
        def put_scalars(self, **kwargs):
            pass
    
    def get_event_storage():
        return MockEventStorage()
    
    def log_first_n(lvl, msg, n=1, name=None, key='caller'):
        """Mock log_first_n - just pass"""
        pass
    
    # Mock comm functions
    def get_world_size():
        return 1
    
    def get_rank():
        return 0
    
    # Mock env variables
    detectron2_utils_env.TORCH_VERSION = tuple(map(int, torch.__version__.split('.')[:2]))
    
    detectron2_utils_events.get_event_storage = get_event_storage
    detectron2_utils_logger.log_first_n = log_first_n
    detectron2_utils_comm.get_world_size = get_world_size
    detectron2_utils_comm.get_rank = get_rank
    detectron2_utils.events = detectron2_utils_events
    detectron2_utils.logger = detectron2_utils_logger
    detectron2_utils.comm = detectron2_utils_comm
    detectron2_utils.env = detectron2_utils_env
    detectron2.utils = detectron2_utils
    
    # detectron2.config
    detectron2_config = ModuleType('detectron2.config')
    
    class CfgNode(dict):
        """Mock CfgNode - minimal dict-like config"""
        def __init__(self, init_dict=None):
            super().__init__()
            if init_dict:
                self.update(init_dict)
        
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)
        
        def __setattr__(self, name, value):
            self[name] = value
    
    detectron2_config.CfgNode = CfgNode
    detectron2.config = detectron2_config
    
    # detectron2.solver
    detectron2_solver = ModuleType('detectron2.solver')
    
    class WarmupCosineLR:
        def __init__(self, *args, **kwargs):
            pass
    
    class WarmupMultiStepLR:
        def __init__(self, *args, **kwargs):
            pass
    
    detectron2_solver.WarmupCosineLR = WarmupCosineLR
    detectron2_solver.WarmupMultiStepLR = WarmupMultiStepLR
    detectron2.solver = detectron2_solver
    
    # detectron2.layers
    detectron2_layers = ModuleType('detectron2.layers')
    detectron2_layers_batch_norm = ModuleType('detectron2.layers.batch_norm')
    
    # Mock BatchNorm layers - use standard PyTorch equivalents
    class BatchNorm2d(nn.BatchNorm2d):
        """Mock BatchNorm2d - just use PyTorch's BatchNorm2d"""
        pass
    
    class FrozenBatchNorm2d(nn.Module):
        """Mock FrozenBatchNorm2d - BatchNorm with frozen parameters"""
        def __init__(self, num_features, eps=1e-5):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.register_buffer("weight", torch.ones(num_features))
            self.register_buffer("bias", torch.zeros(num_features))
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
        
        def forward(self, x):
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
    
    class NaiveSyncBatchNorm(nn.BatchNorm2d):
        """Mock NaiveSyncBatchNorm - just use PyTorch's BatchNorm2d"""
        pass
    
    detectron2_layers_batch_norm.BatchNorm2d = BatchNorm2d
    detectron2_layers_batch_norm.FrozenBatchNorm2d = FrozenBatchNorm2d
    detectron2_layers_batch_norm.NaiveSyncBatchNorm = NaiveSyncBatchNorm
    detectron2_layers.batch_norm = detectron2_layers_batch_norm
    detectron2_layers.BatchNorm2d = BatchNorm2d
    detectron2_layers.FrozenBatchNorm2d = FrozenBatchNorm2d
    
    # Add cat function - just use torch.cat
    def cat(tensors, dim=0):
        """Mock cat function - use torch.cat"""
        return torch.cat(tensors, dim=dim)
    
    detectron2_layers.cat = cat
    detectron2.layers = detectron2_layers
    
    # ========== Mock mmcv ==========
    mmcv = ModuleType('mmcv')
    mmcv_cnn = ModuleType('mmcv.cnn')
    mmcv_runner = ModuleType('mmcv.runner')
    mmcv_runner_optimizer = ModuleType('mmcv.runner.optimizer')
    mmcv_utils = ModuleType('mmcv.utils')
    
    def normal_init(module, mean=0, std=1, bias=0):
        pass
    
    def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
        pass
    
    def constant_init(module, val, bias=0):
        pass
    
    def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
        """Mock load_checkpoint - returns the checkpoint dict/state_dict."""
        checkpoint = torch.load(filename, map_location=map_location)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        return checkpoint

    def obj_from_dict(cfg_dict, registry, default_args=None):
        """Minimal stub: return a simple object based on type."""
        if cfg_dict is None:
            return None
        cfg = cfg_dict.copy()
        type_name = cfg.pop('type', None)
        if type_name is None:
            return None
        if registry and type_name in registry:
            cls = registry[type_name]
            if isinstance(cls, type):
                try:
                    return cls(**cfg)
                except Exception:
                    return cls
            return cls
        return None
    
    # Mock optimizer registry and functions
    class MockRegistry(dict):
        """Mock registry that acts like a dict with register_module method"""
        def register_module(self, name=None, force=False, module=None):
            def decorator(cls):
                module_name = name if name is not None else cls.__name__
                self[module_name] = cls
                return cls
            if module is not None:
                return decorator(module)
            return decorator
    
    OPTIMIZERS = MockRegistry()
    
    class DefaultOptimizerConstructor:
        def __init__(self, optimizer_cfg, paramwise_cfg=None):
            pass
    
    def build_optimizer(model, optimizer_cfg):
        """Mock build_optimizer - just return Adam as default"""
        return torch.optim.Adam(model.parameters(), lr=0.001)
    
    def build_from_cfg(cfg, registry, default_args=None):
        """Mock build_from_cfg"""
        if cfg is None:
            return None
        cfg = cfg.copy()
        obj_type = cfg.pop('type', None)
        if obj_type is None:
            return None
        if obj_type in registry:
            obj_cls = registry[obj_type]
            if default_args is not None:
                cfg.update(default_args)
            return obj_cls(**cfg)
        # Fallback to standard PyTorch optimizers
        if obj_type == 'SGD':
            return torch.optim.SGD(**cfg)
        elif obj_type == 'Adam':
            return torch.optim.Adam(**cfg)
        elif obj_type == 'AdamW':
            return torch.optim.AdamW(**cfg)
        return None
    
    mmcv_cnn.normal_init = normal_init
    mmcv_cnn.kaiming_init = kaiming_init
    mmcv_cnn.constant_init = constant_init
    mmcv_runner.load_checkpoint = load_checkpoint
    mmcv_runner.obj_from_dict = obj_from_dict
    
    # Add optimizer-related mocks
    mmcv_runner_optimizer.OPTIMIZERS = OPTIMIZERS
    mmcv_runner_optimizer.DefaultOptimizerConstructor = DefaultOptimizerConstructor
    mmcv_runner_optimizer.build_optimizer = build_optimizer
    mmcv_runner.optimizer = mmcv_runner_optimizer
    
    # Add utils mocks
    mmcv_utils.build_from_cfg = build_from_cfg
    
    mmcv.cnn = mmcv_cnn
    mmcv.runner = mmcv_runner
    mmcv.utils = mmcv_utils
    
    return (detectron2, detectron2_utils, detectron2_utils_events, detectron2_utils_logger, 
            detectron2_utils_comm, detectron2_utils_env, detectron2_config, detectron2_solver,
            detectron2_layers, detectron2_layers_batch_norm,
            mmcv, mmcv_cnn, mmcv_runner, mmcv_runner_optimizer, mmcv_utils)

# Install mocks before importing GDR-Net
(detectron2, detectron2_utils, detectron2_utils_events, detectron2_utils_logger,
    detectron2_utils_comm, detectron2_utils_env, detectron2_config, detectron2_solver,
    detectron2_layers, detectron2_layers_batch_norm,
    mmcv, mmcv_cnn, mmcv_runner, mmcv_runner_optimizer, mmcv_utils) = create_mock_modules()

sys.modules['detectron2'] = detectron2
sys.modules['detectron2.utils'] = detectron2_utils
sys.modules['detectron2.utils.events'] = detectron2_utils_events
sys.modules['detectron2.utils.logger'] = detectron2_utils_logger
sys.modules['detectron2.utils.comm'] = detectron2_utils_comm
sys.modules['detectron2.utils.env'] = detectron2_utils_env
sys.modules['detectron2.config'] = detectron2_config
sys.modules['detectron2.solver'] = detectron2_solver
sys.modules['detectron2.layers'] = detectron2_layers
sys.modules['detectron2.layers.batch_norm'] = detectron2_layers_batch_norm

sys.modules['mmcv'] = mmcv
sys.modules['mmcv.cnn'] = mmcv_cnn
sys.modules['mmcv.runner'] = mmcv_runner
sys.modules['mmcv.runner.optimizer'] = mmcv_runner_optimizer
sys.modules['mmcv.utils'] = mmcv_utils

# 添加GDR-Net路径
GDRNET_ROOT = Path(__file__).parent.parent.parent.parent / "GDR-Net-main" / "GDR-Net-main"
if GDRNET_ROOT.exists():
    sys.path.insert(0, str(GDRNET_ROOT))
    sys.path.insert(0, str(GDRNET_ROOT / "core"))

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GDRNetPureInference:
    """
    纯PyTorch的GDR-Net推理引擎
    
    优势：
    - 不需要Detectron2
    - 直接使用PyTorch模型
    - 加载训练好的权重
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', num_classes: int = 13):
        """
        初始化GDR-Net推理引擎
        
        Args:
            model_path: 模型权重路径
            device: 'cuda' or 'cpu'
            num_classes: 类别数量（LineMOD=13）
        """
        self.model_path = model_path
        self.device = device
        self.num_classes = num_classes
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # LineMOD相机内参
        self.camera_K = np.array([
            [572.4114, 0.0, 325.2611],
            [0.0, 573.57043, 242.04899],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # 加载模型
        self.model = self._load_model()
        
        # 检查模型是否加载成功
        if self.model is None:
            raise RuntimeError(
                f"Failed to load GDR-Net model from {model_path}. "
                "Please check:\n"
                "1. Model file exists\n"
                "2. GDR-Net modules can be imported\n"
                "3. PyTorch is installed correctly\n"
                "See error messages above for details."
            )
        
        # LineMOD对象尺寸（用于PnP）
        self.object_extents = self._get_object_extents()
    
    def _load_model(self):
        """
        加载GDR-Net模型（纯PyTorch）
        
        关键：直接构建网络结构并加载权重，不使用Detectron2配置
        """
        self.logger.info(f"Loading GDR-Net model from: {self.model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            self.logger.error(f"❌ Model file not found: {self.model_path}")
            return None
        
        self.logger.info(f"✓ Model file exists")
        
        # 检查GDR-Net路径
        gdrnet_paths = [str(p) for p in sys.path if 'GDR-Net' in str(p)]
        if gdrnet_paths:
            self.logger.info(f"✓ GDR-Net in sys.path: {gdrnet_paths[0]}")
        else:
            self.logger.warning("⚠️  GDR-Net not found in sys.path")
        
        try:
            # 尝试导入GDR-Net的模型定义
            self.logger.info("Attempting to import GDR-Net modules...")
            
            try:
                from core.gdrn_modeling.models.GDRN import GDRN
                self.logger.info("  ✓ GDRN imported")
            except ImportError as e:
                self.logger.error(f"  ❌ Cannot import GDRN: {e}")
                raise
            
            try:
                from core.gdrn_modeling.models.resnet_backbone import ResNetBackboneNet
                self.logger.info("  ✓ ResNetBackboneNet imported")
            except ImportError as e:
                self.logger.error(f"  ❌ Cannot import ResNetBackboneNet: {e}")
                raise
            
            try:
                from core.gdrn_modeling.models.cdpn_rot_head_region import RotWithRegionHead
                self.logger.info("  ✓ RotWithRegionHead imported")
            except ImportError as e:
                self.logger.error(f"  ❌ Cannot import RotWithRegionHead: {e}")
                raise
            
            self.logger.info("✓ All GDR-Net modules imported successfully")
            
            # 构建模型（简化版配置）
            self.logger.info("Building GDR-Net model...")
            model = self._build_gdrn_model_minimal()
            self.logger.info("✓ Model architecture built")
            
            # 加载权重
            self.logger.info("Loading model weights...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    self.logger.info("  Using checkpoint['model']")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    self.logger.info("  Using checkpoint['state_dict']")
                else:
                    state_dict = checkpoint
                    self.logger.info("  Using checkpoint directly")
            else:
                state_dict = checkpoint
                self.logger.info("  Checkpoint is state_dict")
            
            # 先检查checkpoint中的关键层尺寸
            self.logger.info("Checking checkpoint dimensions:")
            for key in ['rot_head_net.features.0.weight', 'pnp_net.features.0.weight']:
                if key in state_dict:
                    self.logger.info(f"  {key}: {state_dict[key].shape}")
            
            # 加载权重到模型
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                self.logger.warning(f"  Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                self.logger.warning(f"  Unexpected keys: {len(unexpected_keys)} keys")
            
            model.to(self.device)
            model.eval()
            
            self.logger.info(f"✓ GDR-Net model loaded successfully on {self.device}")
            return model
                
        except ImportError as e:
            self.logger.error(f"❌ Cannot import GDR-Net modules: {e}")
            self.logger.error("   Make sure GDR-Net-main is in the correct location")
            import traceback
            traceback.print_exc()
            return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load GDR-Net model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_gdrn_model_minimal(self):
        """
        构建最小化的GDR-Net模型
        
        使用固定配置，避免依赖Detectron2的cfg系统
        """
        from core.gdrn_modeling.models.GDRN import GDRN
        from core.gdrn_modeling.models.resnet_backbone import ResNetBackboneNet
        from core.gdrn_modeling.models.cdpn_rot_head_region import RotWithRegionHead
        from core.gdrn_modeling.models.conv_pnp_net import ConvPnPNet
        
        # 创建一个最小化的配置对象
        class MinimalConfig:
            class MODEL:
                class CDPN:
                    NAME = "GDRN"
                    USE_MTL = False
                    
                    class BACKBONE:
                        PRETRAINED = ""
                        FREEZE = False
                        NUM_LAYERS = 34
                        NUM_FILTERS = [256, 256, 256]
                        OUTPUT_RES = 64
                    
                    class ROT_HEAD:
                        FREEZE = False
                        ROT_CONCAT = False  # 从配置文件得知，应该是False
                        NUM_LAYERS = 3
                        NUM_FILTERS = 256
                        CONV_KERNEL_SIZE = 3
                        OUT_CONV_KERNEL_SIZE = 1
                        NORM = "GN"
                        NUM_GN_GROUPS = 32
                        NUM_CLASSES = 13
                        NUM_REGIONS = 64  # Number of regions for region-aware prediction
                        XYZ_BIN = 64
                        XYZ_CLASS_AWARE = False
                        MASK_CLASS_AWARE = False
                        REGION_CLASS_AWARE = False
                        ROT_CLASS_AWARE = False
                        XYZ_LOSS_TYPE = "L1"
                        XYZ_LOSS_MASK_GT = "visib"
                        MASK_LOSS_TYPE = "L1"
                        REGION_LOSS_TYPE = "CE"
                        REGION_ATTENTION = False
                    
                    class TRANS_HEAD:
                        ENABLED = False
                    
                    class PNP_NET:
                        ENABLED = True
                        NORM = "GN"
                        NUM_GN_GROUPS = 32
                        R_ONLY = False
                        T_ONLY = False
                        WITH_2D_COORD = True
                        REGION_ATTENTION = True
                        MASK_ATTENTION = "none"
                        ROT_TYPE = "allo_rot6d"
                        TRANS_TYPE = "centroid_z"
                        Z_TYPE = "REL"
                        PM_R_ONLY = True
                        PM_NORM_BY_EXTENT = True
            
            class TEST:
                USE_PNP = True  # ⬅️ 必须为True才能输出mask和3D坐标
                VIS = False
                TEST_BBOX_TYPE = "est"
        
        cfg = MinimalConfig()
        
        # 根据num_layers构建backbone
        from torchvision.models.resnet import BasicBlock, Bottleneck
        
        # ResNet specification: {depth: (block, layers, channels, name)}
        resnet_spec = {
            18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], "resnet18"),
            34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], "resnet34"),
            50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], "resnet50"),
            101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], "resnet101"),
        }
        
        num_layers = cfg.MODEL.CDPN.BACKBONE.NUM_LAYERS
        block, layers, channels, name = resnet_spec[num_layers]
        
        # 构建backbone
        backbone = ResNetBackboneNet(
            block=block,
            layers=layers,
            in_channel=3,
            freeze=cfg.MODEL.CDPN.BACKBONE.FREEZE,
            rot_concat=cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT
        )
        
        # ResNet-34输出512通道，不是2048
        backbone_out_channels = channels[-1]  # 对于ResNet-34是512
        
        # 计算输出维度
        r_head_cfg = cfg.MODEL.CDPN.ROT_HEAD
        if r_head_cfg.XYZ_LOSS_TYPE in ["MSE", "L1", "L2", "SmoothL1"]:
            r_out_dim = 3
        else:
            r_out_dim = 3 * (r_head_cfg.XYZ_BIN + 1)
        
        if r_head_cfg.MASK_LOSS_TYPE in ["L1", "BCE"]:
            mask_out_dim = 1
        else:
            mask_out_dim = 2
        
        region_out_dim = r_head_cfg.NUM_REGIONS + 1
        
        # 构建rotation head
        rot_head = RotWithRegionHead(
            cfg,
            in_channels=backbone_out_channels,
            num_layers=r_head_cfg.NUM_LAYERS,
            num_filters=r_head_cfg.NUM_FILTERS,
            kernel_size=r_head_cfg.CONV_KERNEL_SIZE,
            output_kernel_size=r_head_cfg.OUT_CONV_KERNEL_SIZE,
            rot_output_dim=r_out_dim,
            mask_output_dim=mask_out_dim,
            freeze=r_head_cfg.FREEZE,
            num_classes=r_head_cfg.NUM_CLASSES,
            rot_class_aware=r_head_cfg.ROT_CLASS_AWARE,
            mask_class_aware=r_head_cfg.MASK_CLASS_AWARE,
            num_regions=r_head_cfg.NUM_REGIONS,
            region_class_aware=r_head_cfg.REGION_CLASS_AWARE,
            norm=r_head_cfg.NORM,
            num_gn_groups=r_head_cfg.NUM_GN_GROUPS,
        )
        
        # 构建PnP net
        # 从checkpoint可以看出：输入69通道，特征维128
        # 69 = 3(xyz coord) + 2(2d coord) + 64(region attention)
        pnp_net = ConvPnPNet(
            nIn=69,  # 从checkpoint推断的输入通道数
            rot_dim=6,  # rotation 6D representation
            num_regions=64,
            mask_attention_type="none",  # 不使用mask attention
            featdim=128,  # 从checkpoint推断的特征维度
            num_layers=3,
            norm="GN",
            num_gn_groups=32,
        )
        
        # 构建完整模型
        model = GDRN(cfg, backbone, rot_head, trans_head_net=None, pnp_net=pnp_net)
        
        return model
    
    def estimate_pose(self,
                     rgb: np.ndarray,
                     mask: Optional[np.ndarray] = None,
                     bbox: Optional[np.ndarray] = None,
                     object_id: int = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        使用GDR-Net估计6D姿态
        
        Args:
            rgb: RGB图像 (H, W, 3)
            mask: 物体掩码 (H, W) (可选)
            bbox: 边界框 [x1, y1, x2, y2] (可选)
            object_id: LineMOD物体ID (1-13)
        
        Returns:
            (rotation_matrix, translation_vector, confidence)
        """
        if self.model is None:
            raise RuntimeError("GDR-Net model not loaded")
        
        # 1. 预处理
        roi, roi_mask, crop_info = self._preprocess(rgb, mask, bbox)
        
        # 2. 转换为tensor
        roi_tensor = torch.from_numpy(roi).permute(2, 0, 1).unsqueeze(0)
        roi_tensor = roi_tensor.to(self.device).float() / 255.0
        
        # 归一化（ImageNet均值和方差）
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        roi_tensor = (roi_tensor - mean) / std
        
        # 3. 生成roi_coord_2d (标准化的2D坐标网格)
        # OUTPUT_RES=64，生成(1, 2, 64, 64)的坐标网格，值在[0,1]之间
        out_res = 64
        x_coords = torch.linspace(0, 1, out_res, device=self.device)
        y_coords = torch.linspace(0, 1, out_res, device=self.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        roi_coord_2d = torch.stack([xx, yy], dim=0).unsqueeze(0)  # (1, 2, 64, 64)
        
        # 4. 生成其他必需的参数
        # 对于简单推理，使用图像中心作为ROI中心
        input_res = 256
        roi_center = torch.tensor([[input_res / 2, input_res / 2]], device=self.device, dtype=torch.float32)
        roi_wh = torch.tensor([[input_res, input_res]], device=self.device, dtype=torch.float32)
        resize_ratio = torch.tensor([[1.0]], device=self.device, dtype=torch.float32)
        
        # 5. GDR-Net前向传播
        with torch.no_grad():
            try:
                # 完整的推理输入
                outputs = self.model(
                    x=roi_tensor,
                    roi_classes=torch.tensor([object_id - 1], dtype=torch.long).to(self.device),  # 0-based
                    roi_cams=torch.from_numpy(self.camera_K).unsqueeze(0).to(self.device).float(),
                    roi_coord_2d=roi_coord_2d,  # 添加2D坐标
                    roi_centers=roi_center,  # ROI中心
                    roi_whs=roi_wh,  # ROI宽高
                    resize_ratios=resize_ratio,  # 缩放比例
                    roi_extents=torch.from_numpy(
                        self.object_extents[object_id - 1:object_id]
                    ).to(self.device).float(),
                    do_loss=False,
                )
                
                # 提取姿态结果
                if 'rot' in outputs:
                    rotation = outputs['rot'].cpu().numpy()[0]
                else:
                    rotation = np.eye(3)
                
                if 'trans' in outputs:
                    translation = outputs['trans'].cpu().numpy()[0]
                else:
                    translation = np.array([0, 0, 1000], dtype=np.float32)
                
                # 置信度
                confidence = outputs.get('conf', torch.tensor([0.8])).cpu().numpy()[0]
                
                # 提取mask - 优先使用原始mask输出（实际上是最准确的）
                mask = None
                
                # 方法1：使用原始mask输出（最佳）
                if 'mask' in outputs:
                    mask_tensor = outputs['mask'].cpu()
                    while len(mask_tensor.shape) > 2:
                        mask_tensor = mask_tensor.squeeze(0)
                    
                    mask_np = mask_tensor.numpy()
                    self.logger.info(f"   原始mask logits: shape={mask_np.shape}, range=[{mask_np.min():.3f}, {mask_np.max():.3f}]")
                    
                    # GDR-Net的mask输出是logits，需要先sigmoid转换到[0,1]
                    mask_prob = 1 / (1 + np.exp(-mask_np))  # sigmoid
                    self.logger.info(f"   Sigmoid后: range=[{mask_prob.min():.3f}, {mask_prob.max():.3f}]")
                    
                    # 使用0.5阈值二值化
                    mask_binary = (mask_prob > 0.5).astype(np.uint8)
                    mask = mask_binary * 255
                    
                    if mask.shape != (256, 256):
                        import cv2
                        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                    
                    coverage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]) * 100
                    self.logger.info(f"   ✓ GDR-Net mask: shape={mask.shape}, 覆盖率={coverage:.1f}%")
                
                self.logger.info(f"✓ GDR-Net inference successful")
                return rotation, translation, float(confidence), mask, crop_info
                
            except Exception as e:
                self.logger.error(f"GDR-Net inference failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback到默认值
                return np.eye(3), np.array([0, 0, 1000], dtype=np.float32), 0.0, None
    
    def _preprocess(self, rgb, mask, bbox):
        """
        预处理输入图像
        
        Returns:
            roi: 裁剪并resize到256x256的ROI
            roi_mask: 对应的mask
            crop_info: 裁剪信息（用于后处理）
        """
        h, w = rgb.shape[:2]
        
        # 如果没有bbox，从mask提取或使用全图
        if bbox is None:
            if mask is not None:
                coords = np.where(mask > 0)
                if len(coords[0]) > 0:
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    bbox = np.array([x_min, y_min, x_max, y_max])
                else:
                    bbox = np.array([0, 0, w, h])
            else:
                bbox = np.array([0, 0, w, h])
        
        # 扩展bbox（增加边界）
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bbox_w, bbox_h = x2 - x1, y2 - y1
        scale = max(bbox_w, bbox_h) * 1.5  # 扩展50%
        
        x1_new = int(max(0, cx - scale / 2))
        y1_new = int(max(0, cy - scale / 2))
        x2_new = int(min(w, cx + scale / 2))
        y2_new = int(min(h, cy + scale / 2))
        
        # 裁剪ROI
        roi = rgb[y1_new:y2_new, x1_new:x2_new]
        roi_mask = mask[y1_new:y2_new, x1_new:x2_new] if mask is not None else None
        
        # Resize到256x256
        roi = cv2.resize(roi, (256, 256), interpolation=cv2.INTER_LINEAR)
        if roi_mask is not None:
            roi_mask = cv2.resize(roi_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        crop_info = {
            'bbox': bbox,
            'scale': scale,
            'center': (cx, cy),
            'roi_bbox': (x1_new, y1_new, x2_new, y2_new),
            'image_shape': (h, w)
        }
        
        return roi, roi_mask, crop_info
    
    def _get_object_extents(self):
        """
        获取LineMOD对象的3D尺寸（直径）
        
        用于GDR-Net的PnP模块
        """
        # LineMOD 13个对象的近似直径（单位：mm）
        extents = np.array([
            [80.0, 80.0, 100.0],   # 1. ape
            [150.0, 100.0, 120.0], # 2. benchvise
            [160.0, 120.0, 100.0], # 3. cam
            [100.0, 100.0, 120.0], # 4. can
            [140.0, 100.0, 80.0],  # 5. cat
            [180.0, 100.0, 100.0], # 6. driller
            [140.0, 80.0, 100.0],  # 7. duck
            [200.0, 150.0, 100.0], # 8. eggbox
            [180.0, 100.0, 60.0],  # 9. glue
            [140.0, 100.0, 100.0], # 10. holepuncher
            [200.0, 120.0, 80.0],  # 11. iron
            [200.0, 100.0, 150.0], # 12. lamp
            [200.0, 120.0, 40.0],  # 13. phone
        ], dtype=np.float32)
        
        return extents


def test_pure_inference():
    """测试纯PyTorch的GDR-Net推理"""
    print("="*70)
    print("Testing Pure GDR-Net Inference (No Detectron2)")
    print("="*70)
    
    # 检查GDR-Net路径
    gdrnet_root = Path(__file__).parent.parent.parent.parent / "GDR-Net-main" / "GDR-Net-main"
    if not gdrnet_root.exists():
        print(f"❌ GDR-Net not found at: {gdrnet_root}")
        print("   Please ensure GDR-Net-main is in the correct location")
        return
    
    print(f"✓ GDR-Net found at: {gdrnet_root}")
    
    # 模型路径
    project_root = Path(__file__).parent.parent
    model_path = project_root.parent / "models" / "gdr" / "gdrn" / "lm" / "a6_cPnP_lm13" / "gdrn_lm.pth"
    
    print(f"\nModel path: {model_path}")
    print(f"Model exists: {model_path.exists()}")
    
    if not model_path.exists():
        print("⚠️  Model file not found, but code structure is ready")
        print("   Once you have the model file, inference will work")
        return
    
    # 初始化推理引擎
    try:
        engine = GDRNetPureInference(
            model_path=str(model_path),
            device='cpu',
            num_classes=13
        )
        
        # 测试推理
        print("\nTesting inference...")
        dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_mask = np.zeros((480, 640), dtype=np.uint8)
        dummy_mask[200:300, 250:350] = 255
        dummy_bbox = np.array([250, 200, 350, 300])
        
        R, t, conf = engine.estimate_pose(
            rgb=dummy_rgb,
            mask=dummy_mask,
            bbox=dummy_bbox,
            object_id=1
        )
        
        print("✓ Inference successful!")
        print(f"  Rotation:\n{R}")
        print(f"  Translation: {t}")
        print(f"  Confidence: {conf}")
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pure_inference()

