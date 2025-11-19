#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LineMOD 6D Pose Estimation V2 - Configuration Manager
配置管理系统

Date: 2025-10-10
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，None则使用默认配置
        """
        self.logger = logging.getLogger(__name__)
        self.config = {}
        
        if config_path:
            self.load_config(config_path)
        else:
            self.config = self._get_default_config()
            self.logger.info("使用默认配置")
    
    def load_config(self, config_path: Union[str, Path]) -> Dict:
        """
        加载配置文件
        
        Args:
            config_path: YAML配置文件路径
            
        Returns:
            配置字典
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            self.logger.error(f"配置文件不存在: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 处理继承 (_base_)
            if '_base_' in config:
                base_path = config_path.parent / config['_base_']
                base_config = self.load_config(base_path)
                config = self.merge_configs(base_config, config)
                del config['_base_']
            
            self.config = config
            self.logger.info(f"✅ 配置加载成功: {config_path}")
            
            return config
            
        except yaml.YAMLError as e:
            self.logger.error(f"❌ YAML解析错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ 配置加载失败: {e}")
            raise
    
    def merge_configs(self, base: Dict, override: Dict) -> Dict:
        """
        合并配置字典
        
        Args:
            base: 基础配置
            override: 覆盖配置
            
        Returns:
            合并后的配置
        """
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # 递归合并字典
                merged[key] = self.merge_configs(merged[key], value)
            else:
                # 直接覆盖
                merged[key] = value
        
        return merged
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的路径）
        
        Args:
            key_path: 配置路径，如 'detection.confidence_threshold'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """
        设置配置值
        
        Args:
            key_path: 配置路径
            value: 配置值
        """
        keys = key_path.split('.')
        config = self.config
        
        # 导航到最后一级
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置值
        config[keys[-1]] = value
    
    def validate(self) -> bool:
        """
        验证配置完整性
        
        Returns:
            是否有效
        """
        required_keys = [
            'camera.intrinsics',
            'detection.model_path',
        ]
        
        for key_path in required_keys:
            if self.get(key_path) is None:
                self.logger.error(f"❌ 缺少必需配置: {key_path}")
                return False
        
        # 验证路径存在性
        model_path = self.get('detection.model_path')
        if model_path and not Path(model_path).exists():
            self.logger.warning(f"⚠️ YOLO模型文件不存在: {model_path}")
        
        self.logger.info("✅ 配置验证通过")
        return True
    
    def save(self, output_path: Union[str, Path]):
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            self.logger.info(f"✅ 配置已保存: {output_path}")
        except Exception as e:
            self.logger.error(f"❌ 配置保存失败: {e}")
            raise
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'project': {
                'name': 'LineMOD 6D Pose Estimation V2',
                'version': '2.0.0'
            },
            'camera': {
                'intrinsics': {
                    'fx': 572.4114,
                    'fy': 573.57043,
                    'cx': 325.2611,
                    'cy': 242.04899
                },
                'resolution': {
                    'width': 640,
                    'height': 480
                }
            },
            'detection': {
                'model_type': 'yolo',
                'model_path': None,
                'confidence_threshold': 0.3,
                'nms_threshold': 0.45,
                'device': 'cpu'
            },
            'segmentation': {
                'enable': True,
                'method': 'simplified'
            },
            'pose_estimation': {
                'method': 'pnp',
                'use_mask': True
            },
            'visualization': {
                'show_bbox': True,
                'show_mask': True,
                'show_3d_axes': True,
                'save_results': True
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def get_camera_matrix(self):
        """获取相机内参矩阵"""
        import numpy as np
        
        fx = self.get('camera.intrinsics.fx')
        fy = self.get('camera.intrinsics.fy')
        cx = self.get('camera.intrinsics.cx')
        cy = self.get('camera.intrinsics.cy')
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"ConfigManager(keys={list(self.config.keys())})"
    
    def __str__(self) -> str:
        """打印配置"""
        return yaml.dump(self.config, default_flow_style=False)


def setup_logging(config: ConfigManager):
    """
    根据配置设置日志
    
    Args:
        config: 配置管理器
    """
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.log_format', 
                           '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # 如果配置了保存日志
    if config.get('logging.save_logs', False):
        log_file = config.get('logging.log_file', 'pipeline.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


if __name__ == "__main__":
    # 测试配置管理器
    print("🧪 测试配置管理器")
    
    # 测试默认配置
    config = ConfigManager()
    print("\n默认配置:")
    print(config)
    
    # 测试配置访问
    print("\n测试配置访问:")
    print(f"detection.confidence_threshold = {config.get('detection.confidence_threshold')}")
    print(f"camera.intrinsics.fx = {config.get('camera.intrinsics.fx')}")
    
    # 测试相机矩阵
    print("\n相机内参矩阵:")
    print(config.get_camera_matrix())
    
    print("\n✅ 配置管理器测试完成")

