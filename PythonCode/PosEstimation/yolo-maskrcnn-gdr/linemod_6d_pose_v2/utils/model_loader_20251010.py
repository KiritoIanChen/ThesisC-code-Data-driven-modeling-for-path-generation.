#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Model Loader
3D模型加载工具

加载LineMOD的.ply格式3D模型

Date: 2025-10-10
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging


class ModelLoader:
    """
    LineMOD 3D模型加载器
    """
    
    def __init__(self, models_dir: str):
        """
        初始化模型加载器
        
        Args:
            models_dir: 模型文件目录
        """
        self.models_dir = Path(models_dir)
        self.logger = logging.getLogger(__name__)
        self.models_cache = {}  # 缓存已加载的模型
        
        # LineMOD物体ID到模型文件名的映射
        self.id_to_filename = {
            1: 'obj_01.ply',  # ape
            2: 'obj_02.ply',  # benchvise
            4: 'obj_04.ply',  # cam
            5: 'obj_05.ply',  # can
            6: 'obj_06.ply',  # cat
            8: 'obj_08.ply',  # driller
            9: 'obj_09.ply',  # duck
            10: 'obj_10.ply',  # eggbox
            11: 'obj_11.ply',  # glue
            12: 'obj_12.ply',  # holepuncher
            13: 'obj_13.ply',  # iron
            14: 'obj_14.ply',  # lamp
            15: 'obj_15.ply',  # phone
        }
        
        if not self.models_dir.exists():
            self.logger.warning(f"Models directory not found: {self.models_dir}")
    
    def load_model(self, object_id: int) -> Optional[np.ndarray]:
        """
        加载指定物体的3D模型
        
        Args:
            object_id: LineMOD物体ID (1-15)
        
        Returns:
            点云数组 (N, 3) in mm, or None if failed
        """
        # 检查缓存
        if object_id in self.models_cache:
            return self.models_cache[object_id]
        
        # 获取文件名
        if object_id not in self.id_to_filename:
            self.logger.error(f"Invalid object_id: {object_id}")
            return None
        
        filename = self.id_to_filename[object_id]
        model_path = self.models_dir / filename
        
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            # 加载PLY文件
            points = self._load_ply(model_path)
            
            # 缓存
            self.models_cache[object_id] = points
            
            self.logger.info(f"✅ Loaded model for object {object_id}: {points.shape[0]} points")
            
            return points
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {e}")
            return None
    
    def _load_ply(self, filepath: Path) -> np.ndarray:
        """
        加载PLY格式文件
        
        Args:
            filepath: PLY文件路径
        
        Returns:
            点云 (N, 3)
        """
        try:
            # 尝试使用plyfile库
            from plyfile import PlyData
            
            ply_data = PlyData.read(str(filepath))
            vertex = ply_data['vertex']
            
            # 提取xyz坐标
            x = np.array(vertex['x'])
            y = np.array(vertex['y'])
            z = np.array(vertex['z'])
            
            points = np.stack([x, y, z], axis=1)
            
            return points
            
        except ImportError:
            # 如果没有plyfile，使用简单的文本解析
            self.logger.warning("plyfile not installed, using simple parser")
            return self._load_ply_simple(filepath)
    
    def _load_ply_simple(self, filepath: Path) -> np.ndarray:
        """
        简单的PLY文件解析器（不依赖plyfile）
        
        Args:
            filepath: PLY文件路径
        
        Returns:
            点云 (N, 3)
        """
        points = []
        
        with open(filepath, 'r') as f:
            # 跳过头部
            header_end = False
            for line in f:
                if line.strip() == 'end_header':
                    header_end = True
                    break
            
            if not header_end:
                raise ValueError("Invalid PLY file: no end_header found")
            
            # 读取顶点数据
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                    except ValueError:
                        continue
        
        return np.array(points, dtype=np.float32)
    
    def get_model_bbox(self, object_id: int) -> Optional[np.ndarray]:
        """
        获取模型的边界框尺寸
        
        Args:
            object_id: LineMOD物体ID
        
        Returns:
            [min_x, min_y, min_z, max_x, max_y, max_z] in mm
        """
        points = self.load_model(object_id)
        
        if points is None:
            return None
        
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        
        bbox = np.concatenate([min_xyz, max_xyz])
        
        return bbox

