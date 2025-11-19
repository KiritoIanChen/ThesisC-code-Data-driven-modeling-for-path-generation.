#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LineMOD 6D Pose Estimation V2 - Data Structures
数据结构定义

Date: 2025-10-10
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class Detection:
    """检测结果数据结构"""
    bbox: np.ndarray  # [x1, y1, x2, y2] or [x, y, w, h] - numpy array
    class_id: int  # LineMOD类别ID (0-12)
    class_name: str  # 类别名称 ('ape', 'can', etc.)
    confidence: float  # 置信度 (0.0-1.0)
    detection_id: Optional[int] = None  # 检测ID (用于追踪)
    
    def __post_init__(self):
        """验证数据"""
        if not isinstance(self.bbox, np.ndarray):
            self.bbox = np.array(self.bbox)
        assert len(self.bbox) == 4, "bbox must have 4 elements"
        assert 0 <= self.class_id <= 12, "class_id must be 0-12 for LineMOD"
        assert 0.0 <= self.confidence <= 1.0, "confidence must be 0.0-1.0"
    
    def get_center(self) -> tuple:
        """获取边界框中心点"""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
    def get_area(self) -> int:
        """获取边界框面积"""
        return self.bbox[2] * self.bbox[3]
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'bbox': self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'detection_id': self.detection_id
        }


@dataclass
class Mask:
    """分割掩码数据结构"""
    mask: np.ndarray  # Binary mask (H, W)
    detection_id: int  # 对应的检测ID
    quality_score: float = 1.0  # 掩码质量评分 (0.0-1.0)
    method: str = "unknown"  # 生成方法 ('simplified', 'sam', 'maskrcnn')
    
    def __post_init__(self):
        """验证数据"""
        assert len(self.mask.shape) == 2, "mask must be 2D array"
        assert 0.0 <= self.quality_score <= 1.0, "quality_score must be 0.0-1.0"
    
    def get_area(self) -> int:
        """获取掩码面积（像素数）"""
        return int(np.sum(self.mask > 0))
    
    def get_bbox(self) -> List[int]:
        """从掩码提取边界框"""
        coords = np.where(self.mask > 0)
        if len(coords[0]) == 0:
            return [0, 0, 0, 0]
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    
    def to_dict(self) -> Dict:
        """转换为字典（不包含mask数组）"""
        return {
            'detection_id': self.detection_id,
            'quality_score': self.quality_score,
            'method': self.method,
            'area': self.get_area(),
            'bbox': self.get_bbox()
        }


@dataclass
class Pose6D:
    """6D姿态数据结构"""
    rotation_matrix: np.ndarray  # 旋转矩阵 (3, 3)
    translation: np.ndarray  # 平移向量 (3,) in mm
    object_id: int  # 物体ID
    object_name: str  # 物体名称
    confidence: float  # 姿态置信度
    method: str = "unknown"  # 估计方法 ('pnp', 'gdrnet')
    reprojection_error: Optional[float] = None  # 重投影误差 (pixels)
    num_inliers: Optional[int] = None  # 内点数量
    detection_id: Optional[int] = None  # 对应的检测ID
    
    def __post_init__(self):
        """验证数据"""
        assert self.rotation_matrix.shape == (3, 3), "rotation_matrix must be (3, 3)"
        assert self.translation.shape == (3,), "translation must be (3,)"
        assert 0.0 <= self.confidence <= 1.0, "confidence must be 0.0-1.0"
    
    def get_rotation_angles(self) -> np.ndarray:
        """从旋转矩阵提取欧拉角 (rx, ry, rz) in radians"""
        import cv2
        rvec, _ = cv2.Rodrigues(self.rotation_matrix)
        return rvec.flatten()
    
    def get_rotation_angles_degrees(self) -> np.ndarray:
        """获取欧拉角 (degrees)"""
        return np.degrees(self.get_rotation_angles())
    
    def get_transform_matrix(self) -> np.ndarray:
        """获取4x4变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix
        T[:3, 3] = self.translation
        return T
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'rotation_matrix': self.rotation_matrix.tolist(),
            'translation': self.translation.tolist(),
            'rotation_angles_deg': self.get_rotation_angles_degrees().tolist(),
            'object_id': self.object_id,
            'object_name': self.object_name,
            'confidence': self.confidence,
            'method': self.method,
            'reprojection_error': self.reprojection_error,
            'num_inliers': self.num_inliers,
            'detection_id': self.detection_id
        }


@dataclass
class PipelineResult:
    """Pipeline处理结果"""
    image_name: str
    image_shape: tuple  # (H, W, C)
    success: bool
    
    # 检测结果
    detections: List[Detection] = field(default_factory=list)
    
    # 分割结果
    masks: List[Mask] = field(default_factory=list)
    
    # 姿态结果
    poses: List[Pose6D] = field(default_factory=list)
    
    # 处理时间 (秒)
    processing_times: Dict[str, float] = field(default_factory=dict)
    
    # 相机参数
    camera_K: Optional[np.ndarray] = None
    
    # 可视化图像
    annotated_image: Optional[np.ndarray] = None
    
    # 额外信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_time(self) -> float:
        """获取总处理时间"""
        return self.processing_times.get('total', 
               sum(self.processing_times.values()))
    
    def get_fps(self) -> float:
        """获取处理速度 (FPS)"""
        total_time = self.get_total_time()
        return 1.0 / total_time if total_time > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'image_name': self.image_name,
            'image_shape': self.image_shape,
            'success': self.success,
            'num_detections': len(self.detections),
            'num_masks': len(self.masks),
            'num_poses': len(self.poses),
            'detections': [d.to_dict() for d in self.detections],
            'masks': [m.to_dict() for m in self.masks],
            'poses': [p.to_dict() for p in self.poses],
            'processing_times': self.processing_times,
            'total_time': self.get_total_time(),
            'fps': self.get_fps(),
            'camera_K': self.camera_K.tolist() if self.camera_K is not None else None,
            'metadata': self.metadata
        }
    
    def summary(self) -> str:
        """生成结果摘要"""
        lines = [
            f"Pipeline Result: {self.image_name}",
            f"  Success: {self.success}",
            f"  Detections: {len(self.detections)}",
            f"  Masks: {len(self.masks)}",
            f"  Poses: {len(self.poses)}",
            f"  Processing Time: {self.get_total_time():.3f}s ({self.get_fps():.1f} FPS)",
        ]
        
        if self.processing_times:
            lines.append("  Time Breakdown:")
            for stage, time_val in self.processing_times.items():
                if stage != 'total':
                    percentage = (time_val / self.get_total_time() * 100) if self.get_total_time() > 0 else 0
                    lines.append(f"    - {stage}: {time_val:.3f}s ({percentage:.1f}%)")
        
        return "\n".join(lines)


# LineMOD类别定义
LINEMOD_CLASSES = [
    'ape',          # 0
    'benchvise',    # 1
    'cam',          # 2
    'can',          # 3
    'cat',          # 4
    'driller',      # 5
    'duck',         # 6
    'eggbox',       # 7
    'glue',         # 8
    'holepuncher',  # 9
    'iron',         # 10
    'lamp',         # 11
    'phone',        # 12
]

# LineMOD类别ID映射
LINEMOD_CLASS_IDS = {name: idx for idx, name in enumerate(LINEMOD_CLASSES)}

# LineMOD物体直径 (mm)
LINEMOD_DIAMETERS = {
    'ape': 102.0,
    'benchvise': 247.0,
    'cam': 172.0,
    'can': 202.0,
    'cat': 154.0,
    'driller': 262.0,
    'duck': 109.0,
    'eggbox': 164.0,
    'glue': 176.0,
    'holepuncher': 146.0,
    'iron': 143.0,
    'lamp': 284.0,
    'phone': 213.0,
}


def get_class_name(class_id: int) -> str:
    """根据类别ID获取类别名称"""
    if 0 <= class_id < len(LINEMOD_CLASSES):
        return LINEMOD_CLASSES[class_id]
    return f"unknown_{class_id}"


def get_class_id(class_name: str) -> int:
    """根据类别名称获取类别ID"""
    return LINEMOD_CLASS_IDS.get(class_name, -1)


def get_object_diameter(class_name: str) -> float:
    """获取物体直径"""
    return LINEMOD_DIAMETERS.get(class_name, 100.0)

