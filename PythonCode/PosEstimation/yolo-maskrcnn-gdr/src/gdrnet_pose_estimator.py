#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDR-Net 6D姿态估计器
实现基于GDR-Net的6D物体姿态估计功能
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import time

class GDRNetPoseEstimator:
    """GDR-Net 6D姿态估计器"""
    
    def __init__(self, config: Dict):
        """
        初始化GDR-Net姿态估计器
        
        Args:
            config: 姿态估计配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设备配置
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 估计参数
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.use_segmentation_mask = config.get('use_segmentation_mask', True)
        self.refine_iterations = config.get('refine_iterations', 5)
        
        # LineMOD物体信息
        self.objects_info = config.get('objects', {})
        
        # 模型路径
        self.model_path = config.get('model_path', './models/gdrnet_model.pth')
        
        # 初始化模型（目前使用模拟实现）
        self.model = self._load_model()
        
        # 预定义的物体3D模型点（简化版）
        self.object_3d_models = self._load_3d_models()
        
        self.logger.info(f"✅ GDR-Net姿态估计器初始化完成 (设备: {self.device})")
    
    def _load_model(self):
        """加载GDR-Net模型"""
        # 注意: 这里使用模拟实现，实际项目中需要加载真实的GDR-Net模型
        self.logger.info("📦 使用模拟GDR-Net实现")
        return None  # 模拟模型
    
    def _load_3d_models(self) -> Dict[str, np.ndarray]:
        """加载物体3D模型点"""
        models = {}
        
        # 为每个LineMOD物体定义简化的3D模型点
        for obj_name, obj_info in self.objects_info.items():
            diameter = obj_info.get('diameter', 100.0)
            
            if obj_name == 'can':
                # 圆柱形模型
                models[obj_name] = self._create_cylinder_model(diameter)
            elif obj_name == 'phone':
                # 矩形模型
                models[obj_name] = self._create_box_model(diameter * 0.6, diameter * 0.3, diameter * 0.1)
            elif obj_name == 'ape':
                # 复杂形状模型
                models[obj_name] = self._create_complex_model(diameter)
            elif obj_name == 'lamp':
                # 灯具模型
                models[obj_name] = self._create_lamp_model(diameter)
            else:
                # 默认立方体模型
                models[obj_name] = self._create_box_model(diameter * 0.5, diameter * 0.5, diameter * 0.5)
        
        self.logger.info(f"📐 加载了 {len(models)} 个3D物体模型")
        return models
    
    def _create_cylinder_model(self, diameter: float) -> np.ndarray:
        """创建圆柱形3D模型"""
        radius = diameter / 2
        height = diameter * 1.2
        
        points = []
        
        # 底面圆
        for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y, -height/2])
            points.append([x, y, height/2])
        
        # 中心点
        points.extend([[0, 0, -height/2], [0, 0, height/2]])
        
        return np.array(points, dtype=np.float32)
    
    def _create_box_model(self, width: float, height: float, depth: float) -> np.ndarray:
        """创建立方体3D模型"""
        w, h, d = width/2, height/2, depth/2
        
        points = [
            [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],  # 底面
            [-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d],      # 顶面
            [0, 0, 0]  # 中心点
        ]
        
        return np.array(points, dtype=np.float32)
    
    def _create_complex_model(self, diameter: float) -> np.ndarray:
        """创建复杂形状3D模型（用于ape等）"""
        scale = diameter / 100
        
        # 基于真实ape模型的简化点云
        points = [
            # 头部
            [0, 0, 30*scale], [10*scale, 5*scale, 25*scale], [-10*scale, 5*scale, 25*scale],
            # 身体
            [0, -5*scale, 0], [15*scale, -5*scale, 0], [-15*scale, -5*scale, 0],
            [0, -15*scale, -10*scale], [12*scale, -15*scale, -8*scale], [-12*scale, -15*scale, -8*scale],
            # 四肢
            [20*scale, -10*scale, -20*scale], [-20*scale, -10*scale, -20*scale],
            [8*scale, -25*scale, -25*scale], [-8*scale, -25*scale, -25*scale],
        ]
        
        return np.array(points, dtype=np.float32)
    
    def _create_lamp_model(self, diameter: float) -> np.ndarray:
        """创建灯具3D模型"""
        scale = diameter / 200
        
        points = [
            # 灯罩 (梯形)
            [-30*scale, -20*scale, 20*scale], [30*scale, -20*scale, 20*scale],
            [-40*scale, -20*scale, -10*scale], [40*scale, -20*scale, -10*scale],
            # 灯杆
            [0, -20*scale, 10*scale], [0, -20*scale, -30*scale],
            # 底座
            [-20*scale, -20*scale, -35*scale], [20*scale, -20*scale, -35*scale],
            [0, -20*scale, -40*scale]
        ]
        
        return np.array(points, dtype=np.float32)
    
    def estimate_poses(self, image: np.ndarray, detections: List[Dict],
                      masks: Optional[List[np.ndarray]] = None,
                      camera_K: Optional[np.ndarray] = None) -> List[Dict]:
        """
        估计6D姿态
        
        Args:
            image: 输入图像 (BGR格式)
            detections: 检测结果列表
            masks: 可选的分割掩码列表
            camera_K: 相机内参矩阵
            
        Returns:
            姿态估计结果列表，每个元素包含:
            - rotation_matrix: 3x3旋转矩阵
            - translation: 3D平移向量
            - confidence: 估计置信度
            - object_name: 物体名称
            - method: 估计方法
        """
        if not detections:
            self.logger.warning("⚠️ 无检测结果，无法进行姿态估计")
            return []
        
        poses = []
        
        try:
            for i, detection in enumerate(detections):
                # 获取对应的掩码
                mask = masks[i] if masks and i < len(masks) else None
                
                # 估计单个物体的姿态
                pose = self._estimate_single_pose(image, detection, mask, camera_K)
                
                if pose is not None:
                    poses.append(pose)
                    
                    obj_name = pose['object_name']
                    confidence = pose['confidence']
                    self.logger.debug(f"   姿态估计 #{i+1}: {obj_name} (置信度: {confidence:.3f})")
            
            self.logger.info(f"📐 姿态估计完成: {len(poses)}/{len(detections)} 个物体")
            
        except Exception as e:
            self.logger.error(f"❌ 姿态估计失败: {e}")
        
        return poses
    
    def _estimate_single_pose(self, image: np.ndarray, detection: Dict,
                             mask: Optional[np.ndarray] = None,
                             camera_K: Optional[np.ndarray] = None) -> Optional[Dict]:
        """估计单个物体的6D姿态"""
        
        # 获取物体信息
        object_name = detection.get('mapped_class', detection.get('class_name', 'unknown'))
        bbox = detection['bbox']
        confidence = detection['confidence']
        
        # 检查是否支持该物体
        if object_name not in self.objects_info:
            self.logger.debug(f"⚠️ 不支持的物体类型: {object_name}")
            return None
        
        # 获取ROI
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        # 使用模拟的姿态估计
        pose = self._simulate_pose_estimation(object_name, bbox, confidence, camera_K)
        
        # 如果有掩码，可以进一步优化姿态
        if mask is not None and self.use_segmentation_mask:
            pose = self._refine_pose_with_mask(pose, mask, bbox)
        
        return pose
    
    def _simulate_pose_estimation(self, object_name: str, bbox: List[int],
                                 confidence: float, camera_K: Optional[np.ndarray]) -> Dict:
        """模拟姿态估计（用于演示）"""
        
        x, y, w, h = bbox
        center_x, center_y = x + w//2, y + h//2
        
        # 生成合理的旋转角度
        base_angles = {
            'can': [0.1, -0.05, 0.15],
            'phone': [-0.08, 0.12, -0.20],
            'ape': [0.05, -0.15, 0.08],
            'lamp': [0.12, -0.08, 0.25],
        }
        
        angles = base_angles.get(object_name, [0.0, 0.0, 0.0])
        
        # 添加一些随机变化
        angles = [angle + np.random.uniform(-0.1, 0.1) for angle in angles]
        
        # 构建旋转矩阵
        R = self._euler_to_rotation_matrix(angles)
        
        # 估计3D位置
        if camera_K is not None:
            # 使用相机参数估计深度
            fx, fy = camera_K[0, 0], camera_K[1, 1]
            cx, cy = camera_K[0, 2], camera_K[1, 2]
            
            # 基于物体大小估计深度
            object_diameter = self.objects_info[object_name]['diameter']
            estimated_depth = (object_diameter * fx) / max(w, h) * 2  # 简化的深度估计
            
            # 3D位置
            X = (center_x - cx) * estimated_depth / fx
            Y = (center_y - cy) * estimated_depth / fy
            Z = estimated_depth
            
        else:
            # 使用默认深度
            X = (center_x - 320) * 2.0  # 假设图像中心为(320, 240)
            Y = (center_y - 240) * 2.0
            Z = 800.0 + np.random.uniform(-100, 100)
        
        t = np.array([X, Y, Z], dtype=np.float32)
        
        # 计算姿态置信度
        pose_confidence = confidence * 0.9  # 姿态估计通常比检测置信度略低
        
        return {
            'rotation_matrix': R,
            'translation': t,
            'confidence': pose_confidence,
            'object_name': object_name,
            'method': 'GDR-Net (模拟)',
            'bbox': bbox,
            'angles_deg': [np.degrees(angle) for angle in angles]
        }
    
    def _refine_pose_with_mask(self, pose: Dict, mask: np.ndarray, bbox: List[int]) -> Dict:
        """使用分割掩码优化姿态估计"""
        
        # 获取掩码的几何特征
        x, y, w, h = bbox
        roi_mask = mask[y:y+h, x:x+w]
        
        if roi_mask.size == 0:
            return pose
        
        # 计算掩码的主轴方向
        moments = cv2.moments(roi_mask)
        if moments['m00'] == 0:
            return pose
        
        # 计算方向角
        mu20 = moments['mu20'] / moments['m00']
        mu02 = moments['mu02'] / moments['m00']
        mu11 = moments['mu11'] / moments['m00']
        
        if mu20 + mu02 != 0:
            orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
            
            # 使用方向信息微调旋转
            current_angles = pose['angles_deg']
            current_angles[2] += np.degrees(orientation) * 0.1  # 小幅调整
            
            # 重新计算旋转矩阵
            pose['rotation_matrix'] = self._euler_to_rotation_matrix(
                [np.radians(angle) for angle in current_angles]
            )
            pose['angles_deg'] = current_angles
        
        # 提高置信度（因为有掩码信息）
        pose['confidence'] = min(1.0, pose['confidence'] * 1.1)
        pose['method'] = 'GDR-Net + Mask优化 (模拟)'
        
        return pose
    
    def _euler_to_rotation_matrix(self, angles: List[float]) -> np.ndarray:
        """欧拉角转旋转矩阵 (ZYX顺序)"""
        rx, ry, rz = angles
        
        # 绕X轴旋转
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ], dtype=np.float32)
        
        # 绕Y轴旋转
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ], dtype=np.float32)
        
        # 绕Z轴旋转
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 组合旋转 (ZYX顺序)
        R = Rz @ Ry @ Rx
        return R
    
    def project_3d_points(self, points_3d: np.ndarray, rotation_matrix: np.ndarray,
                         translation: np.ndarray, camera_K: np.ndarray) -> np.ndarray:
        """
        将3D点投影到2D图像平面
        
        Args:
            points_3d: 3D点 (Nx3)
            rotation_matrix: 3x3旋转矩阵
            translation: 3D平移向量
            camera_K: 相机内参矩阵
            
        Returns:
            2D投影点 (Nx2)
        """
        # 变换3D点
        transformed_points = (rotation_matrix @ points_3d.T).T + translation
        
        # 投影到2D
        projected_points = camera_K @ transformed_points.T
        projected_points = projected_points / projected_points[2, :]
        
        return projected_points[:2, :].T
    
    def visualize_pose(self, image: np.ndarray, pose: Dict, 
                      camera_K: Optional[np.ndarray] = None) -> np.ndarray:
        """
        可视化单个物体的6D姿态
        
        Args:
            image: 原始图像
            pose: 姿态估计结果
            camera_K: 相机内参矩阵
            
        Returns:
            标注后的图像
        """
        result_image = image.copy()
        
        if camera_K is None:
            # 使用默认相机参数
            camera_K = np.array([
                [572.4, 0, 320],
                [0, 573.6, 240],
                [0, 0, 1]
            ], dtype=np.float32)
        
        R = pose['rotation_matrix']
        t = pose['translation']
        object_name = pose['object_name']
        
        # 1. 绘制3D坐标轴
        axis_length = 50.0
        axis_points = np.array([
            [0, 0, 0],           # 原点
            [axis_length, 0, 0], # X轴
            [0, axis_length, 0], # Y轴
            [0, 0, axis_length], # Z轴
        ], dtype=np.float32)
        
        projected_axes = self.project_3d_points(axis_points, R, t, camera_K)
        
        if len(projected_axes) == 4:
            origin = tuple(map(int, projected_axes[0]))
            x_end = tuple(map(int, projected_axes[1]))
            y_end = tuple(map(int, projected_axes[2]))
            z_end = tuple(map(int, projected_axes[3]))
            
            # 绘制坐标轴 (RGB对应XYZ)
            cv2.arrowedLine(result_image, origin, x_end, (0, 0, 255), 3, tipLength=0.3)  # X-红
            cv2.arrowedLine(result_image, origin, y_end, (0, 255, 0), 3, tipLength=0.3)  # Y-绿
            cv2.arrowedLine(result_image, origin, z_end, (255, 0, 0), 3, tipLength=0.3)  # Z-蓝
        
        # 2. 绘制3D包围盒
        if object_name in self.object_3d_models:
            model_points = self.object_3d_models[object_name]
            projected_model = self.project_3d_points(model_points, R, t, camera_K)
            
            # 绘制模型点
            for point in projected_model:
                pt = tuple(map(int, point))
                cv2.circle(result_image, pt, 2, (0, 255, 255), -1)
            
            # 如果是立方体，绘制边框
            if len(projected_model) >= 8:
                self._draw_3d_bbox_edges(result_image, projected_model[:8])
        
        # 3. 添加姿态信息
        bbox = pose['bbox']
        x, y, w, h = bbox
        
        info_text = [
            f"{object_name}",
            f"Conf: {pose['confidence']:.2f}",
            f"X: {t[0]:.1f}mm",
            f"Y: {t[1]:.1f}mm", 
            f"Z: {t[2]:.1f}mm"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(result_image, text, (x, y - 10 - i * 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image
    
    def _draw_3d_bbox_edges(self, image: np.ndarray, corners: np.ndarray):
        """绘制3D包围盒边框"""
        if len(corners) < 8:
            return
        
        # 立方体的12条边
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7),  # 竖直边
        ]
        
        for edge in edges:
            if edge[0] < len(corners) and edge[1] < len(corners):
                pt1 = tuple(map(int, corners[edge[0]]))
                pt2 = tuple(map(int, corners[edge[1]]))
                cv2.line(image, pt1, pt2, (0, 0, 255), 2)
    
    def get_pose_statistics(self, poses: List[Dict]) -> Dict:
        """获取姿态估计统计信息"""
        if not poses:
            return {'count': 0}
        
        confidences = [pose['confidence'] for pose in poses]
        translations = [pose['translation'] for pose in poses]
        
        # 计算平均距离
        distances = [np.linalg.norm(t) for t in translations]
        
        return {
            'count': len(poses),
            'average_confidence': np.mean(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'average_distance': np.mean(distances),
            'min_distance': min(distances),
            'max_distance': max(distances),
            'objects': [pose['object_name'] for pose in poses]
        }

def test_gdrnet_pose_estimator():
    """测试GDR-Net姿态估计器"""
    print("🧪 测试GDR-Net姿态估计器...")
    
    # 测试配置
    config = {
        'confidence_threshold': 0.6,
        'device': 'cpu',
        'use_segmentation_mask': True,
        'refine_iterations': 5,
        'objects': {
            'can': {'id': 5, 'diameter': 202.0},
            'phone': {'id': 15, 'diameter': 213.0},
            'ape': {'id': 1, 'diameter': 102.0},
            'lamp': {'id': 14, 'diameter': 284.0}
        }
    }
    
    # 创建姿态估计器
    estimator = GDRNetPoseEstimator(config)
    
    # 创建测试图像
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 120
    
    # 添加一些模拟物体
    cv2.rectangle(test_image, (100, 100), (200, 200), (140, 160, 120), -1)
    cv2.circle(test_image, (350, 150), 60, (120, 140, 180), -1)
    cv2.ellipse(test_image, (500, 300), (50, 80), 0, 0, 360, (160, 120, 140), -1)
    
    # 模拟检测结果
    detections = [
        {'bbox': [100, 100, 100, 100], 'class_name': 'cup', 'mapped_class': 'can', 'confidence': 0.9},
        {'bbox': [290, 90, 120, 120], 'class_name': 'phone', 'mapped_class': 'phone', 'confidence': 0.8},
        {'bbox': [450, 220, 100, 160], 'class_name': 'person', 'mapped_class': 'ape', 'confidence': 0.85}
    ]
    
    # 模拟分割掩码
    masks = []
    for detection in detections:
        x, y, w, h = detection['bbox']
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (x+5, y+5), (x+w-5, y+h-5), 255, -1)
        masks.append(mask)
    
    # 相机内参
    camera_K = np.array([
        [572.4, 0, 320],
        [0, 573.6, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 执行姿态估计
    poses = estimator.estimate_poses(test_image, detections, masks, camera_K)
    
    print(f"✅ 姿态估计结果: {len(poses)} 个物体")
    
    # 获取统计信息
    stats = estimator.get_pose_statistics(poses)
    print(f"📊 姿态统计: {stats}")
    
    # 可视化每个姿态
    result_image = test_image.copy()
    for pose in poses:
        result_image = estimator.visualize_pose(result_image, pose, camera_K)
    
    # 显示结果
    cv2.imshow("GDR-Net姿态估计结果", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("🎉 GDR-Net姿态估计器测试完成!")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_gdrnet_pose_estimator()
