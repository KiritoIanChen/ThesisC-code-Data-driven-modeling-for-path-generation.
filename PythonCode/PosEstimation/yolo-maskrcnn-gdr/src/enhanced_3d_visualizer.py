#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强3D可视化器
实现完整的2D+3D可视化功能，包括分割掩码、3D包围盒、坐标轴等
"""

import cv2
import numpy as np
import open3d as o3d
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path
import time
from datetime import datetime

class Enhanced3DVisualizer:
    """增强3D可视化器"""
    
    def __init__(self, config: Dict):
        """
        初始化增强3D可视化器
        
        Args:
            config: 可视化配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 2D可视化配置
        self.show_detection_boxes = config.get('show_detection_boxes', True)
        self.show_segmentation_masks = config.get('show_segmentation_masks', True)
        self.show_pose_axes = config.get('show_pose_axes', True)
        self.show_3d_bboxes = config.get('show_3d_bboxes', True)
        self.show_info_panel = config.get('show_info_panel', True)
        
        # 颜色配置
        self.colors = config.get('colors', {})
        self._setup_default_colors()
        
        # 3D可视化配置
        self.point_cloud_size = config.get('point_cloud_size', 2.0)
        self.coordinate_frame_size = config.get('coordinate_frame_size', 0.1)
        self.mesh_alpha = config.get('mesh_alpha', 0.8)
        
        # 保存配置
        self.save_results = config.get('save_results', True)
        self.save_format = config.get('save_format', 'jpg')
        self.save_quality = config.get('save_quality', 95)
        
        self.logger.info("✅ 增强3D可视化器初始化完成")
    
    def _setup_default_colors(self):
        """设置默认颜色"""
        default_colors = {
            'detection_box': [0, 255, 0],      # 绿色
            'segmentation_mask': [0, 255, 255], # 黄色
            'bbox_3d': [0, 0, 255],            # 红色
            'axis_x': [0, 0, 255],             # 红色 X轴
            'axis_y': [0, 255, 0],             # 绿色 Y轴  
            'axis_z': [255, 0, 0],             # 蓝色 Z轴
            'text': [255, 255, 255],           # 白色文字
            'background': [0, 0, 0],           # 黑色背景
        }
        
        for key, default_color in default_colors.items():
            if key not in self.colors:
                self.colors[key] = default_color
    
    def create_complete_visualization(self, image: np.ndarray, detections: List[Dict],
                                    masks: Optional[List[np.ndarray]] = None,
                                    poses: Optional[List[Dict]] = None,
                                    camera_K: Optional[np.ndarray] = None) -> Dict:
        """
        创建完整的可视化结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            masks: 分割掩码列表
            poses: 姿态估计结果列表
            camera_K: 相机内参矩阵
            
        Returns:
            完整的可视化结果字典
        """
        result_image = image.copy()
        
        self.logger.info(f"🎨 创建完整可视化: {len(detections)} 个物体")
        
        # 1. 绘制分割掩码 (黄色轮廓)
        if masks and self.show_segmentation_masks:
            result_image = self._draw_segmentation_masks(result_image, masks, detections)
        
        # 2. 绘制3D包围盒 (红色线框)
        if poses and self.show_3d_bboxes:
            result_image = self._draw_3d_bounding_boxes(result_image, poses, camera_K)
        
        # 3. 绘制姿态坐标轴 (RGB三色)
        if poses and self.show_pose_axes:
            result_image = self._draw_pose_axes(result_image, poses, camera_K)
        
        # 4. 绘制检测框和标签
        if self.show_detection_boxes:
            result_image = self._draw_detection_boxes_and_labels(result_image, detections, poses)
        
        # 5. 添加信息面板
        if self.show_info_panel:
            result_image = self._add_comprehensive_info_panel(result_image, detections, masks, poses)
        
        # 6. 创建3D场景
        scene_3d = None
        if poses:
            scene_3d = self._create_3d_scene(poses, camera_K)
        
        visualization_result = {
            'annotated_image': result_image,
            'scene_3d': scene_3d,
            'detections': detections,
            'masks': masks,
            'poses': poses,
            'timestamp': time.time()
        }
        
        # 保存结果
        if self.save_results:
            self._save_visualization_results(visualization_result)
        
        return visualization_result
    
    def _draw_segmentation_masks(self, image: np.ndarray, masks: List[np.ndarray], 
                               detections: List[Dict]) -> np.ndarray:
        """绘制分割掩码 - 黄色轮廓和半透明填充"""
        result = image.copy()
        
        # 分割轮廓颜色 (多种颜色循环)
        mask_colors = [
            tuple(self.colors['segmentation_mask']),  # 主要黄色
            (255, 0, 255),    # 品红
            (255, 255, 0),    # 青色
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 蓝色
            (128, 255, 255),  # 浅黄
            (255, 128, 255),  # 浅品红
            (255, 255, 128),  # 浅青
        ]
        
        for i, (mask, detection) in enumerate(zip(masks, detections)):
            if mask is None:
                continue
            
            color = mask_colors[i % len(mask_colors)]
            
            # 找到轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 绘制粗轮廓线
                cv2.drawContours(result, contours, -1, color, 4)
                
                # 创建半透明填充
                overlay = result.copy()
                cv2.fillPoly(overlay, contours, color)
                cv2.addWeighted(overlay, 0.15, result, 0.85, 0, result)
                
                self.logger.debug(f"   ✅ 绘制分割掩码: {detection.get('mapped_class', 'unknown')} (#{i+1})")
        
        return result
    
    def _draw_3d_bounding_boxes(self, image: np.ndarray, poses: List[Dict],
                              camera_K: Optional[np.ndarray]) -> np.ndarray:
        """绘制3D包围盒 - 红色线框"""
        result = image.copy()
        bbox_color = tuple(self.colors['bbox_3d'])
        
        for i, pose in enumerate(poses):
            if camera_K is not None:
                # 使用真实投影
                bbox_3d = self._get_object_3d_bbox(pose['object_name'])
                if bbox_3d is not None:
                    projected_bbox = self._project_3d_bbox(bbox_3d, pose, camera_K)
                    if projected_bbox is not None:
                        self._draw_projected_3d_bbox(result, projected_bbox, bbox_color)
            else:
                # 使用简化的3D效果
                self._draw_simple_3d_bbox(result, pose, bbox_color, i)
        
        return result
    
    def _draw_pose_axes(self, image: np.ndarray, poses: List[Dict],
                       camera_K: Optional[np.ndarray]) -> np.ndarray:
        """绘制姿态坐标轴 - RGB三色"""
        result = image.copy()
        
        axis_colors = {
            'x': tuple(self.colors['axis_x']),  # 红色 X轴
            'y': tuple(self.colors['axis_y']),  # 绿色 Y轴
            'z': tuple(self.colors['axis_z']),  # 蓝色 Z轴
        }
        
        for pose in poses:
            if camera_K is not None:
                # 使用真实投影
                self._draw_projected_axes(result, pose, camera_K, axis_colors)
            else:
                # 使用简化的坐标轴
                self._draw_simple_axes(result, pose, axis_colors)
        
        return result
    
    def _draw_detection_boxes_and_labels(self, image: np.ndarray, detections: List[Dict],
                                       poses: Optional[List[Dict]] = None) -> np.ndarray:
        """绘制检测框和详细标签"""
        result = image.copy()
        
        # 物体ID颜色
        id_colors = [
            (0, 255, 255),    # 黄色
            (255, 0, 255),    # 品红
            (255, 255, 0),    # 青色
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 蓝色
            (128, 255, 128),  # 浅绿
            (255, 128, 128),  # 浅红
            (128, 128, 255),  # 浅蓝
        ]
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            class_name = detection.get('mapped_class', detection.get('class_name', 'unknown'))
            confidence = detection['confidence']
            
            color = id_colors[i % len(id_colors)]
            center_x, center_y = x + w//2, y + h//2
            
            # 1. 绘制物体ID圆圈
            cv2.circle(result, (center_x, center_y), 20, color, -1)
            cv2.circle(result, (center_x, center_y), 20, (255, 255, 255), 2)
            
            # ID文字
            object_id = f"#{i+1}"
            text_size = cv2.getTextSize(object_id, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(result, object_id, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 2. 绘制详细标签
            labels = [f"{class_name}: {confidence:.2f}"]
            
            # 添加姿态信息
            if poses and i < len(poses):
                pose = poses[i]
                t = pose['translation']
                labels.append(f"位置: ({t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f})")
                labels.append(f"姿态置信度: {pose['confidence']:.2f}")
            
            # 绘制标签背景和文字
            label_y = y - 10
            for label in labels:
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 标签背景
                cv2.rectangle(result, (x, label_y - label_size[1] - 5), 
                             (x + label_size[0] + 10, label_y + 5), color, -1)
                cv2.rectangle(result, (x, label_y - label_size[1] - 5), 
                             (x + label_size[0] + 10, label_y + 5), (255, 255, 255), 1)
                
                # 标签文字
                cv2.putText(result, label, (x + 5, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                label_y -= (label_size[1] + 10)
        
        return result
    
    def _add_comprehensive_info_panel(self, image: np.ndarray, detections: List[Dict],
                                    masks: Optional[List[np.ndarray]] = None,
                                    poses: Optional[List[Dict]] = None) -> np.ndarray:
        """添加综合信息面板"""
        result = image.copy()
        
        panel_height = 140
        panel_width = result.shape[1]
        
        # 创建半透明面板
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), 
                     tuple(self.colors['background']), -1)
        cv2.addWeighted(overlay, 0.85, result, 0.15, 0, result)
        
        text_color = tuple(self.colors['text'])
        
        # 主标题
        title = f"完整6D姿态估计Pipeline - 检测到 {len(detections)} 个物体"
        cv2.putText(result, title, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        
        # 流程说明
        process_text = "流程: YOLO检测 → Mask R-CNN分割 → GDR-Net姿态估计 → 3D可视化"
        cv2.putText(result, process_text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # 物体列表
        if detections:
            objects_info = []
            for i, det in enumerate(detections):
                obj_name = det.get('mapped_class', det.get('class_name', 'unknown'))
                objects_info.append(f"#{i+1} {obj_name} ({det['confidence']:.2f})")
            
            objects_text = "检测物体: " + " | ".join(objects_info)
            if len(objects_text) > 80:  # 如果太长，截断
                objects_text = objects_text[:77] + "..."
            cv2.putText(result, objects_text, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # 技术统计
        tech_stats = []
        if masks:
            tech_stats.append(f"分割掩码: {len(masks)}")
        if poses:
            avg_conf = np.mean([p['confidence'] for p in poses])
            tech_stats.append(f"姿态估计: {len(poses)} (平均置信度: {avg_conf:.2f})")
        
        if tech_stats:
            stats_text = "技术统计: " + " | ".join(tech_stats)
            cv2.putText(result, stats_text, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # 时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result, f"时间: {timestamp}", (15, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return result
    
    def _get_object_3d_bbox(self, object_name: str) -> Optional[np.ndarray]:
        """获取物体的3D包围盒角点"""
        # 基于物体类型定义3D包围盒
        bbox_definitions = {
            'can': np.array([
                [-25, -25, -60], [25, -25, -60], [25, 25, -60], [-25, 25, -60],  # 底面
                [-25, -25, 60], [25, -25, 60], [25, 25, 60], [-25, 25, 60]      # 顶面
            ]),
            'phone': np.array([
                [-30, -50, -5], [30, -50, -5], [30, 50, -5], [-30, 50, -5],     # 底面
                [-30, -50, 5], [30, -50, 5], [30, 50, 5], [-30, 50, 5]         # 顶面
            ]),
            'ape': np.array([
                [-35, -35, -35], [35, -35, -35], [35, 35, -35], [-35, 35, -35], # 底面
                [-35, -35, 35], [35, -35, 35], [35, 35, 35], [-35, 35, 35]     # 顶面
            ]),
            'lamp': np.array([
                [-40, -40, -70], [40, -40, -70], [40, 40, -70], [-40, 40, -70], # 底面
                [-30, -30, 70], [30, -30, 70], [30, 30, 70], [-30, 30, 70]     # 顶面
            ])
        }
        
        return bbox_definitions.get(object_name, bbox_definitions['can'])
    
    def _project_3d_bbox(self, bbox_3d: np.ndarray, pose: Dict, 
                        camera_K: np.ndarray) -> Optional[np.ndarray]:
        """投影3D包围盒到2D"""
        R = pose['rotation_matrix']
        t = pose['translation']
        
        # 变换3D点
        transformed_bbox = (R @ bbox_3d.T).T + t
        
        # 投影到2D
        projected = camera_K @ transformed_bbox.T
        projected = projected / projected[2, :]
        
        return projected[:2, :].T
    
    def _draw_projected_3d_bbox(self, image: np.ndarray, projected_bbox: np.ndarray, color: Tuple[int, int, int]):
        """绘制投影后的3D包围盒"""
        if len(projected_bbox) < 8:
            return
        
        # 立方体的12条边
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7),  # 竖直边
        ]
        
        for edge in edges:
            pt1 = tuple(map(int, projected_bbox[edge[0]]))
            pt2 = tuple(map(int, projected_bbox[edge[1]]))
            cv2.line(image, pt1, pt2, color, 3)
    
    def _draw_simple_3d_bbox(self, image: np.ndarray, pose: Dict, color: Tuple[int, int, int], index: int):
        """绘制简化的3D包围盒效果"""
        bbox = pose['bbox']
        x, y, w, h = bbox
        
        # 深度偏移
        depth_offset = 25 + index * 5
        
        # 前面矩形
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        
        # 后面矩形
        back_x1, back_y1 = x - depth_offset, y - depth_offset
        back_x2, back_y2 = x + w - depth_offset, y + h - depth_offset
        cv2.rectangle(image, (back_x1, back_y1), (back_x2, back_y2), color, 3)
        
        # 连接线构成3D效果
        cv2.line(image, (x, y), (back_x1, back_y1), color, 2)
        cv2.line(image, (x + w, y), (back_x2, back_y1), color, 2)
        cv2.line(image, (x, y + h), (back_x1, back_y2), color, 2)
        cv2.line(image, (x + w, y + h), (back_x2, back_y2), color, 2)
    
    def _draw_projected_axes(self, image: np.ndarray, pose: Dict, camera_K: np.ndarray,
                           axis_colors: Dict[str, Tuple[int, int, int]]):
        """绘制投影后的坐标轴"""
        R = pose['rotation_matrix']
        t = pose['translation']
        
        # 坐标轴点
        axis_length = 50.0
        axes_points = np.array([
            [0, 0, 0],                    # 原点
            [axis_length, 0, 0],          # X轴
            [0, axis_length, 0],          # Y轴
            [0, 0, axis_length],          # Z轴
        ], dtype=np.float32)
        
        # 变换和投影
        transformed_axes = (R @ axes_points.T).T + t
        projected_axes = camera_K @ transformed_axes.T
        projected_axes = projected_axes / projected_axes[2, :]
        projected_axes = projected_axes[:2, :].T
        
        if len(projected_axes) == 4:
            origin = tuple(map(int, projected_axes[0]))
            x_end = tuple(map(int, projected_axes[1]))
            y_end = tuple(map(int, projected_axes[2]))
            z_end = tuple(map(int, projected_axes[3]))
            
            # 绘制坐标轴
            cv2.arrowedLine(image, origin, x_end, axis_colors['x'], 4, tipLength=0.3)
            cv2.arrowedLine(image, origin, y_end, axis_colors['y'], 4, tipLength=0.3)
            cv2.arrowedLine(image, origin, z_end, axis_colors['z'], 4, tipLength=0.3)
    
    def _draw_simple_axes(self, image: np.ndarray, pose: Dict,
                         axis_colors: Dict[str, Tuple[int, int, int]]):
        """绘制简化的坐标轴"""
        bbox = pose['bbox']
        x, y, w, h = bbox
        center_x, center_y = x + w//2, y + h//2
        
        # 获取角度信息
        angles_deg = pose.get('angles_deg', [0, 0, 0])
        angle = np.radians(angles_deg[2])  # 使用Z轴旋转角度
        
        axis_length = 45
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # X轴 (红色)
        x_end_x = center_x + int(axis_length * cos_a)
        x_end_y = center_y + int(axis_length * sin_a * 0.3)
        cv2.arrowedLine(image, (center_x, center_y), (x_end_x, x_end_y), 
                       axis_colors['x'], 4, tipLength=0.3)
        
        # Y轴 (绿色)
        y_end_x = center_x + int(axis_length * sin_a * 0.3)
        y_end_y = center_y - int(axis_length * cos_a)
        cv2.arrowedLine(image, (center_x, center_y), (y_end_x, y_end_y), 
                       axis_colors['y'], 4, tipLength=0.3)
        
        # Z轴 (蓝色)
        z_end_x = center_x - int(axis_length * 0.6 * cos_a)
        z_end_y = center_y - int(axis_length * 0.6 * sin_a)
        cv2.arrowedLine(image, (center_x, center_y), (z_end_x, z_end_y), 
                       axis_colors['z'], 4, tipLength=0.3)
    
    def _create_3d_scene(self, poses: List[Dict], camera_K: Optional[np.ndarray]) -> Optional[o3d.geometry.TriangleMesh]:
        """创建3D场景"""
        try:
            # 创建3D场景
            scene_geometries = []
            
            # 添加坐标系
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.coordinate_frame_size * 1000
            )
            scene_geometries.append(coordinate_frame)
            
            # 为每个物体添加3D表示
            for i, pose in enumerate(poses):
                # 创建物体的3D表示
                object_mesh = self._create_object_mesh(pose)
                if object_mesh is not None:
                    scene_geometries.append(object_mesh)
                
                # 添加坐标轴
                object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=self.coordinate_frame_size * 500
                )
                
                # 应用变换
                R = pose['rotation_matrix']
                t = pose['translation'] / 1000  # 转换为米
                
                transform = np.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = t
                
                object_frame.transform(transform)
                scene_geometries.append(object_frame)
            
            return scene_geometries
            
        except Exception as e:
            self.logger.warning(f"⚠️ 3D场景创建失败: {e}")
            return None
    
    def _create_object_mesh(self, pose: Dict) -> Optional[o3d.geometry.TriangleMesh]:
        """为物体创建3D网格"""
        object_name = pose['object_name']
        
        try:
            # 根据物体类型创建不同的网格
            if object_name == 'can':
                mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.12)
                mesh.paint_uniform_color([0.8, 0.2, 0.2])  # 红色
            elif object_name == 'phone':
                mesh = o3d.geometry.TriangleMesh.create_box(width=0.06, height=0.12, depth=0.01)
                mesh.paint_uniform_color([0.2, 0.2, 0.8])  # 蓝色
            elif object_name == 'ape':
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                mesh.paint_uniform_color([0.8, 0.8, 0.2])  # 黄色
            elif object_name == 'lamp':
                mesh = o3d.geometry.TriangleMesh.create_cone(radius=0.04, height=0.15)
                mesh.paint_uniform_color([0.2, 0.8, 0.2])  # 绿色
            else:
                mesh = o3d.geometry.TriangleMesh.create_box(width=0.05, height=0.05, depth=0.05)
                mesh.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
            
            # 应用变换
            R = pose['rotation_matrix']
            t = pose['translation'] / 1000  # 转换为米
            
            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = t
            
            mesh.transform(transform)
            
            return mesh
            
        except Exception as e:
            self.logger.debug(f"物体网格创建失败: {e}")
            return None
    
    def _save_visualization_results(self, visualization_result: Dict):
        """保存可视化结果"""
        try:
            # 确保保存目录存在
            save_dir = Path("results")
            save_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存2D可视化图像
            if 'annotated_image' in visualization_result:
                img_filename = f"complete_6d_pose_{timestamp}.{self.save_format}"
                img_path = save_dir / img_filename
                
                if self.save_format.lower() == 'jpg':
                    cv2.imwrite(str(img_path), visualization_result['annotated_image'],
                               [cv2.IMWRITE_JPEG_QUALITY, self.save_quality])
                else:
                    cv2.imwrite(str(img_path), visualization_result['annotated_image'])
                
                self.logger.info(f"💾 2D可视化已保存: {img_path}")
            
            # 保存结构化数据
            data_filename = f"pose_data_{timestamp}.json"
            data_path = save_dir / data_filename
            
            # 准备JSON数据
            json_data = {
                'timestamp': timestamp,
                'pipeline': 'YOLO + Mask R-CNN + GDR-Net',
                'num_objects': len(visualization_result.get('detections', [])),
                'objects': []
            }
            
            detections = visualization_result.get('detections', [])
            masks = visualization_result.get('masks', [])
            poses = visualization_result.get('poses', [])
            
            for i, detection in enumerate(detections):
                obj_data = {
                    'id': i + 1,
                    'class_name': detection.get('class_name', 'unknown'),
                    'mapped_class': detection.get('mapped_class', 'unknown'),
                    'confidence': detection.get('confidence', 0.0),
                    'bbox': detection.get('bbox', []),
                    'has_mask': i < len(masks) and masks[i] is not None,
                    'has_pose': i < len(poses) and poses[i] is not None
                }
                
                if obj_data['has_pose'] and i < len(poses):
                    pose = poses[i]
                    obj_data['pose'] = {
                        'translation': pose['translation'].tolist(),
                        'rotation_matrix': pose['rotation_matrix'].tolist(),
                        'confidence': pose['confidence'],
                        'method': pose.get('method', 'unknown')
                    }
                
                json_data['objects'].append(obj_data)
            
            # 保存JSON文件
            import json
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 结构化数据已保存: {data_path}")
            
        except Exception as e:
            self.logger.error(f"❌ 保存可视化结果失败: {e}")
    
    def show_3d_scene(self, scene_geometries: List):
        """显示3D场景"""
        if not scene_geometries:
            self.logger.warning("⚠️ 无3D场景数据")
            return
        
        try:
            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="6D姿态估计 - 3D场景", width=800, height=600)
            
            # 添加几何体
            for geometry in scene_geometries:
                vis.add_geometry(geometry)
            
            # 设置视角
            view_control = vis.get_view_control()
            view_control.set_front([0, 0, -1])
            view_control.set_up([0, -1, 0])
            view_control.set_zoom(0.8)
            
            # 运行可视化
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            self.logger.error(f"❌ 3D场景显示失败: {e}")

def test_enhanced_3d_visualizer():
    """测试增强3D可视化器"""
    print("🧪 测试增强3D可视化器...")
    
    # 测试配置
    config = {
        'show_detection_boxes': True,
        'show_segmentation_masks': True,
        'show_pose_axes': True,
        'show_3d_bboxes': True,
        'show_info_panel': True,
        'save_results': True,
        'save_format': 'jpg'
    }
    
    # 创建可视化器
    visualizer = Enhanced3DVisualizer(config)
    
    # 创建测试数据
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 100
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
        cv2.ellipse(mask, (x + w//2, y + h//2), (w//2-5, h//2-5), 0, 0, 360, 255, -1)
        masks.append(mask)
    
    # 模拟姿态估计结果
    poses = [
        {
            'rotation_matrix': np.eye(3, dtype=np.float32),
            'translation': np.array([150, -50, 800], dtype=np.float32),
            'confidence': 0.85,
            'object_name': 'can',
            'bbox': [100, 100, 100, 100],
            'angles_deg': [10, -5, 15]
        },
        {
            'rotation_matrix': np.eye(3, dtype=np.float32),
            'translation': np.array([-120, -20, 750], dtype=np.float32),
            'confidence': 0.78,
            'object_name': 'phone',
            'bbox': [290, 90, 120, 120],
            'angles_deg': [-8, 12, -20]
        },
        {
            'rotation_matrix': np.eye(3, dtype=np.float32),
            'translation': np.array([20, 80, 650], dtype=np.float32),
            'confidence': 0.82,
            'object_name': 'ape',
            'bbox': [450, 220, 100, 160],
            'angles_deg': [5, -15, 8]
        }
    ]
    
    # 相机内参
    camera_K = np.array([
        [572.4, 0, 320],
        [0, 573.6, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 创建完整可视化
    result = visualizer.create_complete_visualization(
        test_image, detections, masks, poses, camera_K
    )
    
    print(f"✅ 可视化创建完成")
    
    # 显示结果
    if 'annotated_image' in result:
        cv2.imshow("完整6D姿态估计可视化", result['annotated_image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 显示3D场景
    if result.get('scene_3d'):
        visualizer.show_3d_scene(result['scene_3d'])
    
    print("🎉 增强3D可视化器测试完成!")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_enhanced_3d_visualizer()
