#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LineMOD专用检测器 - 备用方案
使用传统计算机视觉方法检测LineMOD物体
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

class LineMODFallbackDetector:
    """LineMOD备用检测器 - 基于传统CV方法"""
    
    def __init__(self, config: Dict = None):
        """
        初始化备用检测器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # LineMOD物体的视觉特征配置
        self.objects_config = {
            'ape': {
                'color_ranges': [([10, 50, 50], [30, 255, 255])],  # 棕色/黄色
                'min_area': 1000,
                'max_area': 50000,
                'aspect_ratio_range': (0.7, 1.5),
                'shape_type': 'complex'
            },
            'can': {
                'color_ranges': [([100, 50, 50], [130, 255, 255]), ([0, 50, 50], [10, 255, 255])],  # 蓝色/红色
                'min_area': 2000,
                'max_area': 60000,
                'aspect_ratio_range': (0.4, 0.8),  # 高瘦形状
                'shape_type': 'cylindrical'
            },
            'phone': {
                'color_ranges': [([0, 0, 50], [180, 50, 200])],  # 深色/黑色
                'min_area': 1500,
                'max_area': 40000,
                'aspect_ratio_range': (0.4, 0.7),  # 矩形
                'shape_type': 'rectangular'
            },
            'lamp': {
                'color_ranges': [([20, 50, 50], [40, 255, 255])],  # 黄色/金色
                'min_area': 3000,
                'max_area': 80000,
                'aspect_ratio_range': (0.6, 1.2),
                'shape_type': 'complex'
            },
            'cat': {
                'color_ranges': [([5, 50, 50], [25, 255, 255])],  # 橙色/棕色
                'min_area': 1200,
                'max_area': 45000,
                'aspect_ratio_range': (0.8, 1.3),
                'shape_type': 'organic'
            },
            'benchvise': {
                'color_ranges': [([0, 0, 100], [180, 50, 255])],  # 金属色/灰色
                'min_area': 4000,
                'max_area': 100000,
                'aspect_ratio_range': (0.5, 1.5),
                'shape_type': 'mechanical'
            },
            'cam': {
                'color_ranges': [([0, 0, 50], [180, 100, 200])],  # 深色
                'min_area': 2000,
                'max_area': 50000,
                'aspect_ratio_range': (0.8, 1.5),
                'shape_type': 'rectangular'
            },
            'driller': {
                'color_ranges': [([20, 100, 100], [40, 255, 255])],  # 黄色
                'min_area': 3000,
                'max_area': 70000,
                'aspect_ratio_range': (0.3, 0.8),
                'shape_type': 'tool'
            },
            'duck': {
                'color_ranges': [([20, 100, 100], [30, 255, 255])],  # 黄色
                'min_area': 800,
                'max_area': 30000,
                'aspect_ratio_range': (0.8, 1.3),
                'shape_type': 'organic'
            },
            'eggbox': {
                'color_ranges': [([0, 0, 150], [180, 50, 255])],  # 浅色
                'min_area': 2000,
                'max_area': 60000,
                'aspect_ratio_range': (0.8, 1.2),
                'shape_type': 'rectangular'
            },
            'glue': {
                'color_ranges': [([0, 0, 200], [180, 30, 255])],  # 白色
                'min_area': 1000,
                'max_area': 30000,
                'aspect_ratio_range': (0.3, 0.7),
                'shape_type': 'cylindrical'
            },
            'holepuncher': {
                'color_ranges': [([0, 0, 50], [180, 100, 150])],  # 深色
                'min_area': 2000,
                'max_area': 50000,
                'aspect_ratio_range': (0.4, 0.8),
                'shape_type': 'mechanical'
            },
            'iron': {
                'color_ranges': [([100, 50, 50], [130, 255, 255])],  # 蓝色
                'min_area': 2500,
                'max_area': 60000,
                'aspect_ratio_range': (0.6, 1.2),
                'shape_type': 'appliance'
            }
        }
        
        self.logger.info("✅ LineMOD备用检测器初始化完成")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        使用传统CV方法检测LineMOD物体
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表
        """
        if image is None or image.size == 0:
            return []
        
        detections = []
        
        try:
            # 预处理
            processed_image = self._preprocess_image(image)
            
            # 对每个物体类别进行检测
            for obj_name, obj_config in self.objects_config.items():
                obj_detections = self._detect_object_type(processed_image, obj_name, obj_config)
                detections.extend(obj_detections)
            
            # 后处理：非最大抑制
            detections = self._apply_nms(detections)
            
            self.logger.info(f"🔍 备用检测完成: 发现 {len(detections)} 个物体")
            
        except Exception as e:
            self.logger.error(f"❌ 备用检测失败: {e}")
        
        return detections
    
    def _preprocess_image(self, image: np.ndarray) -> Dict:
        """预处理图像"""
        # 转换颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯滤波降噪
        hsv_blur = cv2.GaussianBlur(hsv, (5, 5), 0)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return {
            'original': image,
            'hsv': hsv_blur,
            'gray': gray_blur
        }
    
    def _detect_object_type(self, processed_image: Dict, obj_name: str, obj_config: Dict) -> List[Dict]:
        """检测特定类型的物体"""
        detections = []
        
        # 颜色检测
        color_mask = self._create_color_mask(processed_image['hsv'], obj_config['color_ranges'])
        
        # 形态学操作
        color_mask = self._apply_morphology(color_mask, obj_config['shape_type'])
        
        # 查找轮廓
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            detection = self._analyze_contour(contour, obj_name, obj_config, processed_image)
            if detection:
                detections.append(detection)
        
        return detections
    
    def _create_color_mask(self, hsv_image: np.ndarray, color_ranges: List[Tuple]) -> np.ndarray:
        """创建颜色掩码"""
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            range_mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, range_mask)
        
        return mask
    
    def _apply_morphology(self, mask: np.ndarray, shape_type: str) -> np.ndarray:
        """应用形态学操作"""
        if shape_type == 'cylindrical':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        elif shape_type == 'rectangular':
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 闭运算填充空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 开运算去除噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _analyze_contour(self, contour: np.ndarray, obj_name: str, obj_config: Dict, 
                        processed_image: Dict) -> Dict:
        """分析轮廓并生成检测结果"""
        # 计算面积
        area = cv2.contourArea(contour)
        if area < obj_config['min_area'] or area > obj_config['max_area']:
            return None
        
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 检查长宽比
        aspect_ratio = w / h if h > 0 else 0
        min_ratio, max_ratio = obj_config['aspect_ratio_range']
        if not (min_ratio <= aspect_ratio <= max_ratio):
            return None
        
        # 计算置信度
        confidence = self._calculate_confidence(contour, obj_config, processed_image, (x, y, w, h))
        
        if confidence < 0.3:  # 最低置信度阈值
            return None
        
        return {
            'bbox': [x, y, w, h],
            'class_name': obj_name,
            'mapped_class': obj_name,
            'confidence': confidence,
            'detection_id': 0,  # 将在后处理中重新分配
            'area': area,
            'aspect_ratio': aspect_ratio
        }
    
    def _calculate_confidence(self, contour: np.ndarray, obj_config: Dict, 
                            processed_image: Dict, bbox: Tuple[int, int, int, int]) -> float:
        """计算检测置信度"""
        x, y, w, h = bbox
        
        # 基础置信度（基于面积）
        area = cv2.contourArea(contour)
        area_score = min(1.0, area / obj_config['max_area'])
        
        # 形状置信度（基于轮廓复杂度）
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            shape_score = min(1.0, compactness * 2)
        else:
            shape_score = 0.5
        
        # 边缘强度置信度
        roi_gray = processed_image['gray'][y:y+h, x:x+w]
        if roi_gray.size > 0:
            edges = cv2.Canny(roi_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(1.0, edge_density * 5)
        else:
            edge_score = 0.5
        
        # 综合置信度
        confidence = (area_score * 0.4 + shape_score * 0.3 + edge_score * 0.3)
        
        # 添加随机性以模拟真实检测器的不确定性
        confidence += np.random.uniform(-0.05, 0.05)
        
        return max(0.0, min(0.95, confidence))
    
    def _apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """应用非最大抑制"""
        if not detections:
            return detections
        
        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 应用NMS
        keep = []
        while detections:
            # 保留置信度最高的检测
            best = detections.pop(0)
            keep.append(best)
            
            # 移除与当前检测重叠度过高的其他检测
            detections = [det for det in detections 
                         if self._calculate_iou(best['bbox'], det['bbox']) < iou_threshold]
        
        # 重新分配ID
        for i, detection in enumerate(keep):
            detection['detection_id'] = i
        
        return keep
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """计算两个边界框的IoU"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # 计算交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """可视化检测结果"""
        result = image.copy()
        
        colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
        ]
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x, y, w, h = bbox
            obj_name = detection['class_name']
            confidence = detection['confidence']
            
            color = colors[i % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签
            label = f"{obj_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 标签背景
            cv2.rectangle(result, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # 标签文字
            cv2.putText(result, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result

def test_fallback_detector():
    """测试备用检测器"""
    print("🧪 测试LineMOD备用检测器...")
    
    detector = LineMODFallbackDetector()
    
    # 创建测试图像
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    # 添加一些彩色区域模拟物体
    cv2.rectangle(test_image, (100, 100), (200, 250), (120, 180, 100), -1)  # 棕色 - ape
    cv2.rectangle(test_image, (300, 80), (380, 280), (200, 100, 50), -1)    # 蓝色 - can
    cv2.rectangle(test_image, (450, 150), (550, 220), (50, 50, 50), -1)     # 深色 - phone
    
    # 执行检测
    detections = detector.detect(test_image)
    
    print(f"✅ 检测结果: {len(detections)} 个物体")
    for det in detections:
        print(f"   {det['class_name']}: {det['confidence']:.3f}")
    
    # 可视化
    if detections:
        vis_image = detector.visualize_detections(test_image, detections)
        cv2.imshow("LineMOD备用检测器测试", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("🎉 备用检测器测试完成!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_fallback_detector()
