#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO目标检测器
实现基于YOLOv8的目标检测功能
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

class YOLODetector:
    """YOLO目标检测器"""
    
    def __init__(self, config: Dict):
        """
        初始化YOLO检测器
        
        Args:
            config: 检测配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设备配置
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 检测参数
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
        self.max_detections = config.get('max_detections', 10)
        self.input_size = config.get('input_size', [640, 640])
        
        # 类别映射 (YOLO类别ID -> LineMOD物体名称)
        self.class_mapping = config.get('class_mapping', {})
        
        # 初始化模型
        self.model = self._load_model()
        
        self.logger.info(f"✅ YOLO检测器初始化完成 (设备: {self.device})")
    
    def _load_model(self) -> YOLO:
        """加载YOLO模型"""
        model_name = self.config.get('model_name', 'yolov8n.pt')
        
        try:
            # 加载预训练模型
            model = YOLO(model_name)
            
            # 移动到指定设备
            if self.device == 'cuda' and torch.cuda.is_available():
                model.to('cuda')
            
            self.logger.info(f"📦 YOLO模型加载成功: {model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ YOLO模型加载失败: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        执行目标检测
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个元素包含:
            - bbox: [x, y, w, h] 边界框
            - class_name: 物体类别名称
            - class_id: 原始类别ID
            - confidence: 置信度
            - mapped_class: 映射后的LineMOD类别
        """
        if image is None or image.size == 0:
            self.logger.warning("⚠️ 输入图像为空")
            return []
        
        try:
            # YOLO推理
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                imgsz=self.input_size,
                verbose=False
            )
            
            # 解析检测结果
            detections = self._parse_results(results[0], image.shape)
            
            self.logger.info(f"🔍 YOLO检测完成: 发现 {len(detections)} 个物体")
            
            return detections
            
        except Exception as e:
            self.logger.error(f"❌ YOLO检测失败: {e}")
            return []
    
    def _parse_results(self, result, image_shape: Tuple[int, int, int]) -> List[Dict]:
        """解析YOLO检测结果"""
        detections = []
        
        if result.boxes is None:
            return detections
        
        # 获取检测框、置信度和类别
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # 获取类别名称
        class_names = result.names
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            # 转换边界框格式 [x1, y1, x2, y2] -> [x, y, w, h]
            x1, y1, x2, y2 = box
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            
            # 边界检查
            x = max(0, min(x, image_shape[1] - 1))
            y = max(0, min(y, image_shape[0] - 1))
            w = max(1, min(w, image_shape[1] - x))
            h = max(1, min(h, image_shape[0] - y))
            
            # 获取类别信息
            class_name = class_names.get(class_id, f"class_{class_id}")
            mapped_class = self._map_class(class_id, class_name)
            
            detection = {
                'bbox': [x, y, w, h],
                'class_name': class_name,
                'class_id': int(class_id),
                'confidence': float(conf),
                'mapped_class': mapped_class,
                'detection_id': i
            }
            
            detections.append(detection)
            
            self.logger.debug(f"   检测 #{i+1}: {class_name} -> {mapped_class} ({conf:.3f})")
        
        return detections
    
    def _map_class(self, class_id: int, class_name: str) -> Optional[str]:
        """
        将YOLO类别映射到LineMOD物体
        
        Args:
            class_id: YOLO类别ID
            class_name: YOLO类别名称
            
        Returns:
            映射后的LineMOD物体名称，如果无法映射则返回None
        """
        # 优先使用配置中的映射
        if class_id in self.class_mapping:
            return self.class_mapping[class_id]
        
        # 基于类别名称的启发式映射
        name_lower = class_name.lower()
        
        if any(keyword in name_lower for keyword in ['cup', 'bottle', 'wine glass']):
            return 'can'
        elif any(keyword in name_lower for keyword in ['phone', 'cell phone', 'remote', 'keyboard']):
            return 'phone'  
        elif any(keyword in name_lower for keyword in ['tv', 'monitor', 'laptop']):
            return 'lamp'
        elif 'mouse' in name_lower:
            return 'mouse'
        elif any(keyword in name_lower for keyword in ['person', 'cat', 'dog']):
            return 'ape'  # 生物类映射到ape
        
        # 无法映射
        self.logger.debug(f"⚠️ 无法映射类别: {class_name} (ID: {class_id})")
        return None
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            
        Returns:
            标注后的图像
        """
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            mapped_class = detection['mapped_class']
            confidence = detection['confidence']
            
            x, y, w, h = bbox
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            
            # 绘制标签
            label = f"{class_name}"
            if mapped_class:
                label += f" -> {mapped_class}"
            label += f" ({confidence:.2f})"
            
            # 标签背景
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # 标签文字
            cv2.putText(result_image, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def get_supported_classes(self) -> List[str]:
        """获取支持的LineMOD类别列表"""
        supported = set(self.class_mapping.values())
        supported.update(['can', 'phone', 'lamp', 'mouse', 'ape'])  # 启发式映射的类别
        return list(supported)
    
    def update_config(self, new_config: Dict):
        """更新配置"""
        self.config.update(new_config)
        
        # 更新相关参数
        self.confidence_threshold = self.config.get('confidence_threshold', self.confidence_threshold)
        self.iou_threshold = self.config.get('iou_threshold', self.iou_threshold)
        self.max_detections = self.config.get('max_detections', self.max_detections)
        
        self.logger.info("🔄 YOLO检测器配置已更新")

def test_yolo_detector():
    """测试YOLO检测器"""
    print("🧪 测试YOLO检测器...")
    
    # 测试配置
    config = {
        'model_name': 'yolov8n.pt',
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'device': 'cpu',  # 测试时使用CPU
        'class_mapping': {
            41: 'can',      # cup -> can
            67: 'phone',    # cell phone -> phone
            62: 'lamp',     # tv -> lamp
        }
    }
    
    # 创建检测器
    detector = YOLODetector(config)
    
    # 创建测试图像
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # 添加一些模拟物体
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.rectangle(test_image, (300, 150), (400, 250), (255, 0, 0), -1)
    cv2.circle(test_image, (500, 300), 50, (0, 0, 255), -1)
    
    # 执行检测
    detections = detector.detect(test_image)
    
    print(f"✅ 检测结果: {len(detections)} 个物体")
    for i, det in enumerate(detections):
        print(f"   #{i+1}: {det['class_name']} -> {det['mapped_class']} ({det['confidence']:.3f})")
    
    # 可视化
    if detections:
        vis_image = detector.visualize_detections(test_image, detections)
        cv2.imshow("YOLO检测结果", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("🎉 YOLO检测器测试完成!")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_yolo_detector()
