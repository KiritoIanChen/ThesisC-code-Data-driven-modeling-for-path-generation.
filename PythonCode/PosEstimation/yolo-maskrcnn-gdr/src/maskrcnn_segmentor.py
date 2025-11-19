#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask R-CNN实例分割器
实现基于Detectron2的实例分割功能
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union
import logging
from pathlib import Path

# 尝试导入Detectron2
try:
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False
    print("⚠️ Detectron2未安装，将使用简化的分割实现")

class MaskRCNNSegmentor:
    """Mask R-CNN实例分割器"""
    
    def __init__(self, config: Dict):
        """
        初始化Mask R-CNN分割器
        
        Args:
            config: 分割配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设备配置
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 分割参数
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.mask_threshold = config.get('mask_threshold', 0.5)
        self.use_detection_prompts = config.get('use_detection_prompts', True)
        
        # 后处理参数
        self.min_mask_area = config.get('min_mask_area', 100)
        self.max_mask_area = config.get('max_mask_area', 50000)
        self.smooth_mask = config.get('smooth_mask', True)
        
        # 初始化模型
        self.predictor = self._load_model()
        
        self.logger.info(f"✅ Mask R-CNN分割器初始化完成 (设备: {self.device})")
    
    def _load_model(self):
        """加载Mask R-CNN模型"""
        if not DETECTRON2_AVAILABLE:
            self.logger.warning("⚠️ Detectron2不可用，使用简化分割实现")
            return None
        
        try:
            # 配置模型
            cfg = get_cfg()
            
            # 加载预训练配置
            model_name = self.config.get('model_name', 
                                       "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            cfg.merge_from_file(model_zoo.get_config_file(model_name))
            
            # 设置模型权重
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
            
            # 设置置信度阈值
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            
            # 设置设备
            cfg.MODEL.DEVICE = self.device
            
            # 创建预测器
            from detectron2.engine import DefaultPredictor
            predictor = DefaultPredictor(cfg)
            
            self.logger.info(f"📦 Mask R-CNN模型加载成功: {model_name}")
            return predictor
            
        except Exception as e:
            self.logger.error(f"❌ Mask R-CNN模型加载失败: {e}")
            return None
    
    def segment(self, image: np.ndarray, detections: Optional[List[Dict]] = None) -> List[np.ndarray]:
        """
        执行实例分割
        
        Args:
            image: 输入图像 (BGR格式)
            detections: 可选的检测结果，用作分割提示
            
        Returns:
            分割掩码列表，每个掩码对应一个物体实例
        """
        if image is None or image.size == 0:
            self.logger.warning("⚠️ 输入图像为空")
            return []
        
        try:
            if self.predictor is not None:
                # 使用真实的Mask R-CNN
                masks = self._segment_with_maskrcnn(image, detections)
            else:
                # 使用简化的分割实现
                masks = self._segment_simplified(image, detections)
            
            # 后处理
            masks = self._post_process_masks(masks, image.shape[:2])
            
            self.logger.info(f"🎯 分割完成: 生成 {len(masks)} 个掩码")
            
            return masks
            
        except Exception as e:
            self.logger.error(f"❌ 分割失败: {e}")
            return []
    
    def _segment_with_maskrcnn(self, image: np.ndarray, 
                              detections: Optional[List[Dict]] = None) -> List[np.ndarray]:
        """使用真实的Mask R-CNN进行分割"""
        # 执行推理
        outputs = self.predictor(image)
        
        masks = []
        
        if "instances" in outputs:
            instances = outputs["instances"]
            
            if len(instances) > 0:
                # 获取预测掩码
                pred_masks = instances.pred_masks.cpu().numpy()
                pred_classes = instances.pred_classes.cpu().numpy()
                pred_scores = instances.scores.cpu().numpy()
                
                # 如果有检测提示，尝试匹配
                if detections and self.use_detection_prompts:
                    masks = self._match_masks_to_detections(
                        pred_masks, pred_classes, pred_scores, detections, image.shape[:2]
                    )
                else:
                    # 直接使用所有预测掩码
                    for mask in pred_masks:
                        masks.append(mask.astype(np.uint8) * 255)
        
        return masks
    
    def _segment_simplified(self, image: np.ndarray, 
                           detections: Optional[List[Dict]] = None) -> List[np.ndarray]:
        """简化的分割实现（基于检测框生成掩码）"""
        masks = []
        
        if detections is None:
            self.logger.warning("⚠️ 无检测结果，无法生成分割掩码")
            return masks
        
        for detection in detections:
            mask = self._create_mask_from_detection(image, detection)
            if mask is not None:
                masks.append(mask)
        
        return masks
    
    def _create_mask_from_detection(self, image: np.ndarray, detection: Dict) -> Optional[np.ndarray]:
        """基于检测结果创建分割掩码"""
        bbox = detection['bbox']
        x, y, w, h = bbox
        class_name = detection.get('mapped_class', detection.get('class_name', 'unknown'))
        
        # 创建掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 获取ROI
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        
        # 根据物体类型生成不同形状的掩码
        center_x, center_y = x + w//2, y + h//2
        
        if 'can' in class_name.lower():
            # 圆柱形物体 - 椭圆掩码
            cv2.ellipse(mask, (center_x, center_y), (w//2-5, h//2-5), 0, 0, 360, 255, -1)
        elif 'phone' in class_name.lower():
            # 矩形物体 - 带圆角的矩形
            cv2.rectangle(mask, (x+3, y+3), (x+w-3, y+h-3), 255, -1)
            # 添加圆角效果
            corner_radius = min(w, h) // 10
            if corner_radius > 0:
                self._add_rounded_corners(mask, (x+3, y+3, w-6, h-6), corner_radius)
        elif 'ape' in class_name.lower() or 'cat' in class_name.lower():
            # 生物形状 - 不规则椭圆
            cv2.ellipse(mask, (center_x, center_y), (w//2-3, h//2-3), 0, 0, 360, 255, -1)
            # 添加一些变形
            self._add_organic_deformation(mask, (x, y, w, h))
        elif 'lamp' in class_name.lower():
            # 灯具 - 梯形掩码
            self._create_trapezoid_mask(mask, (x, y, w, h))
        else:
            # 默认矩形掩码
            cv2.rectangle(mask, (x+2, y+2), (x+w-2, y+h-2), 255, -1)
        
        # 使用颜色信息优化掩码
        mask = self._refine_mask_with_color(image, mask, (x, y, w, h))
        
        return mask
    
    def _add_rounded_corners(self, mask: np.ndarray, rect: Tuple[int, int, int, int], radius: int):
        """为矩形掩码添加圆角"""
        x, y, w, h = rect
        
        # 四个角的圆形掩码
        corners = [
            (x, y),                    # 左上
            (x + w - radius, y),       # 右上
            (x, y + h - radius),       # 左下
            (x + w - radius, y + h - radius)  # 右下
        ]
        
        for corner_x, corner_y in corners:
            cv2.circle(mask, (corner_x + radius//2, corner_y + radius//2), radius//2, 255, -1)
    
    def _add_organic_deformation(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]):
        """为生物形状添加有机变形"""
        x, y, w, h = bbox
        
        # 添加一些随机的小椭圆来模拟有机形状
        num_deformations = 3
        for i in range(num_deformations):
            # 随机位置和大小
            dx = np.random.randint(-w//4, w//4)
            dy = np.random.randint(-h//4, h//4)
            dw = np.random.randint(w//8, w//4)
            dh = np.random.randint(h//8, h//4)
            
            center = (x + w//2 + dx, y + h//2 + dy)
            axes = (dw, dh)
            
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    def _create_trapezoid_mask(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]):
        """创建梯形掩码（用于灯具等物体）"""
        x, y, w, h = bbox
        
        # 梯形的四个顶点
        top_width = int(w * 0.6)
        bottom_width = w
        
        top_left = (x + (w - top_width)//2, y)
        top_right = (x + (w + top_width)//2, y)
        bottom_left = (x, y + h)
        bottom_right = (x + w, y + h)
        
        points = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
        cv2.fillPoly(mask, [points], 255)
    
    def _refine_mask_with_color(self, image: np.ndarray, mask: np.ndarray, 
                               bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """使用颜色信息优化掩码"""
        if not self.smooth_mask:
            return mask
        
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]
        
        if roi.size == 0 or roi_mask.size == 0:
            return mask
        
        # 使用GrabCut算法优化掩码
        try:
            # 创建GrabCut掩码
            gc_mask = np.zeros(roi.shape[:2], np.uint8)
            gc_mask[roi_mask > 0] = cv2.GC_PR_FGD  # 可能前景
            gc_mask[roi_mask == 0] = cv2.GC_PR_BGD  # 可能背景
            
            # 设置确定前景区域（掩码中心区域）
            center_margin = min(w, h) // 4
            if center_margin > 0:
                gc_mask[center_margin:-center_margin, center_margin:-center_margin] = cv2.GC_FGD
            
            # 应用GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            cv2.grabCut(roi, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            
            # 提取前景
            refined_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            
            # 更新原掩码
            mask[y:y+h, x:x+w] = refined_mask
            
        except Exception as e:
            self.logger.debug(f"颜色优化失败，使用原始掩码: {e}")
        
        return mask
    
    def _match_masks_to_detections(self, pred_masks: np.ndarray, pred_classes: np.ndarray,
                                  pred_scores: np.ndarray, detections: List[Dict],
                                  image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """将预测掩码匹配到检测结果"""
        matched_masks = []
        
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = bbox
            
            best_mask = None
            best_iou = 0.0
            
            # 为每个检测框找到最佳匹配的掩码
            for i, (mask, score) in enumerate(zip(pred_masks, pred_scores)):
                # 计算掩码与检测框的IoU
                mask_bbox = self._get_mask_bbox(mask)
                if mask_bbox is not None:
                    iou = self._calculate_bbox_iou(bbox, mask_bbox)
                    if iou > best_iou and score > self.confidence_threshold:
                        best_iou = iou
                        best_mask = mask
            
            if best_mask is not None and best_iou > 0.3:  # IoU阈值
                matched_masks.append((best_mask * 255).astype(np.uint8))
            else:
                # 如果没有好的匹配，创建简化掩码
                simple_mask = self._create_mask_from_detection(
                    np.zeros((*image_shape, 3), dtype=np.uint8), detection
                )
                if simple_mask is not None:
                    matched_masks.append(simple_mask)
        
        return matched_masks
    
    def _get_mask_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """从掩码获取边界框"""
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _calculate_bbox_iou(self, bbox1: List[int], bbox2: Tuple[int, int, int, int]) -> float:
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
    
    def _post_process_masks(self, masks: List[np.ndarray], 
                           image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """后处理掩码"""
        processed_masks = []
        
        for mask in masks:
            if mask is None:
                continue
            
            # 确保掩码尺寸正确
            if mask.shape != image_shape:
                mask = cv2.resize(mask, (image_shape[1], image_shape[0]))
            
            # 面积过滤
            mask_area = np.sum(mask > 0)
            if mask_area < self.min_mask_area or mask_area > self.max_mask_area:
                continue
            
            # 形态学操作
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 平滑边缘
            if self.smooth_mask:
                mask = cv2.GaussianBlur(mask, (3, 3), 0)
                mask = (mask > 127).astype(np.uint8) * 255
            
            processed_masks.append(mask)
        
        return processed_masks
    
    def visualize_masks(self, image: np.ndarray, masks: List[np.ndarray], 
                       detections: Optional[List[Dict]] = None) -> np.ndarray:
        """
        可视化分割掩码
        
        Args:
            image: 原始图像
            masks: 分割掩码列表
            detections: 可选的检测结果
            
        Returns:
            标注后的图像
        """
        result_image = image.copy()
        
        # 分割掩码颜色
        colors = [
            (0, 255, 255),    # 黄色
            (255, 0, 255),    # 品红
            (255, 255, 0),    # 青色
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 蓝色
            (128, 255, 128),  # 浅绿
            (255, 128, 128),  # 浅红
            (128, 128, 255),  # 浅蓝
        ]
        
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            
            # 找到轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 绘制轮廓
                cv2.drawContours(result_image, contours, -1, color, 3)
                
                # 半透明填充
                overlay = result_image.copy()
                cv2.fillPoly(overlay, contours, color)
                cv2.addWeighted(overlay, 0.2, result_image, 0.8, 0, result_image)
                
                # 添加ID标签
                mask_center = self._get_mask_center(mask)
                if mask_center:
                    cv2.circle(result_image, mask_center, 15, color, -1)
                    cv2.putText(result_image, f"#{i+1}", 
                               (mask_center[0]-8, mask_center[1]+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def _get_mask_center(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """获取掩码中心点"""
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx, cy)
        return None
    
    def get_mask_statistics(self, masks: List[np.ndarray]) -> Dict:
        """获取掩码统计信息"""
        if not masks:
            return {'count': 0}
        
        areas = [np.sum(mask > 0) for mask in masks]
        
        return {
            'count': len(masks),
            'total_area': sum(areas),
            'average_area': np.mean(areas),
            'min_area': min(areas),
            'max_area': max(areas),
            'area_std': np.std(areas)
        }

def test_maskrcnn_segmentor():
    """测试Mask R-CNN分割器"""
    print("🧪 测试Mask R-CNN分割器...")
    
    # 测试配置
    config = {
        'confidence_threshold': 0.7,
        'mask_threshold': 0.5,
        'device': 'cpu',  # 测试时使用CPU
        'use_detection_prompts': True,
        'smooth_mask': True
    }
    
    # 创建分割器
    segmentor = MaskRCNNSegmentor(config)
    
    # 创建测试图像
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    # 添加一些模拟物体
    cv2.rectangle(test_image, (100, 100), (200, 200), (120, 140, 160), -1)
    cv2.circle(test_image, (350, 150), 60, (160, 120, 100), -1)
    cv2.ellipse(test_image, (500, 300), (50, 80), 0, 0, 360, (100, 160, 140), -1)
    
    # 模拟检测结果
    detections = [
        {'bbox': [100, 100, 100, 100], 'class_name': 'cup', 'mapped_class': 'can', 'confidence': 0.9},
        {'bbox': [290, 90, 120, 120], 'class_name': 'phone', 'mapped_class': 'phone', 'confidence': 0.8},
        {'bbox': [450, 220, 100, 160], 'class_name': 'person', 'mapped_class': 'ape', 'confidence': 0.85}
    ]
    
    # 执行分割
    masks = segmentor.segment(test_image, detections)
    
    print(f"✅ 分割结果: {len(masks)} 个掩码")
    
    # 获取统计信息
    stats = segmentor.get_mask_statistics(masks)
    print(f"📊 掩码统计: {stats}")
    
    # 可视化
    if masks:
        vis_image = segmentor.visualize_masks(test_image, masks, detections)
        cv2.imshow("Mask R-CNN分割结果", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("🎉 Mask R-CNN分割器测试完成!")

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_maskrcnn_segmentor()
