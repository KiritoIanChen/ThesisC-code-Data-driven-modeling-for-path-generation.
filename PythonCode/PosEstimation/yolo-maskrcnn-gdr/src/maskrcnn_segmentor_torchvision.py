#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask R-CNN分割器 (使用PyTorch TorchVision)
基于torchvision.models.detection实现，无需安装Detectron2
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MaskRCNNSegmentor:
    """Mask R-CNN分割器 (TorchVision实现)"""
    
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
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.mask_threshold = config.get('mask_threshold', 0.5)
        self.use_detection_prompts = config.get('use_detection_prompts', True)
        
        # 后处理参数
        self.min_mask_area = config.get('min_mask_area', 100)
        self.max_mask_area = config.get('max_mask_area', 50000)
        self.smooth_mask = config.get('smooth_mask', True)
        
        # 初始化模型
        self.model = self._load_model()
        
        self.logger.info(f"✅ Mask R-CNN分割器初始化完成 (TorchVision, 设备: {self.device})")
    
    def _load_model(self):
        """加载Mask R-CNN模型"""
        try:
            # 使用预训练的Mask R-CNN模型
            self.logger.info("📦 加载TorchVision Mask R-CNN模型...")
            
            # 加载预训练权重
            weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            model = maskrcnn_resnet50_fpn(weights=weights)
            
            # 设置为评估模式
            model.eval()
            model.to(self.device)
            
            self.logger.info(f"   ✓ Mask R-CNN模型加载成功 (ResNet-50 FPN)")
            return model
            
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
        
        if self.model is None:
            self.logger.warning("⚠️ 模型未加载，使用简化分割")
            return self._segment_simplified(image, detections)
        
        try:
            # 使用TorchVision Mask R-CNN
            masks = self._segment_with_maskrcnn(image, detections)
            
            # 后处理
            masks = self._post_process_masks(masks, image.shape[:2])
            
            self.logger.info(f"🎯 分割完成: 生成 {len(masks)} 个掩码")
            
            return masks
            
        except Exception as e:
            self.logger.error(f"❌ 分割失败: {e}")
            return []
    
    def _segment_with_maskrcnn(self, image: np.ndarray, 
                              detections: Optional[List[Dict]] = None) -> List[np.ndarray]:
        """使用TorchVision Mask R-CNN进行分割"""
        
        # 1. 预处理图像
        # OpenCV是BGR，需要转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为tensor并归一化到[0, 1]
        image_tensor = torch.from_numpy(rgb_image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # 添加batch维度
        
        # 2. 模型推理
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # 3. 处理输出
        output = outputs[0]
        masks = []
        
        if len(output['masks']) > 0:
            pred_masks = output['masks'].cpu().numpy()  # [N, 1, H, W]
            pred_scores = output['scores'].cpu().numpy()
            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            
            # 如果有检测提示，尝试匹配
            if detections and self.use_detection_prompts:
                masks = self._match_masks_to_detections(
                    pred_masks, pred_boxes, pred_scores, pred_labels, 
                    detections, image.shape[:2]
                )
            else:
                # 直接使用所有高置信度的掩码
                for i, score in enumerate(pred_scores):
                    if score > self.confidence_threshold:
                        # 提取掩码 [1, H, W] -> [H, W]
                        mask = pred_masks[i, 0]
                        # 二值化
                        mask = (mask > self.mask_threshold).astype(np.uint8) * 255
                        masks.append(mask)
        
        return masks
    
    def _match_masks_to_detections(self, pred_masks: np.ndarray, pred_boxes: np.ndarray,
                                  pred_scores: np.ndarray, pred_labels: np.ndarray,
                                  detections: List[Dict],
                                  image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """将预测掩码匹配到YOLO检测结果"""
        matched_masks = []
        
        for detection in detections:
            det_bbox = detection['bbox']  # [x, y, w, h]
            det_x1, det_y1, det_w, det_h = det_bbox
            det_x2, det_y2 = det_x1 + det_w, det_y1 + det_h
            
            best_mask = None
            best_iou = 0.0
            best_score = 0.0
            
            # 为每个检测框找到最佳匹配的掩码
            for i, (mask, box, score) in enumerate(zip(pred_masks, pred_boxes, pred_scores)):
                # 降低置信度要求，使用更宽松的阈值
                if score < 0.1:  # 极低的阈值
                    continue
                
                # 计算IoU
                pred_x1, pred_y1, pred_x2, pred_y2 = box
                iou = self._calculate_bbox_iou(
                    [det_x1, det_y1, det_w, det_h],
                    [pred_x1, pred_y1, pred_x2 - pred_x1, pred_y2 - pred_y1]
                )
                
                if iou > best_iou:
                    best_iou = iou
                    best_mask = mask[0]  # [1, H, W] -> [H, W]
                    best_score = score
            
            # 降低IoU阈值，更容易匹配
            if best_mask is not None and best_iou > 0.2:  # 降低到0.2
                # 二值化掩码
                binary_mask = (best_mask > self.mask_threshold).astype(np.uint8) * 255
                matched_masks.append(binary_mask)
                self.logger.debug(f"   ✓ 使用Mask R-CNN掩码 (IoU: {best_iou:.3f}, 置信度: {best_score:.3f})")
            else:
                # 如果没有好的匹配，创建简化掩码
                self.logger.info(f"   ⚠️ 未找到匹配掩码 (最佳IoU: {best_iou:.3f})，使用简化掩码")
                simple_mask = self._create_mask_from_detection(
                    np.zeros((*image_shape, 3), dtype=np.uint8), detection
                )
                if simple_mask is not None:
                    matched_masks.append(simple_mask)
        
        return matched_masks
    
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
        
        # 根据物体类型生成不同形状的掩码
        center_x, center_y = x + w//2, y + h//2
        
        if 'can' in class_name.lower():
            # 圆柱形物体 - 椭圆掩码
            cv2.ellipse(mask, (center_x, center_y), (w//2-5, h//2-5), 0, 0, 360, 255, -1)
        elif 'phone' in class_name.lower():
            # 矩形物体 - 带圆角的矩形
            cv2.rectangle(mask, (x+3, y+3), (x+w-3, y+h-3), 255, -1)
        else:
            # 默认椭圆掩码
            cv2.ellipse(mask, (center_x, center_y), (w//2-3, h//2-3), 0, 0, 360, 255, -1)
        
        return mask
    
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
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
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
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
    print("🧪 测试TorchVision Mask R-CNN分割器...")
    
    # 测试配置
    config = {
        'confidence_threshold': 0.5,
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
    
    # 执行分割
    masks = segmentor.segment(test_image)
    
    print(f"✅ 分割结果: {len(masks)} 个掩码")
    
    # 获取统计信息
    stats = segmentor.get_mask_statistics(masks)
    print(f"📊 掩码统计: {stats}")
    
    # 可视化
    if masks:
        vis_image = segmentor.visualize_masks(test_image, masks)
        cv2.imshow("TorchVision Mask R-CNN分割结果", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("🎉 Mask R-CNN分割器测试完成!")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_maskrcnn_segmentor()

