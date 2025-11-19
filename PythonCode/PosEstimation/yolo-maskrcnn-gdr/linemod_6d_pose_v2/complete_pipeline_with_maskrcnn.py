#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete YOLO + Mask R-CNN + GDR-Net 6D Pose Estimation Pipeline
完整的YOLO检测 + Mask R-CNN分割 + GDR-Net姿态估计Pipeline

完整流程：
1. YOLO检测物体并提供边界框
2. Mask R-CNN对检测到的物体进行精确分割
3. GDR-Net基于分割结果进行6D姿态估计
4. 结果可视化
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import time
import logging
from typing import List, Tuple, Dict, Optional
import json
import yaml

# 添加路径
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "pose_estimation"))
sys.path.insert(0, str(CURRENT_DIR.parent / "src"))

# ============================================================================
# 配置参数（直接在脚本中设置，不使用命令行）
# ============================================================================

CONFIG = {
    # ========== 设备配置 ==========
    'device': 'cuda',  # 'cuda' 或 'cpu'
    
    # ========== YOLO检测配置 ==========
    'yolo': {
        'model_path': "d:/projects/odb_box/runs/train/lmo_8classes/weights/best.pt",  # 🔥 LMO 8类模型
        'conf_threshold': 0.7,   # 🔥 高置信度阈值（只保留高质量检测）
        'iou_threshold': 0.7,    # 🔥 严格NMS
        'max_det': 50,           # 最大检测数
        'agnostic_nms': False,   # 类别相关NMS
        'filter_large_boxes': True,  # 启用大框过滤
        'max_box_area_ratio': 0.5,    # 大框过滤阈值（超过50%图像面积的框被过滤）
    },
    
    # ========== Mask R-CNN分割配置 ==========
    'maskrcnn': {
        'enabled': False,  # 是否启用Mask R-CNN分割
        'model_name': "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        'confidence_threshold': 0.3,  # 降低阈值以检测更多物体
        'mask_threshold': 0.5,
        'use_detection_prompts': True,  # 使用YOLO检测结果作为提示
        'smooth_mask': True,
        'min_mask_area': 100,
        'max_mask_area': 50000,
    },
    
    # ========== GDR-Net姿态估计配置 ==========
    'gdrnet': {
        'config_path': None,  # None表示使用默认配置
        'model_path': None,   # None表示使用默认路径
        'enabled': True,      # 是否启用姿态估计
    },
    
    # ========== 可视化配置 ==========
    'visualization': {
        'show_bbox': True,
        'show_mask': True,
        'show_axes': True,
        'show_confidence': True,
        'mask_alpha': 0.3,
    },
    
    # ========== 输出配置 ==========
    'output': {
        'save_json': True,
        'save_image': True,
        'image_format': 'jpg',
    }
}

# ============================================================================

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# LineMOD物体3D边界框信息 (单位: mm) - 从models_info.yml加载
# 包含min_x, min_y, min_z 和 size_x, size_y, size_z
LINEMOD_OBJECT_BBOXES = {
    1: {'min_x': -37.93, 'min_y': -38.80, 'min_z': -45.88, 'size_x': 75.87, 'size_y': 77.60, 'size_z': 91.77},
    2: {'min_x': -107.84, 'min_y': -60.93, 'min_z': -109.71, 'size_x': 215.67, 'size_y': 121.86, 'size_z': 219.41},
    4: {'min_x': -68.33, 'min_y': -71.52, 'min_z': -50.25, 'size_x': 136.66, 'size_y': 143.03, 'size_z': 100.50},
    5: {'min_x': -50.40, 'min_y': -90.90, 'min_z': -96.87, 'size_x': 100.79, 'size_y': 181.80, 'size_z': 193.73},
    6: {'min_x': -33.51, 'min_y': -63.82, 'min_z': -58.73, 'size_x': 67.01, 'size_y': 127.63, 'size_z': 117.46},
    8: {'min_x': -114.74, 'min_y': -37.74, 'min_z': -104.00, 'size_x': 229.48, 'size_y': 75.47, 'size_z': 208.00},
    9: {'min_x': -52.21, 'min_y': -38.70, 'min_z': -42.85, 'size_x': 104.43, 'size_y': 77.41, 'size_z': 85.70},
    10: {'min_x': -75.09, 'min_y': -53.54, 'min_z': -34.62, 'size_x': 150.18, 'size_y': 107.08, 'size_z': 69.24},
    11: {'min_x': -18.36, 'min_y': -38.93, 'min_z': -86.41, 'size_x': 36.72, 'size_y': 77.87, 'size_z': 172.82},
    12: {'min_x': -50.44, 'min_y': -54.25, 'min_z': -45.40, 'size_x': 100.89, 'size_y': 108.50, 'size_z': 90.80},
    13: {'min_x': -129.11, 'min_y': -59.24, 'min_z': -70.57, 'size_x': 258.23, 'size_y': 118.48, 'size_z': 141.13},
    14: {'min_x': -101.57, 'min_y': -58.88, 'min_z': -106.56, 'size_x': 203.15, 'size_y': 117.75, 'size_z': 213.12},
    15: {'min_x': -46.96, 'min_y': -73.72, 'min_z': -92.37, 'size_x': 93.92, 'size_y': 147.43, 'size_z': 184.75},
}


class CompletePipeline:
    """
    完整的6D姿态估计Pipeline
    
    流程：
    1. YOLO检测物体 → 提供边界框
    2. Mask R-CNN分割 → 提供精确掩码
    3. GDR-Net姿态估计 → 提供6D姿态
    4. 结果可视化
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化Pipeline
        
        Args:
            config: 配置字典（使用全局CONFIG如果为None）
        """
        self.config = config if config is not None else CONFIG
        self.device = self.config['device']
        
        logger.info("="*70)
        logger.info("🚀 初始化完整6D姿态估计Pipeline")
        logger.info("="*70)
        logger.info(f"设备: {self.device}")
        
        # LineMOD类别映射（ID到名称）- 匹配YOLO训练的13个类别
        self.linemod_classes = {
            0: 'ape', 1: 'benchvise', 2: 'cam', 3: 'can', 4: 'cat',
            5: 'driller', 6: 'duck', 7: 'eggbox', 8: 'glue', 9: 'holepuncher',
            10: 'iron', 11: 'lamp', 12: 'phone'
        }
        
        # YOLO类别ID到LineMOD物体ID的映射
        self.yolo_to_linemod_id = {
            0: 1,   # ape
            1: 2,   # benchvise
            2: 4,   # cam -> camera (LineMOD ID 4)
            3: 5,   # can
            4: 6,   # cat
            5: 8,   # driller
            6: 9,   # duck
            7: 10,  # eggbox
            8: 11,  # glue
            9: 12,  # holepuncher
            10: 13, # iron
            11: 14, # lamp
            12: 15  # phone
        }
        
        # LineMOD默认相机内参（从info.yml）
        self.camera_K = np.array([
            [572.4114, 0.0, 325.2611],
            [0.0, 573.57043, 242.04899],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # 初始化各个模块
        self.yolo_model = self._init_yolo()
        self.maskrcnn = self._init_maskrcnn()
        self.gdrnet = self._init_gdrnet()
        
        logger.info("="*70)
        logger.info("✅ Pipeline初始化完成")
        logger.info("="*70)
    
    def _init_yolo(self):
        """初始化YOLO检测器"""
        logger.info("\n📦 初始化YOLO检测器...")
        
        try:
            from ultralytics import YOLO
            
            yolo_cfg = self.config['yolo']
            model_path = yolo_cfg['model_path']
            
            # 仅使用配置中的8类物体模型
            if model_path is None:
                logger.error(f"   ❌ 未配置YOLO模型路径")
                return None
            
            model = YOLO(model_path)
            logger.info(f"   ✓ YOLO模型加载成功: {Path(model_path).name}")
            logger.info(f"   类别数: {len(model.names)}")
            
            return model
            
        except Exception as e:
            logger.error(f"   ❌ YOLO初始化失败: {e}")
            return None
    
    def _init_maskrcnn(self):
        """初始化Mask R-CNN分割器"""
        if not self.config['maskrcnn']['enabled']:
            logger.info("\n⏭️ Mask R-CNN分割已禁用")
            return None
        
        logger.info("\n📦 初始化Mask R-CNN分割器...")
        
        try:
            # 优先使用TorchVision版本（无需Detectron2）
            from maskrcnn_segmentor_torchvision import MaskRCNNSegmentor
            
            maskrcnn_cfg = self.config['maskrcnn'].copy()
            maskrcnn_cfg['device'] = self.device
            
            segmentor = MaskRCNNSegmentor(maskrcnn_cfg)
            logger.info(f"   ✓ Mask R-CNN初始化成功 (TorchVision)")
            
            return segmentor
            
        except Exception as e:
            logger.warning(f"   ⚠️ Mask R-CNN初始化失败: {e}")
            logger.warning(f"   将跳过分割步骤")
            import traceback
            traceback.print_exc()
            return None
    
    def _init_gdrnet(self):
        """初始化GDR-Net姿态估计器（使用官方推理流程）"""
        if not self.config['gdrnet']['enabled']:
            logger.info("\n⏭️ GDR-Net姿态估计已禁用")
            return None
        
        logger.info("\n📦 初始化GDR-Net姿态估计器（官方流程）...")
        
        try:
            from gdrnet_official_wrapper import GDRNetOfficialInference
            
            # 使用官方配置和权重路径
            config_path = self.config['gdrnet'].get('config_path', None)
            model_path = self.config['gdrnet'].get('model_path', None)
            
            gdrnet = GDRNetOfficialInference(
                config_path=config_path,  # None表示使用默认
                model_path=model_path,    # None表示使用默认
                device=self.device
            )
            logger.info(f"   ✓ GDR-Net官方推理引擎初始化成功")
            
            return gdrnet
            
        except Exception as e:
            logger.error(f"   ❌ GDR-Net初始化失败: {e}")
            logger.error(f"   将跳过姿态估计步骤")
            import traceback
            logger.error("详细错误信息：")
            traceback.print_exc()
            return None
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        步骤1: YOLO检测物体
        
        Args:
            image: BGR图像
            
        Returns:
            检测结果列表
        """
        if self.yolo_model is None:
            return []
        
        yolo_cfg = self.config['yolo']
        
        # YOLO推理（添加max_det和agnostic_nms参数）
        results = self.yolo_model(
            image,
            conf=yolo_cfg['conf_threshold'],
            iou=yolo_cfg['iou_threshold'],
            max_det=yolo_cfg.get('max_det', 50),  # 最大检测数
            agnostic_nms=yolo_cfg.get('agnostic_nms', True),  # 类别无关的NMS
            verbose=False
        )
        
        detections = []
        img_area = image.shape[0] * image.shape[1]
        
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                
                x1, y1, x2, y2 = map(int, box)
                
                # 大框过滤：如果启用，过滤掉过大的检测框
                if yolo_cfg.get('filter_large_boxes', False):
                    box_area = (x2 - x1) * (y2 - y1)
                    max_area_ratio = yolo_cfg.get('max_box_area_ratio', 0.4)
                    
                    if box_area / img_area > max_area_ratio:
                        logger.warning(f"   ⚠️  过滤掉过大检测框: class_id={cls_id}, "
                                     f"面积占比={box_area/img_area:.1%}")
                        continue
                
                # 获取类别名称
                class_name = self.linemod_classes.get(cls_id, f'class_{cls_id}')
                linemod_id = self.yolo_to_linemod_id.get(cls_id, cls_id + 1)
                
                detections.append({
                    'yolo_class_id': cls_id,
                    'linemod_object_id': linemod_id,
                    'class_name': class_name,
                    'bbox_xyxy': [x1, y1, x2, y2],
                    'bbox_xywh': [x1, y1, x2-x1, y2-y1],
                    'confidence': conf
                })
        
        logger.info(f"   ✓ 检测到 {len(detections)} 个物体")
        return detections
    
    def color_based_mask_refinement(
        self,
        image: np.ndarray,
        bbox_xyxy: List[int],
        object_name: str,
        base_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        基于颜色的mask精炼（针对纯色物体）
        从bbox中心采样参考颜色，然后在bbox区域内进行相似颜色分割
        
        Args:
            image: BGR图像
            bbox_xyxy: 检测框 [x1, y1, x2, y2]
            object_name: 物体名称
            base_mask: GDR-Net生成的基础mask（可选）
            
        Returns:
            精炼后的mask
        """
        # 只对特定纯色物体进行颜色分割
        color_objects = ['duck']  # 可以添加更多纯色物体
        
        if object_name not in color_objects:
            # 如果不是纯色物体，返回基础mask
            return base_mask if base_mask is not None else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 提取检测框区域
        x1, y1, x2, y2 = bbox_xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        w, h = x2 - x1, y2 - y1
        
        if w <= 0 or h <= 0:
            return base_mask if base_mask is not None else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 扩展检测框（增加10%边界以包含完整物体）
        margin_w, margin_h = int(w * 0.1), int(h * 0.1)
        x1_exp = max(0, x1 - margin_w)
        y1_exp = max(0, y1 - margin_h)
        x2_exp = min(image.shape[1], x2 + margin_w)
        y2_exp = min(image.shape[0], y2 + margin_h)
        
        roi = image[y1_exp:y2_exp, x1_exp:x2_exp]
        
        if roi.size == 0:
            return base_mask if base_mask is not None else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # ========== 步骤1: 从bbox中心区域采样参考颜色（BGR） ==========
        # 计算bbox中心区域（取中心30%的区域）
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        sample_w, sample_h = int(w * 0.3), int(h * 0.3)
        sample_x1 = max(x1, center_x - sample_w // 2)
        sample_y1 = max(y1, center_y - sample_h // 2)
        sample_x2 = min(x2, center_x + sample_w // 2)
        sample_y2 = min(y2, center_y + sample_h // 2)
        
        # 提取中心采样区域
        center_sample = image[sample_y1:sample_y2, sample_x1:sample_x2]
        
        if center_sample.size == 0:
            logger.warning(f"      中心采样区域为空，使用基础mask")
            return base_mask if base_mask is not None else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # 计算中心区域的平均BGR颜色
        mean_b = np.mean(center_sample[:, :, 0])
        mean_g = np.mean(center_sample[:, :, 1])
        mean_r = np.mean(center_sample[:, :, 2])
        
        logger.info(f"      中心采样区域BGR: B={mean_b:.1f}, G={mean_g:.1f}, R={mean_r:.1f}")
        
        # ========== 步骤2: 基于参考颜色在扩展bbox内进行分割（BGR空间） ==========
        # 定义颜色容差（根据物体类型调整）
        if object_name == 'duck':
            # 黄色鸭子：BGR容差 - 黄色主要是G和R都高，B低
            color_tolerance = 40  # 每个通道的容差
        else:
            # 默认容差
            color_tolerance = 50
        
        # 计算颜色范围
        lower_b = max(0, mean_b - color_tolerance)
        upper_b = min(255, mean_b + color_tolerance)
        lower_g = max(0, mean_g - color_tolerance)
        upper_g = min(255, mean_g + color_tolerance)
        lower_r = max(0, mean_r - color_tolerance)
        upper_r = min(255, mean_r + color_tolerance)
        
        lower_bound = np.array([lower_b, lower_g, lower_r])
        upper_bound = np.array([upper_b, upper_g, upper_r])
        
        logger.info(f"      颜色范围(BGR): B=[{lower_b:.1f}, {upper_b:.1f}], "
                   f"G=[{lower_g:.1f}, {upper_g:.1f}], R=[{lower_r:.1f}, {upper_r:.1f}]")
        
        # 创建颜色mask（直接在BGR空间）
        color_mask_roi = cv2.inRange(roi, lower_bound, upper_bound)
        
        # ========== 步骤3: 形态学操作优化mask ==========
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 闭运算：填充小孔
        color_mask_roi = cv2.morphologyEx(color_mask_roi, cv2.MORPH_CLOSE, kernel)
        # 开运算：去除噪点
        color_mask_roi = cv2.morphologyEx(color_mask_roi, cv2.MORPH_OPEN, kernel)
        
        # ========== 步骤4: 将ROI mask映射回全图 ==========
        color_mask_full = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        color_mask_full[y1_exp:y2_exp, x1_exp:x2_exp] = color_mask_roi
        
        # ========== 步骤5: 与基础mask结合（如果有） ==========
        if base_mask is not None:
            # 取并集，扩大分割区域
            combined_mask = cv2.bitwise_or(base_mask, color_mask_full)
            
            # 限制在扩展bbox内
            bbox_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            bbox_mask[y1_exp:y2_exp, x1_exp:x2_exp] = 255
            
            final_mask = cv2.bitwise_and(combined_mask, bbox_mask)
        else:
            final_mask = color_mask_full
        
        # 统计信息
        color_pixels = cv2.countNonZero(color_mask_full)
        final_pixels = cv2.countNonZero(final_mask)
        logger.info(f"      颜色分割像素数: {color_pixels}, 最终mask像素数: {final_pixels}")
        
        return final_mask
    
    def segment_objects(self, image: np.ndarray, detections: List[Dict]) -> List[np.ndarray]:
        """
        步骤2: Mask R-CNN分割物体
        
        Args:
            image: BGR图像
            detections: YOLO检测结果
            
        Returns:
            分割掩码列表
        """
        if self.maskrcnn is None or len(detections) == 0:
            return []
        
        try:
            # 准备检测结果供Mask R-CNN使用
            det_for_maskrcnn = []
            for det in detections:
                det_for_maskrcnn.append({
                    'bbox': det['bbox_xywh'],
                    'class_name': det['class_name'],
                    'mapped_class': det['class_name'],
                    'confidence': det['confidence']
                })
            
            # 执行分割
            masks = self.maskrcnn.segment(image, det_for_maskrcnn)
            
            return masks
            
        except Exception as e:
            logger.error(f"   分割失败: {e}")
            return []
    
    def estimate_poses(
        self,
        image: np.ndarray,
        detections: List[Dict],
        masks: List[np.ndarray] = None
    ) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        步骤3: GDR-Net 6D姿态估计（使用官方推理流程，同时获取GDR-Net的mask）
        
        Args:
            image: BGR图像
            detections: 检测结果
            masks: 分割掩码（可选，来自Mask R-CNN，但官方GDR-Net不使用）
            
        Returns:
            姿态估计结果列表, GDR-Net生成的掩码列表
        """
        if self.gdrnet is None or len(detections) == 0:
            return [], []
        
        poses = []
        gdrnet_masks = []
        
        for i, det in enumerate(detections):
            try:
                # 转换为RGB（官方GDR-Net需要RGB）
                if len(image.shape) == 3 and image.shape[2] == 3:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = image
                
                # 获取bbox（xyxy格式）
                bbox_xyxy = tuple(det['bbox_xyxy'])
                
                # 获取对应的掩码（来自Mask R-CNN，可选，但官方GDR-Net不使用）
                mask_rcnn = masks[i] if masks and i < len(masks) else None
                
                # 调用官方GDR-Net推理（新接口）
                # 新接口: estimate_pose(image, bbox_xyxy, object_id, camera_K, score=1.0)
                rotation, translation, confidence, gdrnet_mask, roi_info = self.gdrnet.estimate_pose(
                    image=rgb_image,
                    bbox_xyxy=bbox_xyxy,
                    object_id=det['linemod_object_id'],
                    camera_K=self.camera_K,
                    score=det.get('confidence', 1.0)
                )
                
                # 将GDR-Net的mask（256x256）映射回原图（使用官方的逆仿射变换）
                final_mask = None
                mask_source = 'None'
                
                if gdrnet_mask is not None and roi_info is not None:
                    # 导入GDR-Net的仿射变换函数
                    import sys
                    from pathlib import Path
                    gdrnet_root = Path(__file__).resolve().parents[2] / "GDR-Net-main" / "GDR-Net-main"
                    if str(gdrnet_root) not in sys.path:
                        sys.path.insert(0, str(gdrnet_root))
                    from core.utils.data_utils import get_affine_transform
                    
                    # 获取ROI参数
                    center = roi_info['center']
                    scale = roi_info['scale']  # 这是一个标量
                    input_res = roi_info['input_res']
                    
                    # 获取逆仿射变换矩阵（从ROI空间回到原图空间）
                    trans_inv = get_affine_transform(
                        center=center,
                        scale=scale,  # scale是标量，会被转为[scale, scale]
                        rot=0,
                        output_size=input_res,
                        inv=True  # 逆变换！
                    )
                    
                    # 应用逆变换将mask映射回原图
                    full_mask = cv2.warpAffine(
                        gdrnet_mask, trans_inv, (image.shape[1], image.shape[0]),
                        flags=cv2.INTER_LINEAR, borderValue=0
                    )
                    
                    # 二值化
                    full_mask = ((full_mask > 50) * 255).astype(np.uint8)
                    
                    # 针对纯色物体（如鸭子），使用颜色辅助分割来改善mask
                    object_name = det['class_name']
                    if object_name in ['duck']:  # 可以添加更多纯色物体
                        logger.info(f"      检测到纯色物体 '{object_name}'，使用颜色辅助分割...")
                        final_mask = self.color_based_mask_refinement(
                            image=image,
                            bbox_xyxy=det['bbox_xyxy'],
                            object_name=object_name,
                            base_mask=full_mask
                        )
                        mask_source = 'GDR-Net + Color'
                    else:
                        final_mask = full_mask
                        mask_source = 'GDR-Net'
                        
                elif mask_rcnn is not None:
                    final_mask = mask_rcnn
                    mask_source = 'Mask R-CNN'
                
                has_gdrnet_mask = gdrnet_mask is not None
                
                poses.append({
                    'detection_index': i,
                    'object_id': det['linemod_object_id'],
                    'class_name': det['class_name'],
                    'rotation': rotation,
                    'translation': translation,
                    'confidence': confidence,
                    'has_mask': final_mask is not None,
                    'mask_source': mask_source
                })
                
                # 收集mask（优先使用GDR-Net的）
                if final_mask is not None:
                    gdrnet_masks.append(final_mask)
                
            except Exception as e:
                logger.error(f"   姿态估计失败 (物体{i+1}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return poses, gdrnet_masks
    
    def process_image(self, image: np.ndarray, visualize: bool = True) -> Dict:
        """
        处理单张图像（完整流程）
        
        Args:
            image: BGR图像
            visualize: 是否生成可视化结果
            
        Returns:
            完整结果字典
        """
        start_time = time.time()
        
        results = {
            'detections': [],
            'masks': [],
            'poses': [],
            'processing_time': {},
            'image_shape': image.shape
        }
        
        # ========== 步骤1: YOLO检测 ==========
        logger.info("\n" + "="*70)
        logger.info("🔍 步骤1: YOLO目标检测")
        logger.info("="*70)
        
        detect_start = time.time()
        detections = self.detect_objects(image)
        results['detections'] = detections
        results['processing_time']['detection'] = time.time() - detect_start
        
        logger.info(f"   ✓ 检测到 {len(detections)} 个物体")
        for i, det in enumerate(detections):
            logger.info(f"      {i+1}. {det['class_name']} (ID:{det['linemod_object_id']}, 置信度:{det['confidence']:.3f})")
        
        if len(detections) == 0:
            logger.info("   ℹ️ 未检测到物体，流程结束")
            return results
        
        # ========== 步骤2: Mask R-CNN分割 ==========
        if self.maskrcnn is not None:
            logger.info("\n" + "="*70)
            logger.info("✂️ 步骤2: Mask R-CNN实例分割")
            logger.info("="*70)
            
            segment_start = time.time()
            masks = self.segment_objects(image, detections)
            results['masks'] = masks
            results['processing_time']['segmentation'] = time.time() - segment_start
            
            logger.info(f"   ✓ 生成 {len(masks)} 个分割掩码")
        else:
            masks = []
            logger.info("\n   ⏭️ 跳过分割步骤（Mask R-CNN未启用）")
        
        # ========== 步骤3: GDR-Net姿态估计 ==========
        if self.gdrnet is not None:
            logger.info("\n" + "="*70)
            logger.info("🎯 步骤3: GDR-Net 6D姿态估计 (含内置mask分割)")
            logger.info("="*70)
            
            pose_start = time.time()
            poses, gdrnet_masks = self.estimate_poses(image, detections, masks)
            results['poses'] = poses
            results['processing_time']['pose_estimation'] = time.time() - pose_start
            
            # 使用GDR-Net的mask（如果有）替换或补充Mask R-CNN的mask
            if gdrnet_masks:
                if not masks:
                    results['masks'] = gdrnet_masks
                    logger.info(f"   ✓ 使用GDR-Net生成的 {len(gdrnet_masks)} 个掩码")
                else:
                    # 合并masks
                    results['masks'] = gdrnet_masks
                    logger.info(f"   ✓ 使用GDR-Net掩码替换Mask R-CNN掩码")
            
            logger.info(f"   ✓ 估计 {len(poses)} 个物体的6D姿态")
            for i, pose in enumerate(poses):
                logger.info(f"      {i+1}. {pose['class_name']}: 置信度={pose['confidence']:.3f}, 掩码来源={pose.get('mask_source', 'Unknown')}")
        else:
            logger.info("\n   ⏭️ 跳过姿态估计步骤（GDR-Net未启用）")
        
        # ========== 步骤4: 可视化 ==========
        if visualize:
            logger.info("\n" + "="*70)
            logger.info("🎨 步骤4: 结果可视化")
            logger.info("="*70)
            
            vis_start = time.time()
            vis_image = self.visualize_results(image, results)
            results['visualization'] = vis_image
            results['processing_time']['visualization'] = time.time() - vis_start
            
            logger.info(f"   ✓ 可视化完成")
        
        # 总时间
        results['processing_time']['total'] = time.time() - start_time
        
        logger.info("\n" + "="*70)
        logger.info(f"⏱️ 总处理时间: {results['processing_time']['total']:.3f}秒")
        logger.info("="*70)
        
        return results
    
    def visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        可视化所有结果
        
        Args:
            image: 原始图像
            results: 处理结果
            
        Returns:
            可视化图像
        """
        vis_image = image.copy()
        vis_cfg = self.config['visualization']
        
        detections = results.get('detections', [])
        masks = results.get('masks', [])
        poses = results.get('poses', [])
        
        # 颜色列表
        colors = [
            (0, 255, 255),    # 黄色
            (255, 0, 255),    # 品红
            (255, 255, 0),    # 青色
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 蓝色
            (0, 165, 255),    # 橙色
            (128, 0, 128),    # 紫色
            (255, 192, 203),  # 粉色
        ]
        
        # 1. 绘制分割掩码（黑色边界线）
        if vis_cfg['show_mask'] and len(masks) > 0:
            for i, mask in enumerate(masks):
                # 确保mask是uint8类型
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)
                
                # 二值化
                _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                
                # 找到轮廓
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 绘制黑色轮廓线
                    cv2.drawContours(vis_image, contours, -1, (0, 0, 0), 3)  # 黑色，3像素宽
        
        # 3. 绘制6D姿态（3D边界框和坐标轴）
        if vis_cfg['show_axes'] and len(poses) > 0:
            for pose in poses:
                try:
                    rotation = pose['rotation']
                    translation = pose['translation']  # 单位：米
                    object_id = pose['object_id']
                    camera_K = self.camera_K
                    
                    # ===== 绘制3D边界框 =====
                    if object_id in LINEMOD_OBJECT_BBOXES:
                        bbox_info = LINEMOD_OBJECT_BBOXES[object_id]
                        min_x, min_y, min_z = bbox_info['min_x'], bbox_info['min_y'], bbox_info['min_z']
                        size_x, size_y, size_z = bbox_info['size_x'], bbox_info['size_y'], bbox_info['size_z']
                        max_x, max_y, max_z = min_x + size_x, min_y + size_y, min_z + size_z
                        
                        # 定义8个顶点（单位：mm，物体坐标系）
                        bbox_3d = np.array([
                            [min_x, min_y, min_z],  # 0: 左下后
                            [max_x, min_y, min_z],  # 1: 右下后
                            [max_x, max_y, min_z],  # 2: 右上后
                            [min_x, max_y, min_z],  # 3: 左上后
                            [min_x, min_y, max_z],  # 4: 左下前
                            [max_x, min_y, max_z],  # 5: 右下前
                            [max_x, max_y, max_z],  # 6: 右上前
                            [min_x, max_y, max_z],  # 7: 左上前
                        ]) / 1000.0  # 转换为米
                        
                        # 变换到相机坐标系（translation也是米）
                        bbox_cam = (rotation @ bbox_3d.T).T + translation
                        
                        # 投影到2D
                        bbox_2d = (camera_K @ bbox_cam.T).T
                        bbox_2d = bbox_2d[:, :2] / bbox_2d[:, 2:3]
                        bbox_2d = bbox_2d.astype(int)
                        
                        # 定义12条边
                        edges = [
                            # 后面（z=-d/2）
                            (0, 1), (1, 2), (2, 3), (3, 0),
                            # 前面（z=d/2）
                            (4, 5), (5, 6), (6, 7), (7, 4),
                            # 连接前后
                            (0, 4), (1, 5), (2, 6), (3, 7)
                        ]
                        
                        # 绘制边界框（红色）
                        for i, j in edges:
                            pt1 = tuple(bbox_2d[i])
                            pt2 = tuple(bbox_2d[j])
                            cv2.line(vis_image, pt1, pt2, (0, 0, 255), 2)  # 红色
                    
                    # ===== 绘制坐标轴 =====
                    axis_length = 0.08  # 8cm = 0.08米
                    
                    # 定义坐标轴端点（3D，单位：米）
                    axes_3d = np.array([
                        [0, 0, 0],                # 原点
                        [axis_length, 0, 0],      # X轴端点 - 红色
                        [0, axis_length, 0],      # Y轴端点 - 绿色
                        [0, 0, axis_length],      # Z轴端点 - 蓝色
                    ])
                    
                    # 变换到相机坐标系
                    axes_cam = (rotation @ axes_3d.T).T + translation
                    
                    # 投影到图像平面
                    axes_2d = (camera_K @ axes_cam.T).T
                    axes_2d = axes_2d[:, :2] / axes_2d[:, 2:3]
                    
                    origin = tuple(axes_2d[0].astype(int))
                    x_end = tuple(axes_2d[1].astype(int))
                    y_end = tuple(axes_2d[2].astype(int))
                    z_end = tuple(axes_2d[3].astype(int))
                    
                    # 绘制坐标轴（加粗）
                    cv2.line(vis_image, origin, x_end, (0, 0, 255), 4)    # X - 红色
                    cv2.line(vis_image, origin, y_end, (0, 255, 0), 4)    # Y - 绿色
                    cv2.line(vis_image, origin, z_end, (255, 0, 0), 4)    # Z - 蓝色
                        
                except Exception as e:
                    logger.debug(f"绘制姿态可视化失败: {e}")
        
        return vis_image
    
    def save_results(self, results: Dict, output_path: str):
        """
        保存结果
        
        Args:
            results: 处理结果
            output_path: 输出路径（不含扩展名）
        """
        output_path = Path(output_path)
        output_cfg = self.config['output']
        
        # 保存图像
        if output_cfg['save_image'] and 'visualization' in results:
            image_path = output_path.with_suffix(f".{output_cfg['image_format']}")
            cv2.imwrite(str(image_path), results['visualization'])
            logger.info(f"   ✓ 图像已保存: {image_path}")
        
        # 保存JSON
        if output_cfg['save_json']:
            json_path = output_path.with_suffix('.json')
            
            # 将numpy数组转换为列表
            json_results = {
                'detections': results['detections'],
                'num_masks': len(results.get('masks', [])),
                'poses': [
                    {
                        'detection_index': p['detection_index'],
                        'object_id': p['object_id'],
                        'class_name': p['class_name'],
                        'rotation': p['rotation'].tolist(),
                        'translation': p['translation'].tolist(),
                        'confidence': p['confidence'],
                        'has_mask': p['has_mask']
                    }
                    for p in results.get('poses', [])
                ],
                'processing_time': results['processing_time'],
                'image_shape': results['image_shape']
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"   ✓ JSON已保存: {json_path}")


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """主函数 - 从13类物体中各挑选100张图片进行全流程测试"""
    import random
    
    # 使用全局CONFIG配置
    pipeline = CompletePipeline(CONFIG)
    
    # ========== 从13类物体中各挑选100张图片 ==========
    dataset_path = CURRENT_DIR.parent.parent / "dataset" / "Linemod_preprocessed" / "Linemod_preprocessed" / "data"
    
    # LineMOD物体ID列表（对应13个类别）
    linemod_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    object_names = ['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck', 
                    'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']
    
    # 设置随机种子
    random.seed(42)
    
    # 为每个类别随机选择100张图片
    logger.info("\n" + "="*70)
    logger.info("📋 从13类物体中各随机选择100张图片")
    logger.info("="*70)
    
    test_images_per_class = {}
    
    for obj_id, obj_name in zip(linemod_ids, object_names):
        rgb_dir = dataset_path / f"{obj_id:02d}" / "rgb"
        
        if not rgb_dir.exists():
            logger.warning(f"   ⚠️ 目录不存在: {rgb_dir}")
            test_images_per_class[obj_id] = []
            continue
        
        # 获取所有图片
        all_images = sorted([f.name for f in rgb_dir.glob('*.png')])
        
        # 随机选择100张（如果不足100张则全选）
        if len(all_images) >= 100:
            selected_images = random.sample(all_images, 100)
        else:
            selected_images = all_images
            logger.warning(f"   ⚠️ {obj_name} 只有 {len(all_images)} 张图片")
        
        test_images_per_class[obj_id] = selected_images
        logger.info(f"   ✓ {obj_name}: 选择了 {len(selected_images)} 张图片")
    
    # 保存测试图片列表到 testdata.json
    testdata = {
        'description': '从LineMOD 13类物体中各随机选择100张图片',
        'total_images': sum(len(imgs) for imgs in test_images_per_class.values()),
        'images_per_class': {
            obj_name: {
                'object_id': obj_id,
                'images': test_images_per_class[obj_id]
            }
            for obj_id, obj_name in zip(linemod_ids, object_names)
        }
    }
    
    testdata_path = CURRENT_DIR.parent.parent / "testdata.json"
    with open(testdata_path, 'w', encoding='utf-8') as f:
        json.dump(testdata, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✓ 测试图片列表已保存: {testdata_path}")
    logger.info(f"   总图片数: {testdata['total_images']}")
    
    # 创建输出目录
    output_dir = CURRENT_DIR.parent.parent / "output_multi_performance"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"\n📂 输出目录: {output_dir}")
    
    # 统计信息
    total_images = 0
    successful_images = 0
    failed_images = []
    performance_summary = []
    
    logger.info("\n" + "="*70)
    logger.info("🚀 开始全流程测试")
    logger.info("="*70)
    
    # 遍历所有类别
    for obj_id, obj_name in zip(linemod_ids, object_names):
        logger.info(f"\n{'='*70}")
        logger.info(f"📦 处理类别: {obj_name} (ID: {obj_id})")
        logger.info(f"{'='*70}")
        
        test_images = test_images_per_class.get(obj_id, [])
        
        for img_name in test_images:
            total_images += 1
            
            # 构建图像路径
            image_path = dataset_path / f"{obj_id:02d}" / "rgb" / img_name
            
            if not image_path.exists():
                logger.warning(f"   ⚠️ 图像不存在: {image_path}")
                failed_images.append({
                    'object_id': obj_id,
                    'object_name': obj_name,
                    'image_name': img_name,
                    'reason': 'File not found'
                })
                continue
            
            logger.info(f"\n🖼️ 处理图像 [{total_images}/{testdata['total_images']}]: {obj_name}/{img_name}")
            
            try:
                # 读取图像
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.error(f"   ❌ 无法加载图像: {image_path}")
                    failed_images.append({
                        'object_id': obj_id,
                        'object_name': obj_name,
                        'image_name': img_name,
                        'reason': 'Failed to load'
                    })
                    continue
                
                # 处理图像
                results = pipeline.process_image(image, visualize=True)
                
                # 保存结果
                output_filename = f"{obj_name}_{img_name.replace('.png', '')}"
                output_path = output_dir / output_filename
                pipeline.save_results(results, str(output_path))
                
                # 记录性能信息
                performance_info = {
                    'object_id': obj_id,
                    'object_name': obj_name,
                    'image_name': img_name,
                    'num_detections': len(results.get('detections', [])),
                    'num_poses': len(results.get('poses', [])),
                    'processing_time': results.get('processing_time', {}),
                    'output_file': output_filename
                }
                performance_summary.append(performance_info)
                
                successful_images += 1
                logger.info(f"   ✓ 成功处理并保存结果")
                
            except Exception as e:
                logger.error(f"   ❌ 处理失败: {e}")
                import traceback
                traceback.print_exc()
                failed_images.append({
                    'object_id': obj_id,
                    'object_name': obj_name,
                    'image_name': img_name,
                    'reason': str(e)
                })
    
    # ========== 保存性能测试总结报告 ==========
    logger.info("\n" + "="*70)
    logger.info("📊 生成性能测试总结报告")
    logger.info("="*70)
    
    summary_report = {
        'test_info': {
            'total_images': total_images,
            'successful_images': successful_images,
            'failed_images_count': len(failed_images),
            'success_rate': f"{successful_images/total_images*100:.2f}%" if total_images > 0 else "0%"
        },
        'performance_details': performance_summary,
        'failed_images': failed_images
    }
    
    # 保存JSON报告
    report_path = output_dir / "performance_test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"   ✓ 报告已保存: {report_path}")
    
    # ========== 打印总结 ==========
    logger.info("\n" + "="*70)
    logger.info("🎉 全流程测试完成")
    logger.info("="*70)
    logger.info(f"总图像数: {total_images}")
    logger.info(f"成功处理: {successful_images}")
    logger.info(f"失败数量: {len(failed_images)}")
    logger.info(f"成功率: {summary_report['test_info']['success_rate']}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("="*70)
    
    if failed_images:
        logger.info("\n失败的图像:")
        for fail in failed_images:
            logger.info(f"   - {fail['object_name']}/{fail['image_name']}: {fail['reason']}")
    


if __name__ == "__main__":
    main()

