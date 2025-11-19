#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化指定图片的检测结果
展示使用低阈值时，模型在多物体场景下的检测能力
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO

# LineMOD 13类物体映射
LINEMOD_CLASSES = {
    '01': 'ape',
    '02': 'benchvise', 
    '04': 'camera',
    '05': 'can',
    '06': 'cat',
    '08': 'driller',
    '09': 'duck',
    '10': 'eggbox',
    '11': 'glue',
    '12': 'holepuncher',
    '13': 'iron',
    '14': 'lamp',
    '15': 'phone'
}

# YOLO类别映射
YOLO_CLASS_TO_NAME = {
    0: 'ape',
    1: 'benchvise',
    2: 'camera',
    3: 'can',
    4: 'cat',
    5: 'driller',
    6: 'duck',
    7: 'eggbox',
    8: 'glue',
    9: 'holepuncher',
    10: 'iron',
    11: 'lamp',
    12: 'phone'
}

# 颜色映射（BGR格式）
COLORS = [
    (255, 0, 0),      # 蓝色
    (0, 255, 0),      # 绿色
    (0, 0, 255),      # 红色
    (255, 255, 0),    # 青色
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 黄色
    (128, 0, 0),      # 深蓝
    (0, 128, 0),      # 深绿
    (0, 0, 128),      # 深红
    (128, 128, 0),    # 深青
    (128, 0, 128),    # 深品红
    (0, 128, 128),    # 深黄
    (192, 192, 192),  # 银色
]


def load_ground_truth(class_id, img_idx):
    """加载ground truth信息"""
    dataset_path = 'dataset/Linemod_preprocessed/Linemod_preprocessed/data'
    gt_file = Path(dataset_path) / class_id / 'gt.yml'
    
    if not gt_file.exists():
        return None
    
    with open(gt_file, 'r') as f:
        gt_data = yaml.safe_load(f)
    
    if img_idx not in gt_data:
        return None
    
    return gt_data[img_idx][0] if gt_data[img_idx] else None


def visualize_detection(class_id, img_idx, conf_threshold=0.05, iou_threshold=0.2, save_output=True):
    """
    可视化指定图片的检测结果
    
    Args:
        class_id: 类别ID (如 '01')
        img_idx: 图片索引 (如 735)
        conf_threshold: 置信度阈值
        iou_threshold: NMS阈值
        save_output: 是否保存结果图片
    """
    # 构建图片路径
    dataset_path = Path('dataset/Linemod_preprocessed/Linemod_preprocessed/data')
    img_path = dataset_path / class_id / 'rgb' / f'{img_idx:04d}.png'
    
    if not img_path.exists():
        print(f"错误: 图片不存在 {img_path}")
        return
    
    # 加载图片
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"错误: 无法读取图片 {img_path}")
        return
    
    # 获取ground truth
    gt_info = load_ground_truth(class_id, img_idx)
    gt_class_name = LINEMOD_CLASSES[class_id]
    
    print(f"\n{'='*80}")
    print(f"图片: {img_path}")
    print(f"标注物体: {gt_class_name} (class {class_id})")
    if gt_info:
        print(f"Ground Truth BBox: {gt_info.get('obj_bb', 'N/A')}")
    print(f"{'='*80}\n")
    
    # 加载YOLO模型
    model_path = 'yolo_linemod_training/runs/train/linemod_multi_object/weights/best.pt'
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 检测
    print(f"检测中... (conf={conf_threshold}, iou={iou_threshold})")
    results = model.predict(
        img,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=100,
        verbose=False
    )
    
    # 绘制结果
    vis_img = img.copy()
    
    # 绘制ground truth边界框（如果有）
    if gt_info and 'obj_bb' in gt_info:
        x, y, w, h = gt_info['obj_bb']
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(vis_img, f'GT: {gt_class_name}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # 统计检测结果
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].cpu().numpy()
            
            class_name = YOLO_CLASS_TO_NAME.get(cls_id, f"class_{cls_id}")
            
            detections.append({
                'class_id': cls_id,
                'class_name': class_name,
                'confidence': conf,
                'bbox': bbox
            })
            
            # 绘制检测框
            x1, y1, x2, y2 = map(int, bbox)
            color = COLORS[cls_id % len(COLORS)]
            
            # 判断是否是标注的物体
            is_gt_object = (class_name == gt_class_name)
            thickness = 3 if is_gt_object else 2
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
            
            # 标签
            label = f'{class_name}: {conf:.2f}'
            if is_gt_object:
                label = f'[GT] {label}'
            
            # 背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_img, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
            
            # 文字
            cv2.putText(vis_img, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 打印检测结果
    print(f"\n检测结果: 共检测到 {len(detections)} 个物体")
    print(f"{'='*80}")
    
    gt_detected = False
    for i, det in enumerate(detections, 1):
        is_gt = '✓ [标注物体]' if det['class_name'] == gt_class_name else '  [背景物体]'
        print(f"{i}. {det['class_name']:15s} - 置信度: {det['confidence']:.3f} {is_gt}")
        if det['class_name'] == gt_class_name:
            gt_detected = True
    
    if not gt_detected:
        print(f"\n⚠️ 警告: 标注物体 '{gt_class_name}' 未被检测到!")
    
    print(f"{'='*80}\n")
    
    # 添加统计信息到图片
    stats_y = 30
    cv2.putText(vis_img, f"Ground Truth: {gt_class_name}", (10, stats_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    stats_y += 30
    cv2.putText(vis_img, f"Detections: {len(detections)} objects", (10, stats_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    stats_y += 30
    cv2.putText(vis_img, f"Conf >= {conf_threshold}", (10, stats_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 保存或显示
    if save_output:
        output_file = f'detection_vis_{class_id}_{img_idx:04d}.jpg'
        cv2.imwrite(output_file, vis_img)
        print(f"✓ 结果已保存到: {output_file}")
    
    # 显示（注释掉，避免在无GUI环境下出错）
    # cv2.imshow('Detection Result', vis_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return vis_img, detections


if __name__ == '__main__':
    import sys
    
    # 检测参数
    conf_threshold = 0.05
    iou_threshold = 0.2
    
    # 要检测的图片列表
    test_images = [
        ('01', 735),  # ape, 图片735
        ('13', 151),  # iron, 图片151
    ]
    
    print(f"\n{'='*80}")
    print(f"批量检测 {len(test_images)} 张图片")
    print(f"置信度阈值: {conf_threshold}")
    print(f"IOU阈值: {iou_threshold}")
    print(f"{'='*80}\n")
    
    # 依次检测每张图片
    for i, (class_id, img_idx) in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] 检测图片...")
        visualize_detection(class_id, img_idx, conf_threshold, iou_threshold)
        print(f"\n{'='*80}\n")

