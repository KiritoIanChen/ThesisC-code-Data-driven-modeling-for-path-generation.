#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LineMOD 单目标检测统计脚本
从13类物体中各随机选取100张图片，使用低阈值检测，统计结果写入Excel

重要说明：
- LineMOD数据集每张图片中实际有多个物体
- 但训练标注时，每张图片只标注了一个目标物体
- 其他物体作为背景/干扰物存在但未标注

统计内容：
- 图片标注的目标物体名称（ground truth）
- 每张图片检测出的物体总数（包括标注物体和背景物体）
- 13类物体各自检测出来的数量
- 标注物体的检出率（主要关注指标）
- 背景物体的额外检出情况（不算误检，可能是真实存在的物体）
"""

import os
import sys
import yaml
import cv2
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# YOLO检测
from ultralytics import YOLO

# 配置参数
CONFIG = {
    'dataset_path': 'dataset/Linemod_preprocessed/Linemod_preprocessed/data',
    'yolo_model_path': 'yolo_linemod_training/runs/train/linemod_multi_object/weights/best.pt',
    'output_excel': f'linemod_detection_statistics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
    
    # 检测参数 - 尽可能低的阈值
    'conf_threshold': 0.05,  # 极低的置信度阈值
    'iou_threshold': 0.2,    # 极低的NMS阈值，允许更多重叠
    'max_det': 100,          # 增加最大检测数
    
    # 随机采样参数
    'samples_per_class': 100,
    'random_seed': 42,
}

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

# YOLO类别映射（根据训练时的顺序）
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


def load_ground_truth(gt_file_path):
    """加载ground truth标注信息"""
    if not os.path.exists(gt_file_path):
        return {}
    
    with open(gt_file_path, 'r') as f:
        gt_data = yaml.safe_load(f)
    
    return gt_data


def get_image_ground_truth(gt_data, img_idx, class_id):
    """获取特定图片的ground truth物体ID"""
    if img_idx not in gt_data:
        return None
    
    gt_list = gt_data[img_idx]
    if not gt_list or len(gt_list) == 0:
        return None
    
    # 返回物体ID（应该是当前类别的ID）
    obj_id = gt_list[0].get('obj_id', int(class_id))
    return obj_id


def select_random_images(dataset_path, samples_per_class, random_seed=42):
    """从每个类别中随机选取指定数量的图片"""
    random.seed(random_seed)
    selected_images = []
    
    for class_id, class_name in LINEMOD_CLASSES.items():
        class_dir = os.path.join(dataset_path, class_id)
        rgb_dir = os.path.join(class_dir, 'rgb')
        
        if not os.path.exists(rgb_dir):
            print(f"警告: {class_name} ({class_id}) 目录不存在")
            continue
        
        # 获取所有图片
        all_images = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
        
        # 随机采样
        if len(all_images) < samples_per_class:
            print(f"警告: {class_name} 只有 {len(all_images)} 张图片，少于 {samples_per_class}")
            selected = all_images
        else:
            selected = random.sample(all_images, samples_per_class)
        
        # 添加到结果列表
        for img_file in selected:
            img_idx = int(os.path.splitext(img_file)[0])
            selected_images.append({
                'class_id': class_id,
                'class_name': class_name,
                'image_file': img_file,
                'image_idx': img_idx,
                'image_path': os.path.join(rgb_dir, img_file),
                'gt_file': os.path.join(class_dir, 'gt.yml')
            })
    
    print(f"\n总共选择了 {len(selected_images)} 张图片")
    for class_id, class_name in LINEMOD_CLASSES.items():
        count = sum(1 for img in selected_images if img['class_id'] == class_id)
        print(f"  {class_name}: {count} 张")
    
    return selected_images


def detect_objects(model, image_path, conf_threshold, iou_threshold, max_det):
    """使用YOLO模型检测物体"""
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    # 检测
    results = model.predict(
        img,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=max_det,
        verbose=False
    )
    
    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            
            # 获取类别名称
            class_name = YOLO_CLASS_TO_NAME.get(cls_id, f"class_{cls_id}")
            
            detections.append({
                'class_id': cls_id,
                'class_name': class_name,
                'confidence': conf
            })
    
    return detections


def run_statistics():
    """运行统计"""
    print("=" * 80)
    print("LineMOD 单目标检测统计")
    print("=" * 80)
    
    # 检查数据集路径
    dataset_path = CONFIG['dataset_path']
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return
    
    # 检查模型路径
    model_path = CONFIG['yolo_model_path']
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return
    
    print(f"\n配置:")
    print(f"  数据集路径: {dataset_path}")
    print(f"  YOLO模型: {model_path}")
    print(f"  置信度阈值: {CONFIG['conf_threshold']}")
    print(f"  IOU阈值: {CONFIG['iou_threshold']}")
    print(f"  最大检测数: {CONFIG['max_det']}")
    print(f"  每类采样数: {CONFIG['samples_per_class']}")
    
    # 加载YOLO模型
    print("\n加载YOLO模型...")
    model = YOLO(model_path)
    print("模型加载完成")
    
    # 选择图片
    print("\n选择测试图片...")
    selected_images = select_random_images(
        dataset_path,
        CONFIG['samples_per_class'],
        CONFIG['random_seed']
    )
    
    if len(selected_images) == 0:
        print("错误: 没有选择到任何图片")
        return
    
    # 运行检测和统计
    print("\n开始检测...")
    results_data = []
    
    # 用于统计13类物体检测数量
    detection_count_by_class = defaultdict(int)
    
    for img_info in tqdm(selected_images, desc="检测进度"):
        # 加载ground truth
        gt_data = load_ground_truth(img_info['gt_file'])
        obj_id = get_image_ground_truth(gt_data, img_info['image_idx'], img_info['class_id'])
        
        # 检测
        detections = detect_objects(
            model,
            img_info['image_path'],
            CONFIG['conf_threshold'],
            CONFIG['iou_threshold'],
            CONFIG['max_det']
        )
        
        # 统计每类物体检测数量
        detected_classes = defaultdict(int)
        for det in detections:
            detected_classes[det['class_name']] += 1
            detection_count_by_class[det['class_name']] += 1
        
        # 构建结果行
        result_row = {
            '图片路径': img_info['image_path'],
            '类别ID': img_info['class_id'],
            '标注物体名称': img_info['class_name'],
            '图片编号': img_info['image_idx'],
            '检测物体总数': len(detections),
        }
        
        # 添加13类物体各自检测数量
        for class_name in YOLO_CLASS_TO_NAME.values():
            result_row[f'检测_{class_name}_数量'] = detected_classes.get(class_name, 0)
        
        # 添加检测详情（可选）
        if len(detections) > 0:
            det_details = '; '.join([
                f"{det['class_name']}({det['confidence']:.2f})" 
                for det in detections
            ])
            result_row['检测详情'] = det_details
        else:
            result_row['检测详情'] = '无检测'
        
        results_data.append(result_row)
    
    # 创建DataFrame
    df = pd.DataFrame(results_data)
    
    # 创建统计摘要
    summary_data = {
        '统计项': ['总图片数', '总检测数', '平均每张检测数'],
        '数值': [
            len(selected_images),
            df['检测物体总数'].sum(),
            df['检测物体总数'].mean()
        ]
    }
    
    # 添加每类物体的检测统计
    for class_name in YOLO_CLASS_TO_NAME.values():
        summary_data['统计项'].append(f'{class_name} 总检测数')
        summary_data['数值'].append(detection_count_by_class.get(class_name, 0))
    
    # 添加每类ground truth的检测率
    for class_id, class_name in LINEMOD_CLASSES.items():
        class_images = df[df['类别ID'] == class_id]
        if len(class_images) > 0:
            detected_in_own_class = class_images[f'检测_{class_name}_数量'].sum()
            total_images = len(class_images)
            summary_data['统计项'].append(f'{class_name} 在本类图片中检测数')
            summary_data['数值'].append(detected_in_own_class)
            summary_data['统计项'].append(f'{class_name} 在本类图片中检测率')
            summary_data['数值'].append(f"{detected_in_own_class/total_images:.2%}")
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存到Excel
    output_file = CONFIG['output_excel']
    print(f"\n保存结果到 {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 写入详细结果
        df.to_excel(writer, sheet_name='详细检测结果', index=False)
        
        # 写入统计摘要
        summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
        
        # 创建每类物体的统计
        class_stats_data = []
        for class_id, class_name in LINEMOD_CLASSES.items():
            class_images = df[df['类别ID'] == class_id]
            if len(class_images) > 0:
                stat_row = {
                    '类别': class_name,
                    '图片数': len(class_images),
                    '总检测数': class_images['检测物体总数'].sum(),
                    '平均检测数': class_images['检测物体总数'].mean(),
                    '本类检测数': class_images[f'检测_{class_name}_数量'].sum(),
                    '本类检测率': class_images[f'检测_{class_name}_数量'].sum() / len(class_images),
                }
                
                # 添加检测到其他类别的统计（可能是真实存在的背景物体）
                for other_name in YOLO_CLASS_TO_NAME.values():
                    if other_name != class_name:
                        count = class_images[f'检测_{other_name}_数量'].sum()
                        if count > 0:
                            stat_row[f'额外检出_{other_name}'] = count
                
                class_stats_data.append(stat_row)
        
        class_stats_df = pd.DataFrame(class_stats_data)
        class_stats_df.to_excel(writer, sheet_name='各类别统计', index=False)
    
    print(f"✓ 结果已保存到: {output_file}")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    print(f"总图片数: {len(selected_images)}")
    print(f"总检测数: {df['检测物体总数'].sum()}")
    print(f"平均每张检测数: {df['检测物体总数'].mean():.2f}")
    print("\n各类物体检测统计:")
    for class_name in YOLO_CLASS_TO_NAME.values():
        count = detection_count_by_class.get(class_name, 0)
        print(f"  {class_name:15s}: {count:5d} 次")
    
    print("\n各类物体在本类图片中的检测率:")
    for class_id, class_name in LINEMOD_CLASSES.items():
        class_images = df[df['类别ID'] == class_id]
        if len(class_images) > 0:
            detected_in_own_class = class_images[f'检测_{class_name}_数量'].sum()
            total_images = len(class_images)
            detection_rate = detected_in_own_class / total_images
            print(f"  {class_name:15s}: {detected_in_own_class:4d}/{total_images:4d} = {detection_rate:.2%}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    try:
        run_statistics()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

