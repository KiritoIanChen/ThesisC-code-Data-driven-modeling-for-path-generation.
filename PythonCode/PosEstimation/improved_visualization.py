#!/usr/bin/env python3
"""
改进的可视化函数，用于YOLO+GDR-Net Pipeline
- 显示分割mask
- 显示3D坐标轴
- 显示详细的姿态信息
"""

import cv2
import numpy as np
from typing import Dict, List


def visualize_results_improved(
    image: np.ndarray,
    results: Dict,
    show_mask: bool = True,
    show_pose: bool = True,
    show_confidence: bool = True
) -> np.ndarray:
    """
    改进的结果可视化
    
    Args:
        image: BGR图像
        results: Pipeline输出结果
        show_mask: 是否显示分割mask
        show_pose: 是否显示姿态坐标轴
        show_confidence: 是否显示置信度
    
    Returns:
        可视化后的图像
    """
    vis_image = image.copy()
    
    # 1. 绘制分割mask（半透明覆盖）
    if show_mask and 'poses' in results:
        mask_overlay = vis_image.copy()
        
        for i, pose in enumerate(results['poses']):
            if 'mask' in pose and pose['mask'] is not None:
                mask = pose['mask']
                
                # 确保mask尺寸匹配
                if mask.shape[:2] != vis_image.shape[:2]:
                    mask = cv2.resize(mask, (vis_image.shape[1], vis_image.shape[0]))
                
                # 生成随机颜色
                color = get_color_for_index(i)
                
                # 创建彩色mask
                mask_bool = mask > 0
                mask_overlay[mask_bool] = (
                    mask_overlay[mask_bool] * 0.6 + 
                    np.array(color) * 0.4
                ).astype(np.uint8)
                
                # 绘制轮廓
                contours, _ = cv2.findContours(
                    (mask > 0).astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis_image, contours, -1, color, 2)
        
        # 混合
        cv2.addWeighted(mask_overlay, 0.5, vis_image, 0.5, 0, vis_image)
    
    # 2. 绘制检测框
    for det in results['detections']:
        x1, y1, x2, y2 = det['bbox']
        
        # 框的颜色
        color = (0, 255, 0)  # 绿色
        
        # 绘制边界框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # 绘制类别标签和置信度
        if show_confidence:
            label = f"{det['class_name']}: {det['confidence']:.2f}"
        else:
            label = det['class_name']
        
        # 标签背景
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            vis_image,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        
        # 标签文字
        cv2.putText(
            vis_image, label, (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )
    
    # 3. 绘制姿态信息
    if show_pose and 'poses' in results:
        for pose in results['poses']:
            if pose.get('confidence', 0) < 0.1:
                continue  # 跳过低置信度的姿态
            
            det = pose['detection']
            x1, y1, x2, y2 = det['bbox']
            
            # 计算中心点
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # 坐标轴长度
            axis_length = min(x2 - x1, y2 - y1) // 3
            
            # 绘制3D坐标轴
            # X轴（红色）
            cv2.arrowedLine(
                vis_image, (cx, cy), (cx + axis_length, cy),
                (0, 0, 255), 3, tipLength=0.3
            )
            cv2.putText(
                vis_image, 'X', (cx + axis_length + 5, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
            )
            
            # Y轴（绿色）
            cv2.arrowedLine(
                vis_image, (cx, cy), (cx, cy + axis_length),
                (0, 255, 0), 3, tipLength=0.3
            )
            cv2.putText(
                vis_image, 'Y', (cx, cy + axis_length + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            
            # Z轴（蓝色）- 向上
            cv2.arrowedLine(
                vis_image, (cx, cy), (cx, cy - axis_length),
                (255, 0, 0), 3, tipLength=0.3
            )
            cv2.putText(
                vis_image, 'Z', (cx, cy - axis_length - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
            )
            
            # 显示姿态置信度和平移信息
            if show_confidence:
                translation = pose.get('translation', [0, 0, 0])
                pose_info = [
                    f"Pose: {pose['confidence']:.2f}",
                    f"T: [{translation[0]:.1f}, {translation[1]:.1f}, {translation[2]:.1f}]"
                ]
                
                y_offset = y2 + 20
                for info_line in pose_info:
                    cv2.putText(
                        vis_image, info_line, (x1, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                    )
                    y_offset += 15
    
    return vis_image


def get_color_for_index(index: int) -> tuple:
    """
    为给定索引生成一个独特的颜色
    
    Args:
        index: 物体索引
    
    Returns:
        (B, G, R) 颜色元组
    """
    # 预定义的颜色列表
    colors = [
        (255, 0, 0),    # 蓝
        (0, 255, 0),    # 绿
        (0, 0, 255),    # 红
        (255, 255, 0),  # 青
        (255, 0, 255),  # 品红
        (0, 255, 255),  # 黄
        (128, 0, 255),  # 紫
        (255, 128, 0),  # 橙
        (0, 255, 128),  # 春绿
        (128, 255, 0),  # 黄绿
        (255, 0, 128),  # 玫红
        (0, 128, 255),  # 天蓝
    ]
    
    return colors[index % len(colors)]


# 将这个函数添加到Pipeline中
def patch_pipeline_visualization(pipeline):
    """
    给Pipeline打补丁，使用改进的可视化
    
    Args:
        pipeline: YOLOGDRNetPipeline实例
    """
    original_visualize = pipeline.visualize_results
    
    def improved_visualize(image, results):
        return visualize_results_improved(image, results)
    
    pipeline.visualize_results = improved_visualize
    return pipeline

