#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量多物体6D姿态估计演示
使用多张图片（多种类）运行完整流程，并保存到output_vis文件夹

"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import logging
from typing import List
import json
import time

# 添加路径
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR / "pose_estimation"))

# 导入完整流程
from complete_pipeline_with_maskrcnn import CompletePipeline, CONFIG

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def get_test_images() -> List[Path]:
    """
    获取测试图像列表（包括测试图片和LineMOD数据集样本）
    """
    test_images = []
    
    # 1. 项目根目录的测试图片
    root_dir = CURRENT_DIR.parent.parent
    test_image_files = [
        root_dir / "test_image.jpg",
        root_dir / "test_image_1.jpg",
        root_dir / "test_image_2.jpg",
        root_dir / "test_image_3.jpg",
    ]
    
    for img_path in test_image_files:
        if img_path.exists():
            test_images.append(img_path)
    
    # 2. LineMOD数据集样本（选择不同物体的样本）
    dataset_path = root_dir / "dataset" / "Linemod_preprocessed" / "Linemod_preprocessed" / "data"
    
    if dataset_path.exists():
        # 选择多个不同物体的样本
        linemod_samples = [
            (1, "0000.png"),   # ape
            (2, "0000.png"),   # benchvise
            (5, "0000.png"),   # can
            (6, "0000.png"),   # cat
            (8, "0000.png"),   # driller
            (9, "0000.png"),   # duck
            (11, "0000.png"),  # glue
            (15, "0000.png"),  # phone
        ]
        
        for obj_id, img_name in linemod_samples:
            img_path = dataset_path / f"{obj_id:02d}" / "rgb" / img_name
            if img_path.exists():
                test_images.append(img_path)
    
    return test_images


def main():
    """主函数"""
    
    logger.info("="*80)
    logger.info("🚀 批量多物体6D姿态估计演示")
    logger.info("="*80)
    
    # 创建输出文件夹
    output_dir = CURRENT_DIR / "output_vis"
    output_dir.mkdir(exist_ok=True)
    logger.info(f"📁 输出文件夹: {output_dir}")
    
    # 初始化pipeline
    logger.info("\n初始化Pipeline...")
    pipeline = CompletePipeline(CONFIG)
    
    # 获取测试图像
    test_images = get_test_images()
    logger.info(f"\n📸 找到 {len(test_images)} 张测试图像")
    
    if len(test_images) == 0:
        logger.error("❌ 未找到测试图像！")
        return
    
    # 统计信息
    total_detections = 0
    total_poses = 0
    processing_times = []
    results_summary = []
    
    # 批量处理
    logger.info("\n" + "="*80)
    logger.info("开始批量处理...")
    logger.info("="*80)
    
    for idx, image_path in enumerate(test_images):
        logger.info(f"\n{'='*80}")
        logger.info(f"🖼️  [{idx+1}/{len(test_images)}] 处理: {image_path.name}")
        logger.info(f"{'='*80}")
        
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"   ❌ 无法加载图像: {image_path}")
            continue
        
        logger.info(f"   图像尺寸: {image.shape[1]}x{image.shape[0]}")
        
        try:
            # 处理图像
            start_time = time.time()
            results = pipeline.process_image(image, visualize=True)
            processing_time = time.time() - start_time
            
            # 统计
            num_detections = len(results['detections'])
            num_poses = len(results['poses'])
            total_detections += num_detections
            total_poses += num_poses
            processing_times.append(processing_time)
            
            # 保存结果
            output_name = f"{image_path.stem}_{idx:04d}"
            output_path = output_dir / output_name
            
            # 保存可视化图像
            vis_path = output_path.with_suffix('.jpg')
            cv2.imwrite(str(vis_path), results['visualization'])
            logger.info(f"   ✅ 可视化结果已保存: {vis_path.name}")
            
            # 保存JSON结果
            json_path = output_path.with_suffix('.json')
            json_results = {
                'source_image': str(image_path),
                'image_shape': results['image_shape'],
                'num_detections': num_detections,
                'num_poses': num_poses,
                'detections': results['detections'],
                'poses': [
                    {
                        'detection_index': p['detection_index'],
                        'object_id': p['object_id'],
                        'class_name': p['class_name'],
                        'rotation': p['rotation'].tolist(),
                        'translation': p['translation'].tolist(),
                        'confidence': p['confidence'],
                        'has_mask': p['has_mask'],
                        'mask_source': p.get('mask_source', 'Unknown')
                    }
                    for p in results.get('poses', [])
                ],
                'processing_time': results['processing_time']
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            # 结果摘要
            results_summary.append({
                'image': image_path.name,
                'detections': num_detections,
                'poses': num_poses,
                'time': processing_time,
                'output': vis_path.name
            })
            
            logger.info(f"   📊 检测: {num_detections}, 姿态: {num_poses}, 时间: {processing_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印总结
    logger.info("\n" + "="*80)
    logger.info("📊 批量处理完成 - 总结")
    logger.info("="*80)
    logger.info(f"✅ 成功处理: {len(results_summary)}/{len(test_images)} 张图像")
    logger.info(f"🔍 总检测数: {total_detections}")
    logger.info(f"🎯 总姿态数: {total_poses}")
    
    if processing_times:
        avg_time = np.mean(processing_times)
        logger.info(f"⏱️  平均处理时间: {avg_time:.2f}秒/图像")
    
    logger.info(f"\n📁 所有结果已保存到: {output_dir}")
    
    # 保存总结报告
    summary_path = output_dir / "batch_summary.json"
    summary_data = {
        'total_images': len(test_images),
        'processed_images': len(results_summary),
        'total_detections': total_detections,
        'total_poses': total_poses,
        'average_time': float(np.mean(processing_times)) if processing_times else 0,
        'results': results_summary
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 总结报告已保存: {summary_path.name}")
    
    # 显示结果列表
    logger.info("\n" + "="*80)
    logger.info("🖼️  结果列表:")
    logger.info("="*80)
    for i, res in enumerate(results_summary, 1):
        logger.info(f"{i:2d}. {res['image']:<30s} → {res['output']:<40s} "
                   f"(检测:{res['detections']}, 姿态:{res['poses']}, {res['time']:.1f}s)")
    
    logger.info("\n" + "="*80)
    logger.info("✨ 完成！所有结果已保存到 output_vis 文件夹")
    logger.info("="*80)


if __name__ == "__main__":
    main()

