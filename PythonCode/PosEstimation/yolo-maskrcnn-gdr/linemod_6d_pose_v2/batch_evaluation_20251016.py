#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量评估GDR-Net姿态估计
在LineMOD数据集上评估精度

Date: 2025-10-16
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import json

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.linemod_dataset_20251010 import LineMODDataset
from pose_estimation.gdrnet_estimator_20251010 import GDRNetPoseEstimator
from utils.model_loader_20251010 import ModelLoader
from visualization.pose_visualizer_20251010 import PoseVisualizer


def rotation_error(R_pred, R_gt):
    """计算旋转误差（度）"""
    R_diff = R_pred @ R_gt.T
    trace = np.trace(R_diff)
    error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    return np.degrees(error)


def translation_error(t_pred, t_gt):
    """计算平移误差（mm）"""
    return np.linalg.norm(t_pred - t_gt)


def batch_evaluate(object_id=1, num_samples=100, save_visualizations=False):
    """批量评估"""
    print("="*70)
    print("Batch Evaluation - GDR-Net 6D Pose Estimation")
    print("="*70)
    
    # 加载数据集
    dataset_root = PROJECT_ROOT.parent.parent / "dataset" / "Linemod_preprocessed" / "Linemod_preprocessed"
    dataset = LineMODDataset(str(dataset_root), object_id=object_id, load_depth=False)
    print(f"\n✓ Dataset loaded: {len(dataset)} frames")
    print(f"  Object: {dataset.object_name} (ID: {object_id})")
    
    # 加载3D模型
    models_dir = dataset_root / "models"
    loader = ModelLoader(str(models_dir))
    model_3d = loader.load_model(object_id=object_id)
    print(f"✓ 3D model loaded: {len(model_3d)} points")
    
    # 初始化估计器
    model_path = PROJECT_ROOT.parent / "models" / "gdr" / "gdrn" / "lm" / "a6_cPnP_lm13" / "gdrn_lm.pth"
    estimator = GDRNetPoseEstimator({
        'model_path': str(model_path),
        'device': 'cpu'
    })
    print("✓ GDR-Net estimator initialized")
    
    # 初始化可视化器
    visualizer = PoseVisualizer(camera_K=dataset.camera_K)
    
    # 评估
    num_samples = min(num_samples, len(dataset))
    print(f"\nEvaluating on {num_samples} samples...")
    
    results = {
        'rotation_errors': [],
        'translation_errors': [],
        'confidences': [],
        'methods': [],
        'success': 0,
        'failure': 0
    }
    
    output_dir = PROJECT_ROOT / "batch_results"
    output_dir.mkdir(exist_ok=True)
    
    for i in tqdm(range(num_samples), desc="Processing"):
        sample = dataset[i]
        
        try:
            # 估计姿态
            pose = estimator.estimate_pose(
                rgb=sample['rgb'],
                mask=sample['mask'],
                bbox=sample['bbox'],
                object_id=object_id
            )
            
            # 计算误差
            rot_err = rotation_error(pose.rotation_matrix, sample['pose']['R'])
            trans_err = translation_error(pose.translation, sample['pose']['t'])
            
            results['rotation_errors'].append(rot_err)
            results['translation_errors'].append(trans_err)
            results['confidences'].append(pose.confidence)
            results['methods'].append(pose.method)
            results['success'] += 1
            
            # 保存可视化（可选）
            if save_visualizations and i % 10 == 0:
                vis_img = visualizer.visualize_pose(
                    image=sample['rgb'].copy(),
                    pose=pose,
                    model_3d=model_3d
                )
                cv2.imwrite(str(output_dir / f"frame_{i:04d}.jpg"), vis_img)
            
        except Exception as e:
            print(f"\n✗ Frame {i} failed: {e}")
            results['failure'] += 1
    
    # 统计结果
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    
    print(f"\nDataset: LineMOD - {dataset.object_name}")
    print(f"Samples: {num_samples}")
    print(f"Success: {results['success']} ({results['success']/num_samples*100:.1f}%)")
    print(f"Failure: {results['failure']}")
    
    if results['rotation_errors']:
        rot_errors = np.array(results['rotation_errors'])
        trans_errors = np.array(results['translation_errors'])
        confidences = np.array(results['confidences'])
        
        print(f"\n--- Rotation Error ---")
        print(f"Mean:   {rot_errors.mean():.2f}°")
        print(f"Median: {np.median(rot_errors):.2f}°")
        print(f"Std:    {rot_errors.std():.2f}°")
        print(f"Min:    {rot_errors.min():.2f}°")
        print(f"Max:    {rot_errors.max():.2f}°")
        
        print(f"\n--- Translation Error ---")
        print(f"Mean:   {trans_errors.mean():.2f} mm")
        print(f"Median: {np.median(trans_errors):.2f} mm")
        print(f"Std:    {trans_errors.std():.2f} mm")
        print(f"Min:    {trans_errors.min():.2f} mm")
        print(f"Max:    {trans_errors.max():.2f} mm")
        
        print(f"\n--- Confidence ---")
        print(f"Mean:   {confidences.mean():.3f}")
        print(f"Median: {np.median(confidences):.3f}")
        
        print(f"\n--- Methods Used ---")
        from collections import Counter
        method_counts = Counter(results['methods'])
        for method, count in method_counts.items():
            print(f"{method}: {count} ({count/len(results['methods'])*100:.1f}%)")
        
        # ADD指标（如果可以计算）
        # ADD = Average Distance of Model Points
        
    # 保存结果
    results_file = output_dir / f"evaluation_results_{dataset.object_name}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'object_id': object_id,
            'object_name': dataset.object_name,
            'num_samples': num_samples,
            'success_rate': results['success'] / num_samples,
            'mean_rotation_error': float(np.mean(results['rotation_errors'])) if results['rotation_errors'] else None,
            'mean_translation_error': float(np.mean(results['translation_errors'])) if results['translation_errors'] else None,
            'mean_confidence': float(np.mean(results['confidences'])) if results['confidences'] else None,
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print("="*70)


if __name__ == "__main__":
    # 评估ape（object_id=1）
    batch_evaluate(object_id=1, num_samples=100, save_visualizations=True)
    
    # 可以评估所有物体
    # for obj_id in range(1, 16):  # LineMOD有15个物体
    #     if obj_id not in [3, 7]:  # 跳过没有的
    #         batch_evaluate(object_id=obj_id, num_samples=50)

