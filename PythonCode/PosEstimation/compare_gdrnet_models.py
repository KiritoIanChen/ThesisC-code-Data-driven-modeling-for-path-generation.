#!/usr/bin/env python3
"""
对比LineMOD 13类和LMO 8类GDR-Net模型的输出
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 添加路径
PIPELINE_DIR = Path(r"d:\projects\odb_box\yolo-maskrcnn-gdr\linemod_6d_pose_v2")
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "pose_estimation"))

from gdrnet_pure_inference_20251016 import GDRNetPureInference

def test_model(model_path, num_classes, model_name, rgb, bbox, object_id):
    """测试单个模型"""
    print(f"\n{'='*80}")
    print(f"🔍 测试模型: {model_name}")
    print(f"{'='*80}")
    print(f"   模型路径: {Path(model_path).name}")
    print(f"   类别数: {num_classes}")
    
    # 加载模型
    try:
        gdrnet = GDRNetPureInference(
            model_path=str(model_path),
            device='cuda',
            num_classes=num_classes
        )
        print(f"✓ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 运行推理
    print(f"\n运行推理...")
    print(f"   bbox: {bbox}")
    print(f"   object_id: {object_id}")
    
    try:
        result = gdrnet.estimate_pose(rgb, mask=None, bbox=bbox, object_id=object_id)
        
        if result is None or len(result) < 4:
            print(f"❌ 推理失败")
            return None
        
        rotation, translation, confidence, mask = result
        
        print(f"\n✓ 推理成功")
        print(f"   rotation shape: {rotation.shape}")
        print(f"   translation: {translation}")
        print(f"   confidence: {confidence}")
        
        if mask is not None:
            print(f"\n   Mask分析:")
            print(f"     shape: {mask.shape}")
            print(f"     dtype: {mask.dtype}")
            print(f"     range: [{mask.min():.3f}, {mask.max():.3f}]")
            print(f"     mean: {mask.mean():.3f}")
            
            # 统计
            print(f"\n   Mask覆盖:")
            print(f"     == 0: {np.sum(mask == 0)/mask.size*100:.1f}%")
            print(f"     > 0: {np.sum(mask > 0)/mask.size*100:.1f}%")
            print(f"     > 127: {np.sum(mask > 127)/mask.size*100:.1f}%")
            print(f"     == 255: {np.sum(mask == 255)/mask.size*100:.1f}%")
        else:
            print(f"\n   ❌ 没有mask输出")
        
        return {
            'rotation': rotation,
            'translation': translation,
            'confidence': confidence,
            'mask': mask,
            'model_name': model_name
        }
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("="*80)
    print("🔬 GDR-Net模型对比测试")
    print("="*80)
    
    # 加载测试图像
    test_image = r'd:\projects\odb_box\dataset\Linemod_preprocessed\Linemod_preprocessed\data\01\rgb\0000.png'
    print(f"\n📸 加载测试图像: {Path(test_image).name}")
    
    image = cv2.imread(test_image)
    if image is None:
        print(f"❌ 无法加载图像")
        return
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✓ 图像尺寸: {rgb.shape}")
    
    # 设置bbox和object_id
    h, w = image.shape[:2]
    bbox = [w//4, h//4, w*3//4, h*3//4]
    object_id = 1  # ape
    
    print(f"\n测试参数:")
    print(f"   bbox: {bbox}")
    print(f"   object_id: {object_id} (ape)")
    
    # 测试两个模型
    models = [
        {
            'path': r"d:\projects\odb_box\yolo-maskrcnn-gdr\models\gdr\gdrn\lm\a6_cPnP_lm13\gdrn_lm.pth",
            'num_classes': 13,
            'name': 'LineMOD 13类'
        },
        {
            'path': r"d:\projects\odb_box\yolo-maskrcnn-gdr\models\gdr\gdrn\lmo\a6_cPnP_AugAAETrunc_BG0.5_lmo_real_pbr0.1_40e\gdrn_lmo_real_pbr.pth",
            'num_classes': 8,
            'name': 'LMO 8类'
        }
    ]
    
    results = []
    for model_info in models:
        model_path = Path(model_info['path'])
        if not model_path.exists():
            print(f"\n⚠️  模型不存在: {model_path.name}")
            continue
        
        result = test_model(
            model_path=model_path,
            num_classes=model_info['num_classes'],
            model_name=model_info['name'],
            rgb=rgb,
            bbox=bbox,
            object_id=object_id
        )
        
        if result:
            results.append(result)
    
    # 可视化对比
    if len(results) > 0:
        print(f"\n{'='*80}")
        print("📊 生成对比可视化")
        print("="*80)
        
        n_models = len(results)
        fig, axes = plt.subplots(2, n_models + 1, figsize=(6*(n_models+1), 12))
        
        # 第一行：原图 + 各模型mask
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        for i, result in enumerate(results):
            mask = result['mask']
            model_name = result['model_name']
            
            if mask is not None:
                axes[0, i+1].imshow(mask, cmap='viridis')
                coverage = np.sum(mask > 127) / mask.size * 100
                axes[0, i+1].set_title(f"{model_name}\nMask (覆盖率: {coverage:.1f}%)", 
                                      fontsize=12, fontweight='bold')
            else:
                axes[0, i+1].text(0.5, 0.5, 'No Mask', ha='center', va='center', fontsize=16)
                axes[0, i+1].set_title(f"{model_name}\n(无Mask)", fontsize=12)
            axes[0, i+1].axis('off')
        
        # 第二行：mask叠加到原图
        axes[1, 0].imshow(rgb)
        axes[1, 0].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                                          fill=False, edgecolor='green', linewidth=2))
        axes[1, 0].set_title('Input BBox', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        for i, result in enumerate(results):
            mask = result['mask']
            model_name = result['model_name']
            translation = result['translation']
            confidence = result['confidence']
            
            if mask is not None:
                # 叠加mask到bbox区域
                x1, y1, x2, y2 = bbox
                bbox_h, bbox_w = y2 - y1, x2 - x1
                mask_resized = cv2.resize(mask, (bbox_w, bbox_h))
                
                overlay = rgb.copy()
                roi = overlay[y1:y2, x1:x2]
                
                # 创建彩色mask
                mask_colored = np.zeros_like(roi)
                mask_colored[mask_resized > 127] = [255, 0, 0]  # 红色
                
                roi_with_mask = cv2.addWeighted(roi, 0.6, mask_colored, 0.4, 0)
                overlay[y1:y2, x1:x2] = roi_with_mask
                
                axes[1, i+1].imshow(overlay)
                axes[1, i+1].set_title(f"{model_name}\nT=[{translation[0]:.1f}, {translation[1]:.1f}, {translation[2]:.1f}]\nConf={confidence:.3f}",
                                      fontsize=11, fontweight='bold')
            else:
                axes[1, i+1].imshow(rgb)
                axes[1, i+1].set_title(f"{model_name}\n(无Mask)", fontsize=12)
            axes[1, i+1].axis('off')
        
        plt.tight_layout()
        output_path = Path(r'd:\projects\odb_box\test_results\gdrnet_model_comparison.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 对比图保存到: {output_path}")
        
        plt.show()
    
    # 总结
    print(f"\n{'='*80}")
    print("🎯 对比总结")
    print("="*80)
    
    if len(results) == 2:
        print(f"\n模型对比:")
        for i, result in enumerate(results):
            mask = result['mask']
            print(f"\n{i+1}. {result['model_name']}:")
            print(f"   Translation: {result['translation']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            if mask is not None:
                print(f"   Mask覆盖率: {np.sum(mask > 127)/mask.size*100:.1f}%")
                print(f"   Mask质量: {'✓ 合理' if np.sum(mask > 127)/mask.size > 0.05 else '❌ 太少'}")
            else:
                print(f"   Mask: 无")
        
        print(f"\n💡 结论:")
        print(f"   - 如果两个模型的mask都不对 → GDR-Net的mask输出可能不适合直接使用")
        print(f"   - 如果LineMOD模型的mask正常 → 说明LMO模型有问题")
        print(f"   - 建议：专注于pose estimation（R和T），忽略mask输出")
        print(f"   - 或者：使用YOLO的检测框生成简单的矩形mask")

if __name__ == '__main__':
    main()

