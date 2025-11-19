#!/usr/bin/env python3
"""
训练LMO 8类物体检测模型
使用真实多物体场景数据（LMO）
"""

from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    print("="*80)
    print("🚀 训练 LMO 8类物体检测模型")
    print("="*80)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️  未检测到GPU，将使用CPU训练（会很慢）")
    
    # 配置
    DATA_YAML = r'd:\projects\odb_box\yolo_linemod_training\lmo_yolo_dataset\dataset.yaml'
    MODEL_SIZE = 'yolov8s.pt'  # 6GB GPU使用small
    BATCH = 8
    EPOCHS = 150
    
    print(f"\n训练配置:")
    print(f"  数据集: {DATA_YAML}")
    print(f"  模型: {MODEL_SIZE}")
    print(f"  Batch: {BATCH}")
    print(f"  Epochs: {EPOCHS}")
    print(f"\n数据集特点:")
    print(f"  ✅ 训练集: 1214张 LMO真实多物体图像（平均7.6个物体/图）")
    print(f"  ✅ 测试集: 400张 LineMOD单物体图像（8类 x 50张）")
    print(f"  ✅ 8个类别: ape, can, cat, driller, duck, eggbox, glue, holepuncher")
    
    # 加载模型
    print(f"\n{'='*80}")
    print("📦 加载预训练模型")
    print("="*80)
    
    model = YOLO(MODEL_SIZE)
    print(f"✅ 已加载: {MODEL_SIZE}")
    
    # 开始训练
    print(f"\n{'='*80}")
    print("🎯 开始训练")
    print("="*80)
    
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=640,
            patience=50,
            device=0,
            name='lmo_8classes',
            project='runs/train',
            
            # 真实数据，减少数据增强
            mosaic=0.5,        # 降低Mosaic（真实数据已经是多物体）
            mixup=0.0,         # 不使用MixUp
            copy_paste=0.0,    # 不使用Copy-Paste
            
            # 其他参数
            workers=4,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=False,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,   # 最后10个epoch关闭Mosaic
            amp=True,          # 自动混合精度
            fraction=1.0,
            profile=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=True,
            save=True,
            save_period=-1,
            cache=False,
            resume=False,
            
            # 学习率
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # HSV数据增强（适度）
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            
            # 空间增强（适度）
            degrees=0.0,       # 不旋转（真实场景已经有各种角度）
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            
            # NMS
            iou=0.7,
            max_det=300,       # 最多检测300个物体
        )
        
        print(f"\n{'='*80}")
        print("✅ 训练完成！")
        print("="*80)
        
        # 显示结果
        best_model_path = Path('runs/train/lmo_8classes/weights/best.pt')
        if best_model_path.exists():
            print(f"\n最佳模型: {best_model_path}")
            print(f"\n训练结果:")
            print(f"  - 检查点: runs/train/lmo_8classes/weights/")
            print(f"  - 曲线图: runs/train/lmo_8classes/")
            print(f"  - 验证结果: runs/train/lmo_8classes/val_batch*_pred.jpg")
        
        print(f"\n{'='*80}")
        print("🎯 下一步: 测试模型")
        print("="*80)
        print(f"""
测试命令：

# 1. 在LMO测试集上测试（多物体场景）
python test_lmo_model.py

# 2. 对比合成数据模型 vs LMO模型
python compare_models.py

预期：
✅ LMO模型在真实多物体场景上应该表现更好
✅ 精度应该高于合成数据训练的模型
✅ 检测框质量和准确性都会提升
""")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        
        if "CUDA out of memory" in str(e):
            print(f"\n💡 建议:")
            print(f"  1. 减少batch size: batch=4 或 batch=2")
            print(f"  2. 使用更小的模型: yolov8n.pt")
            print(f"  3. 减少图像大小: imgsz=512")

if __name__ == '__main__':
    main()

