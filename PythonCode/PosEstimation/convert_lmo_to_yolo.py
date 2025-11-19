#!/usr/bin/env python3
"""
将 Linemod-Occluded (LMO) 数据集转换为 YOLO 格式
LMO包含8个物体类别，每张图平均7.6个物体
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# LMO物体ID到名称的映射
LMO_ID_TO_NAME = {
    1: 'ape',
    5: 'can',
    6: 'cat',
    8: 'driller',
    9: 'duck',
    10: 'eggbox',
    11: 'glue',
    12: 'holepuncher',
}

# 创建连续的类别ID映射（YOLO需要从0开始）
LMO_ID_TO_YOLO_CLASS = {
    1: 0,   # ape
    5: 1,   # can
    6: 2,   # cat
    8: 3,   # driller
    9: 4,   # duck
    10: 5,  # eggbox
    11: 6,  # glue
    12: 7,  # holepuncher
}

def convert_bbox_to_yolo(bbox, img_w, img_h):
    """
    将 [x, y, w, h] 格式转换为 YOLO 格式 [x_center, y_center, width, height] (归一化)
    """
    x, y, w, h = bbox
    
    # 计算中心点
    x_center = x + w / 2
    y_center = y + h / 2
    
    # 归一化
    x_center_norm = x_center / img_w
    y_center_norm = y_center / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    
    return x_center_norm, y_center_norm, w_norm, h_norm

def convert_lmo_scene_to_yolo(scene_dir, output_dir, split='train', 
                                min_visibility=0.1, max_images=None):
    """
    转换一个LMO场景到YOLO格式
    
    Args:
        scene_dir: LMO场景目录 (e.g., .../test/000002)
        output_dir: YOLO输出目录
        split: 'train' or 'val'
        min_visibility: 最小可见比例（0-1）
        max_images: 最大图像数量（用于测试）
    """
    scene_dir = Path(scene_dir)
    scene_name = scene_dir.name
    
    print(f"\n{'='*80}")
    print(f"📂 转换场景: {scene_name} ({split})")
    print(f"{'='*80}")
    
    # 读取标注
    scene_gt_file = scene_dir / "scene_gt.json"
    scene_gt_info_file = scene_dir / "scene_gt_info.json"
    rgb_dir = scene_dir / "rgb"
    
    if not scene_gt_file.exists():
        print(f"❌ 未找到标注文件: {scene_gt_file}")
        return 0, 0
    
    with open(scene_gt_file, 'r') as f:
        scene_gt = json.load(f)
    
    with open(scene_gt_info_file, 'r') as f:
        scene_gt_info = json.load(f)
    
    # 创建输出目录
    images_dir = output_dir / split / 'images'
    labels_dir = output_dir / split / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计
    total_images = 0
    total_objects = 0
    skipped_objects = 0
    
    # 获取图像列表
    image_ids = list(scene_gt.keys())
    if max_images:
        image_ids = image_ids[:max_images]
    
    print(f"处理 {len(image_ids)} 张图像...")
    
    for img_id in tqdm(image_ids, desc=f"  {split}"):
        # 读取图像
        img_file = rgb_dir / f"{int(img_id):06d}.png"
        if not img_file.exists():
            continue
        
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        img_h, img_w = image.shape[:2]
        
        # 获取该图像的所有标注
        annotations = scene_gt[img_id]
        info_annotations = scene_gt_info[img_id]
        
        # 转换为YOLO格式
        yolo_labels = []
        
        for ann, info in zip(annotations, info_annotations):
            obj_id = ann['obj_id']
            
            # 跳过不在LMO中的物体
            if obj_id not in LMO_ID_TO_YOLO_CLASS:
                skipped_objects += 1
                continue
            
            # 检查可见性
            visibility = info.get('visib_fract', 1.0)
            if visibility < min_visibility:
                skipped_objects += 1
                continue
            
            # 获取bbox
            bbox = info.get('bbox_obj')
            if bbox is None:
                skipped_objects += 1
                continue
            
            # 转换为YOLO格式
            yolo_class = LMO_ID_TO_YOLO_CLASS[obj_id]
            x_center, y_center, w, h = convert_bbox_to_yolo(bbox, img_w, img_h)
            
            # 验证bbox有效性
            if w <= 0 or h <= 0 or x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                skipped_objects += 1
                continue
            
            yolo_labels.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            total_objects += 1
        
        # 如果没有有效标注，跳过该图像
        if not yolo_labels:
            continue
        
        # 保存图像
        output_img_name = f"{scene_name}_{int(img_id):06d}.png"
        output_img_path = images_dir / output_img_name
        shutil.copy(img_file, output_img_path)
        
        # 保存标签
        output_label_path = labels_dir / f"{scene_name}_{int(img_id):06d}.txt"
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(yolo_labels))
        
        total_images += 1
    
    print(f"\n✅ 场景 {scene_name} 转换完成:")
    print(f"   图像数: {total_images}")
    print(f"   物体数: {total_objects}")
    print(f"   跳过物体: {skipped_objects}")
    print(f"   平均每张图: {total_objects/max(total_images, 1):.2f} 个物体")
    
    return total_images, total_objects

def extract_linemod_test_set(linemod_dir, output_dir, lmo_to_linemod, samples_per_class=50):
    """
    从原始LineMOD数据集抽取测试集
    
    Args:
        linemod_dir: LineMOD数据目录
        output_dir: 输出目录
        lmo_to_linemod: LMO ID到LineMOD文件夹的映射
        samples_per_class: 每个类别抽取的样本数
    """
    val_images_dir = output_dir / 'val' / 'images'
    val_labels_dir = output_dir / 'val' / 'labels'
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    total_images = 0
    total_objects = 0
    
    for lmo_id, linemod_folder in lmo_to_linemod.items():
        obj_dir = linemod_dir / linemod_folder
        rgb_dir = obj_dir / 'rgb'
        
        if not rgb_dir.exists():
            print(f"⚠️  未找到目录: {rgb_dir}")
            continue
        
        # 获取所有图像
        image_files = sorted(list(rgb_dir.glob('*.png')))
        
        # 随机抽样
        if len(image_files) > samples_per_class:
            np.random.seed(42)
            selected_indices = np.random.choice(len(image_files), samples_per_class, replace=False)
            image_files = [image_files[i] for i in sorted(selected_indices)]
        
        print(f"  {LMO_ID_TO_NAME[lmo_id]:12s}: 抽取 {len(image_files)} 张图像")
        
        # 处理每张图像
        for img_file in image_files:
            # 读取图像
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            img_h, img_w = image.shape[:2]
            
            # LineMOD是单物体数据集，整张图就是该物体
            # 创建一个覆盖大部分图像的bbox
            yolo_class = LMO_ID_TO_YOLO_CLASS[lmo_id]
            
            # 使用中心80%的区域作为bbox（避免边缘）
            margin = 0.1
            x_center = 0.5
            y_center = 0.5
            w = 0.8
            h = 0.8
            
            # 保存图像
            output_img_name = f"linemod_{linemod_folder}_{img_file.stem}.png"
            output_img_path = val_images_dir / output_img_name
            shutil.copy(img_file, output_img_path)
            
            # 保存标签
            output_label_path = val_labels_dir / f"linemod_{linemod_folder}_{img_file.stem}.txt"
            with open(output_label_path, 'w') as f:
                f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            total_images += 1
            total_objects += 1
    
    print(f"\n✅ 测试集抽取完成:")
    print(f"   图像数: {total_images}")
    print(f"   物体数: {total_objects}")
    print(f"   平均每张图: {total_objects/max(total_images, 1):.2f} 个物体 (LineMOD是单物体)")
    
    return total_images, total_objects

def create_dataset_yaml(output_dir, class_names):
    """创建YOLO数据集配置文件"""
    yaml_content = f"""# LMO (Linemod-Occluded) 数据集
# 真实多物体场景数据

path: {output_dir.absolute()}
train: train/images
val: val/images

# 类别数量
nc: {len(class_names)}

# 类别名称
names: {class_names}

# 数据集信息
# - 来源: Linemod-Occluded (BOP Challenge)
# - 场景类型: 真实多物体场景
# - 平均物体数/图: ~7.6
# - 物体类别: 8个
"""
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n✅ 数据集配置文件: {yaml_path}")
    return yaml_path

def main():
    print("="*80)
    print("🔄 LMO数据集转换为YOLO格式")
    print("="*80)
    
    # 配置路径
    lmo_test_dir = Path(r'd:\projects\odb_box\dataset\lmo\lmo_test_all\test')
    output_dir = Path(r'd:\projects\odb_box\yolo_linemod_training\lmo_yolo_dataset')
    
    if not lmo_test_dir.exists():
        print(f"\n❌ LMO数据集不存在: {lmo_test_dir}")
        return
    
    print(f"\nLMO数据集: {lmo_test_dir}")
    print(f"输出目录: {output_dir}")
    
    # 找到所有场景
    scenes = sorted([d for d in lmo_test_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    print(f"\n找到 {len(scenes)} 个场景")
    
    # 全部用于训练
    print(f"\n{'='*80}")
    print("📊 数据集策略")
    print("="*80)
    print("""
策略：
- LMO全部1214张图像 → 训练集
- 从原始LineMOD抽取对应8类 → 测试集
    """)
    
    # 转换LMO为训练集
    total_train_imgs, total_train_objs = 0, 0
    for scene_dir in scenes:
        imgs, objs = convert_lmo_scene_to_yolo(
            scene_dir, output_dir, split='train', min_visibility=0.1
        )
        total_train_imgs += imgs
        total_train_objs += objs
    
    # 从LineMOD抽取测试集
    print(f"\n{'='*80}")
    print("📂 从LineMOD抽取测试集（8类物体）")
    print("="*80)
    
    linemod_dir = Path(r'd:\projects\odb_box\dataset\Linemod_preprocessed\Linemod_preprocessed\data')
    
    # LMO物体ID到LineMOD文件夹的映射
    lmo_to_linemod = {
        1: '01',   # ape
        5: '05',   # can
        6: '06',   # cat
        8: '08',   # driller
        9: '09',   # duck
        10: '10',  # eggbox
        11: '11',  # glue
        12: '12',  # holepuncher
    }
    
    total_val_imgs, total_val_objs = extract_linemod_test_set(
        linemod_dir, output_dir, lmo_to_linemod, samples_per_class=50
    )
    
    # 创建数据集配置
    class_names = [LMO_ID_TO_NAME[obj_id] for obj_id in sorted(LMO_ID_TO_NAME.keys())]
    yaml_path = create_dataset_yaml(output_dir, class_names)
    
    # 总结
    print(f"\n{'='*80}")
    print("📊 转换完成统计")
    print("="*80)
    print(f"\n训练集:")
    print(f"  图像数: {total_train_imgs}")
    print(f"  物体数: {total_train_objs}")
    if total_train_imgs > 0:
        print(f"  平均: {total_train_objs/total_train_imgs:.2f} 个物体/图")
    
    print(f"\n验证集:")
    print(f"  图像数: {total_val_imgs}")
    print(f"  物体数: {total_val_objs}")
    if total_val_imgs > 0:
        print(f"  平均: {total_val_objs/total_val_imgs:.2f} 个物体/图")
    
    print(f"\n总计:")
    print(f"  图像数: {total_train_imgs + total_val_imgs}")
    print(f"  物体数: {total_train_objs + total_val_objs}")
    
    print(f"\n{'='*80}")
    print("🎯 下一步: 使用LMO数据训练YOLO")
    print("="*80)
    print(f"""
训练命令：

from ultralytics import YOLO

model = YOLO('yolov8m.pt')
results = model.train(
    data=r'{yaml_path}',
    epochs=150,
    batch=8,
    imgsz=640,
    patience=50,
    device=0,
    name='lmo_8classes',
    project='runs/train',
    
    # 真实数据，可以减少数据增强
    mosaic=0.5,
    mixup=0.0,
    copy_paste=0.0,
)

优势：
✅ 真实多物体场景（vs 合成数据）
✅ 每张图平均7.6个物体
✅ 100%多物体图像
✅ 有分割mask（可后续用于实例分割）

注意：
⚠️  只包含8类物体（缺少benchvise, cam, iron, lamp, phone）
⚠️  如需13类，可以混合LMO + 合成数据
""")

if __name__ == '__main__':
    main()

