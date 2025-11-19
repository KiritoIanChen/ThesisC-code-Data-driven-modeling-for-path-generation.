#!/usr/bin/env python3
"""
生成合成的多物体数据集
从LineMOD单物体图像中，使用mask提取物体，合成到一张图上
"""

import cv2
import numpy as np
import yaml
import random
from pathlib import Path
from tqdm import tqdm
import shutil

# 配置
LINEMOD_ROOT = Path(r"d:\projects\odb_box\dataset\Linemod_preprocessed\Linemod_preprocessed\data")
OUTPUT_ROOT = Path(r"d:\projects\odb_box\yolo_linemod_training\synthetic_multi_object_v2")

# 生成参数
NUM_TRAIN = 3000  # 训练集图像数量
NUM_VAL = 500     # 验证集图像数量
MIN_OBJECTS = 2   # 每张图最少物体数
MAX_OBJECTS = 4   # 每张图最多物体数

# LineMOD ID映射
LINEMOD_OBJECTS = {
    1: 'ape', 2: 'benchvise', 4: 'cam', 5: 'can', 6: 'cat',
    8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue', 12: 'holepuncher',
    13: 'iron', 14: 'lamp', 15: 'phone'
}

LINEMOD_TO_YOLO_CLASS = {
    1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 8: 5, 9: 6,
    10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12
}

CLASS_NAMES = ['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck',
               'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']


def load_object_samples(obj_id, num_samples=30):
    """加载某个物体的样本（图像+mask）"""
    obj_folder = LINEMOD_ROOT / f"{obj_id:02d}"
    rgb_folder = obj_folder / "rgb"
    mask_folder = obj_folder / "mask"
    
    if not rgb_folder.exists() or not mask_folder.exists():
        return []
    
    # 获取所有可用的图像
    rgb_files = sorted(list(rgb_folder.glob("*.png")))
    
    # 随机选择一些
    selected = random.sample(rgb_files, min(num_samples, len(rgb_files)))
    
    samples = []
    for rgb_path in selected:
        mask_path = mask_folder / rgb_path.name
        if not mask_path.exists():
            continue
        
        rgb = cv2.imread(str(rgb_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if rgb is None or mask is None:
            continue
        
        # 提取物体区域
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            continue
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # 裁剪物体
        obj_img = rgb[y_min:y_max+1, x_min:x_max+1]
        obj_mask = mask[y_min:y_max+1, x_min:x_max+1]
        
        samples.append({
            'image': obj_img,
            'mask': obj_mask,
            'obj_id': obj_id
        })
    
    return samples


def create_synthetic_image(object_samples_dict, img_width=640, img_height=480):
    """创建一张合成的多物体图像"""
    
    # 创建背景（灰色噪声）
    background = np.random.randint(100, 150, (img_height, img_width, 3), dtype=np.uint8)
    
    # 随机选择要放置的物体数量
    num_objects = random.randint(MIN_OBJECTS, MAX_OBJECTS)
    
    # 随机选择物体类别（不重复）
    available_classes = list(object_samples_dict.keys())
    selected_classes = random.sample(available_classes, min(num_objects, len(available_classes)))
    
    labels = []
    
    for obj_id in selected_classes:
        samples = object_samples_dict[obj_id]
        if not samples:
            continue
        
        # 随机选一个样本
        sample = random.choice(samples)
        obj_img = sample['image']
        obj_mask = sample['mask']
        
        obj_h, obj_w = obj_img.shape[:2]
        
        # 随机缩放（0.3 ~ 0.8倍）
        scale = random.uniform(0.3, 0.8)
        new_w = int(obj_w * scale)
        new_h = int(obj_h * scale)
        
        if new_w < 10 or new_h < 10:
            continue
        
        obj_img = cv2.resize(obj_img, (new_w, new_h))
        obj_mask = cv2.resize(obj_mask, (new_w, new_h))
        
        # 随机位置（确保完全在图像内）
        max_x = img_width - new_w
        max_y = img_height - new_h
        
        if max_x <= 0 or max_y <= 0:
            continue
        
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        # 将物体贴到背景上（使用mask）
        mask_3c = cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2BGR) > 0
        background[y:y+new_h, x:x+new_w][mask_3c] = obj_img[mask_3c]
        
        # 计算YOLO格式的标注
        yolo_class = LINEMOD_TO_YOLO_CLASS[obj_id]
        x_center = (x + new_w / 2) / img_width
        y_center = (y + new_h / 2) / img_height
        bbox_w = new_w / img_width
        bbox_h = new_h / img_height
        
        labels.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")
    
    return background, labels


def generate_dataset():
    """生成完整数据集"""
    
    print("="*80)
    print("🎨 生成合成多物体数据集")
    print("="*80)
    
    # 创建输出目录
    for split in ['train', 'val']:
        (OUTPUT_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("\n📦 加载物体样本...")
    object_samples = {}
    for obj_id, obj_name in tqdm(LINEMOD_OBJECTS.items(), desc="加载物体"):
        samples = load_object_samples(obj_id, num_samples=50)
        if samples:
            object_samples[obj_id] = samples
            print(f"  ✓ {obj_name}: {len(samples)}个样本")
    
    if not object_samples:
        print("❌ 没有加载到任何物体样本！")
        return
    
    print(f"\n✓ 总共加载了 {len(object_samples)} 种物体")
    
    # 生成训练集
    print(f"\n🎨 生成训练集 ({NUM_TRAIN}张)...")
    for i in tqdm(range(NUM_TRAIN), desc="训练集"):
        img, labels = create_synthetic_image(object_samples)
        
        if not labels:
            continue
        
        # 保存图像
        img_path = OUTPUT_ROOT / 'images' / 'train' / f"syn_train_{i:05d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # 保存标签
        label_path = OUTPUT_ROOT / 'labels' / 'train' / f"syn_train_{i:05d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
    
    # 生成验证集
    print(f"\n🎨 生成验证集 ({NUM_VAL}张)...")
    for i in tqdm(range(NUM_VAL), desc="验证集"):
        img, labels = create_synthetic_image(object_samples)
        
        if not labels:
            continue
        
        # 保存图像
        img_path = OUTPUT_ROOT / 'images' / 'val' / f"syn_val_{i:05d}.jpg"
        cv2.imwrite(str(img_path), img)
        
        # 保存标签
        label_path = OUTPUT_ROOT / 'labels' / 'val' / f"syn_val_{i:05d}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))
    
    # 生成dataset.yaml
    yaml_content = {
        'path': str(OUTPUT_ROOT.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 13,
        'names': CLASS_NAMES
    }
    
    yaml_path = OUTPUT_ROOT / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\n✓ 数据集生成完成！")
    print(f"  位置: {OUTPUT_ROOT}")
    print(f"  训练集: {len(list((OUTPUT_ROOT / 'images' / 'train').glob('*.jpg')))}张")
    print(f"  验证集: {len(list((OUTPUT_ROOT / 'images' / 'val').glob('*.jpg')))}张")
    print(f"  配置文件: {yaml_path}")
    
    # 显示示例
    print("\n📋 查看标签示例...")
    sample_labels = list((OUTPUT_ROOT / 'labels' / 'train').glob('*.txt'))[:3]
    for label_path in sample_labels:
        print(f"\n  {label_path.name}:")
        with open(label_path) as f:
            lines = f.readlines()
        print(f"    物体数量: {len(lines)}")
        for line in lines:
            cls_id = int(line.split()[0])
            print(f"      - {CLASS_NAMES[cls_id]}")


if __name__ == '__main__':
    generate_dataset()
    
    print("\n" + "="*80)
    print("🚀 下一步：训练新模型")
    print("="*80)
    print("""
from ultralytics import YOLO

model = YOLO('yolov8m.pt')
results = model.train(
    data=r'd:\\projects\\odb_box\\yolo_linemod_training\\synthetic_multi_object_v2\\dataset.yaml',
    epochs=150,
    batch=16,
    imgsz=640,
    patience=50,
    name='linemod_synthetic_multi_object',
    project='runs/train'
)
""")



