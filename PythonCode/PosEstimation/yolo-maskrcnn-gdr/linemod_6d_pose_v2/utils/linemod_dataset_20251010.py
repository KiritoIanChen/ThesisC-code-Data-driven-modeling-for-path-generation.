#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LineMOD Dataset Loader
LineMOD数据集加载器

基于数据集分析，实现标准的数据加载接口

Date: 2025-10-10
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging


class LineMODDataset:
    """
    LineMOD数据集加载器
    
    数据集结构：
    - data/01/, data/02/, ... : 各物体数据
    - models/ : 3D模型文件(.ply)
    - segnet_results/ : 预计算分割结果
    
    注意：
    - 物体ID不连续（1,2,4,5,6,8,9,10,11,12,13,14,15）
    - 没有ID 3 和 7
    """
    
    # LineMOD物体ID到名称的映射
    OBJ_ID_TO_NAME = {
        1: 'ape',
        2: 'benchvise',
        4: 'cam',
        5: 'can',
        6: 'cat',
        8: 'driller',
        9: 'duck',
        10: 'eggbox',
        11: 'glue',
        12: 'holepuncher',
        13: 'iron',
        14: 'lamp',
        15: 'phone'
    }
    
    # LineMOD物体名称到ID的映射
    OBJ_NAME_TO_ID = {v: k for k, v in OBJ_ID_TO_NAME.items()}
    
    # LineMOD物体ID到文件夹编号的映射
    OBJ_ID_TO_FOLDER = {
        1: '01',
        2: '02',
        4: '04',
        5: '05',
        6: '06',
        8: '08',
        9: '09',
        10: '10',
        11: '11',
        12: '12',
        13: '13',
        14: '14',
        15: '15'
    }
    
    def __init__(self, dataset_root: str, object_id: int = None, 
                 split: str = 'train', load_depth: bool = False):
        """
        初始化数据集加载器
        
        Args:
            dataset_root: 数据集根目录
            object_id: 物体ID（1-15，跳过3和7），None则加载所有
            split: 'train' or 'test'
            load_depth: 是否加载深度图
        """
        self.logger = logging.getLogger(__name__)
        self.dataset_root = Path(dataset_root)
        self.object_id = object_id
        self.split = split
        self.load_depth = load_depth
        
        # 验证数据集路径
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_root}")
        
        # 验证关键子目录
        if not (self.dataset_root / "data").exists():
            raise FileNotFoundError(f"数据集data目录不存在: {self.dataset_root / 'data'}")
        
        # 加载数据集
        self.data_items = []
        self._load_dataset()
        
        self.logger.info(f"✅ LineMOD数据集加载完成")
        self.logger.info(f"   - 物体: {self.get_object_names()}")
        self.logger.info(f"   - 划分: {split}")
        self.logger.info(f"   - 样本数: {len(self.data_items)}")
    
    def _load_dataset(self):
        """加载数据集索引"""
        if self.object_id is not None:
            # 加载单个物体
            self._load_object_data(self.object_id)
        else:
            # 加载所有物体
            for obj_id in self.OBJ_ID_TO_NAME.keys():
                self._load_object_data(obj_id)
    
    def _load_object_data(self, object_id: int):
        """加载单个物体的数据"""
        folder_id = self.OBJ_ID_TO_FOLDER[object_id]
        object_name = self.OBJ_ID_TO_NAME[object_id]
        object_dir = self.dataset_root / "data" / folder_id
        
        if not object_dir.exists():
            self.logger.warning(f"⚠️ 物体目录不存在: {object_dir}")
            return
        
        # 加载split文件
        split_file = object_dir / f"{self.split}.txt"
        if not split_file.exists():
            self.logger.warning(f"⚠️ Split文件不存在: {split_file}")
            return
        
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]
        
        # 加载ground truth
        gt_file = object_dir / "gt.yml"
        info_file = object_dir / "info.yml"
        
        with open(gt_file, 'r') as f:
            gt_data = yaml.safe_load(f)
        
        with open(info_file, 'r') as f:
            info_data = yaml.safe_load(f)
        
        # 创建数据项
        for img_id in image_ids:
            frame_id = int(img_id)
            
            # 检查GT是否存在
            if frame_id not in gt_data:
                continue
            
            item = {
                'object_id': object_id,
                'object_name': object_name,
                'folder_id': folder_id,
                'frame_id': frame_id,
                'image_id': img_id,
                'rgb_path': object_dir / "rgb" / f"{img_id}.png",
                'depth_path': object_dir / "depth" / f"{img_id}.png" if self.load_depth else None,
                'mask_path': object_dir / "mask" / f"{img_id}.png",
                'gt': gt_data[frame_id][0],  # Ground truth姿态
                'camera_K': np.array(info_data[frame_id]['cam_K']).reshape(3, 3)
            }
            
            self.data_items.append(item)
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.data_items)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取数据项
        
        Returns:
            {
                'rgb': np.ndarray (H, W, 3),
                'depth': np.ndarray (H, W) or None,
                'mask': np.ndarray (H, W) binary,
                'pose': {
                    'R': np.ndarray (3, 3),
                    't': np.ndarray (3,) in mm
                },
                'bbox': List[int] [x, y, w, h],
                'object_id': int,
                'object_name': str,
                'camera_K': np.ndarray (3, 3),
                'frame_id': int
            }
        """
        item = self.data_items[idx]
        
        # 加载RGB图像
        rgb = cv2.imread(str(item['rgb_path']))
        if rgb is None:
            raise ValueError(f"无法加载RGB图像: {item['rgb_path']}")
        
        # 加载深度图像（可选）
        depth = None
        if self.load_depth and item['depth_path']:
            depth = cv2.imread(str(item['depth_path']), cv2.IMREAD_UNCHANGED)
        
        # 加载掩码
        mask = cv2.imread(str(item['mask_path']), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法加载掩码: {item['mask_path']}")
        
        # 解析ground truth
        gt = item['gt']
        rotation_matrix = np.array(gt['cam_R_m2c']).reshape(3, 3)
        translation = np.array(gt['cam_t_m2c'])
        bbox = gt['obj_bb']
        
        return {
            'rgb': rgb,
            'depth': depth,
            'mask': mask,
            'pose': {
                'R': rotation_matrix,
                't': translation
            },
            'bbox': bbox,
            'object_id': item['object_id'],
            'object_name': item['object_name'],
            'camera_K': item['camera_K'],
            'frame_id': item['frame_id']
        }
    
    def get_object_names(self) -> List[str]:
        """获取当前数据集包含的物体名称"""
        if self.object_id:
            return [self.OBJ_ID_TO_NAME[self.object_id]]
        else:
            return list(self.OBJ_ID_TO_NAME.values())
    
    def get_object_count(self, object_name: str) -> int:
        """获取某个物体的样本数"""
        return sum(1 for item in self.data_items 
                  if item['object_name'] == object_name)
    
    @staticmethod
    def load_3d_model(dataset_root: str, object_id: int) -> np.ndarray:
        """
        加载物体的3D模型点
        
        Args:
            dataset_root: 数据集根目录
            object_id: 物体ID
            
        Returns:
            模型点 (N, 3)
        """
        dataset_root = Path(dataset_root) / "Linemod_preprocessed"
        model_path = dataset_root / "models" / f"obj_{object_id:02d}.ply"
        
        if not model_path.exists():
            raise FileNotFoundError(f"3D模型文件不存在: {model_path}")
        
        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(str(model_path))
            vertices = np.asarray(mesh.vertices)
            return vertices
        except ImportError:
            try:
                import trimesh
                mesh = trimesh.load(str(model_path))
                return np.array(mesh.vertices)
            except ImportError:
                raise ImportError("需要安装 open3d 或 trimesh: pip install open3d")
    
    @staticmethod
    def load_models_info(dataset_root: str) -> Dict:
        """
        加载所有模型的信息
        
        Returns:
            {object_id: {diameter, size_x, size_y, size_z, ...}}
        """
        dataset_root = Path(dataset_root) / "Linemod_preprocessed"
        info_path = dataset_root / "models" / "models_info.yml"
        
        with open(info_path, 'r') as f:
            models_info = yaml.safe_load(f)
        
        return models_info


if __name__ == "__main__":
    # 测试数据集加载器
    print("🧪 测试LineMOD数据集加载器\n")
    
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 数据集路径
    dataset_root = "../../../dataset/Linemod_preprocessed"
    
    # 测试1: 加载单个物体
    print("="*60)
    print("测试1: 加载单个物体 (ape)")
    print("="*60)
    
    try:
        dataset = LineMODDataset(
            dataset_root=dataset_root,
            object_id=1,  # ape
            split='train',
            load_depth=False
        )
        
        print(f"\n数据集大小: {len(dataset)}")
        
        # 获取第一个样本
        sample = dataset[0]
        
        print(f"\n样本信息:")
        print(f"  物体: {sample['object_name']} (ID: {sample['object_id']})")
        print(f"  RGB尺寸: {sample['rgb'].shape}")
        print(f"  掩码尺寸: {sample['mask'].shape}")
        print(f"  边界框: {sample['bbox']}")
        print(f"  旋转矩阵:\n{sample['pose']['R']}")
        print(f"  平移向量: {sample['pose']['t']} mm")
        print(f"  相机内参:\n{sample['camera_K']}")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("请确保数据集路径正确")
    
    # 测试2: 加载模型信息
    print("\n" + "="*60)
    print("测试2: 加载模型信息")
    print("="*60)
    
    try:
        models_info = LineMODDataset.load_models_info(dataset_root)
        
        print(f"\n物体信息:")
        for obj_id, info in list(models_info.items())[:3]:
            obj_name = LineMODDataset.OBJ_ID_TO_NAME.get(obj_id, 'unknown')
            print(f"\n  {obj_id}: {obj_name}")
            print(f"    直径: {info['diameter']:.1f} mm")
            print(f"    尺寸: {info['size_x']:.1f} × {info['size_y']:.1f} × {info['size_z']:.1f} mm")
    
    except FileNotFoundError as e:
        print(f"❌ {e}")
    
    print("\n🎉 数据集加载器测试完成")

