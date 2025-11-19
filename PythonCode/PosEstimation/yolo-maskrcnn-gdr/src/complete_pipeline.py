#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整6D姿态估计Pipeline
整合YOLO检测、Mask R-CNN分割、GDR-Net姿态估计和3D可视化
"""

import cv2
import numpy as np
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

# 导入各个模块
from .yolo_detector import YOLODetector
from .maskrcnn_segmentor import MaskRCNNSegmentor
from .gdrnet_pose_estimator import GDRNetPoseEstimator
from .enhanced_3d_visualizer import Enhanced3DVisualizer

class Complete6DPosePipeline:
    """完整的6D姿态估计Pipeline"""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        初始化完整Pipeline
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典（优先级高于config_path）
        """
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 加载配置
        self.config = self._load_config(config_path, config_dict)
        
        # 性能统计
        self.performance_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'detection_time': 0.0,
            'segmentation_time': 0.0,
            'pose_estimation_time': 0.0,
            'visualization_time': 0.0
        }
        
        print("🚀 初始化完整6D姿态估计Pipeline...")
        print("=" * 60)
        
        # 初始化各个模块
        self._initialize_modules()
        
        print("=" * 60)
        print("✅ Pipeline初始化完成!")
        print(f"📊 支持的物体类别: {list(self.config['pose_estimation']['objects'].keys())}")
        print(f"🎯 相机分辨率: {self.config['camera']['width']}x{self.config['camera']['height']}")
    
    def _load_config(self, config_path: Optional[str], config_dict: Optional[Dict]) -> Dict:
        """加载配置"""
        if config_dict:
            return config_dict
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        
        # 使用默认配置
        return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'detection': {
                'model_name': 'yolov8n.pt',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'device': 'cpu'
            },
            'segmentation': {
                'confidence_threshold': 0.7,
                'device': 'cpu',
                'use_detection_prompts': True
            },
            'pose_estimation': {
                'confidence_threshold': 0.6,
                'device': 'cpu',
                'use_segmentation_mask': True,
                'objects': {
                    'can': {'id': 5, 'diameter': 202.0},
                    'phone': {'id': 15, 'diameter': 213.0},
                    'ape': {'id': 1, 'diameter': 102.0},
                    'lamp': {'id': 14, 'diameter': 284.0}
                }
            },
            'camera': {
                'K': [[572.4114, 0.0, 325.2611],
                      [0.0, 573.57043, 242.04899], 
                      [0.0, 0.0, 1.0]],
                'width': 640,
                'height': 480
            },
            'visualization': {
                'save_results': True,
                'save_format': 'jpg'
            }
        }
    
    def _initialize_modules(self):
        """初始化各个模块"""
        try:
            # 1. 初始化YOLO检测器
            print("📦 初始化YOLO检测器...")
            self.detector = YOLODetector(self.config['detection'])
            
            # 2. 初始化Mask R-CNN分割器
            print("🎯 初始化Mask R-CNN分割器...")
            self.segmentor = MaskRCNNSegmentor(self.config['segmentation'])
            
            # 3. 初始化GDR-Net姿态估计器
            print("📐 初始化GDR-Net姿态估计器...")
            self.pose_estimator = GDRNetPoseEstimator(self.config['pose_estimation'])
            
            # 4. 初始化3D可视化器
            print("🎨 初始化增强3D可视化器...")
            self.visualizer = Enhanced3DVisualizer(self.config['visualization'])
            
            # 5. 设置相机参数
            self.camera_K = np.array(self.config['camera']['K'], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline初始化失败: {e}")
            raise
    
    def process_image(self, image: Union[str, np.ndarray], 
                     save_results: Optional[bool] = None,
                     show_visualization: bool = False) -> Dict:
        """
        处理单张图像的完整流程
        
        Args:
            image: 输入图像路径或numpy数组
            save_results: 是否保存结果（覆盖配置）
            show_visualization: 是否显示可视化结果
            
        Returns:
            完整的处理结果字典
        """
        start_time = time.time()
        
        # 加载图像
        if isinstance(image, str):
            img = cv2.imread(image)
            image_name = Path(image).stem
            if img is None:
                raise ValueError(f"无法加载图像: {image}")
        else:
            img = image.copy()
            image_name = f"image_{int(time.time())}"
        
        print(f"\n🖼️ 处理图像: {image_name} ({img.shape[1]}x{img.shape[0]})")
        print("=" * 50)
        
        # 步骤1: 目标检测
        print("🔍 步骤1: YOLO目标检测...")
        detection_start = time.time()
        detections = self.detector.detect(img)
        detection_time = time.time() - detection_start
        
        print(f"   ✅ 检测完成: 发现 {len(detections)} 个物体 ({detection_time:.3f}s)")
        for i, det in enumerate(detections):
            mapped_class = det.get('mapped_class', 'N/A')
            print(f"      #{i+1}: {det['class_name']} -> {mapped_class} ({det['confidence']:.3f})")
        
        if not detections:
            return self._create_empty_result(image_name, img.shape, start_time)
        
        # 步骤2: 实例分割
        print("\n🎯 步骤2: Mask R-CNN实例分割...")
        segmentation_start = time.time()
        masks = self.segmentor.segment(img, detections)
        segmentation_time = time.time() - segmentation_start
        
        print(f"   ✅ 分割完成: 生成 {len(masks)} 个掩码 ({segmentation_time:.3f}s)")
        
        # 步骤3: 6D姿态估计
        print("\n📐 步骤3: GDR-Net 6D姿态估计...")
        pose_start = time.time()
        poses = self.pose_estimator.estimate_poses(img, detections, masks, self.camera_K)
        pose_time = time.time() - pose_start
        
        print(f"   ✅ 姿态估计完成: {len(poses)} 个物体姿态 ({pose_time:.3f}s)")
        for i, pose in enumerate(poses):
            t = pose['translation']
            print(f"      #{i+1}: {pose['object_name']} - 位置({t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}) 置信度({pose['confidence']:.3f})")
        
        # 步骤4: 可视化
        print("\n🎨 步骤4: 生成增强可视化...")
        visualization_start = time.time()
        
        visualization_result = self.visualizer.create_complete_visualization(
            img, detections, masks, poses, self.camera_K
        )
        
        visualization_time = time.time() - visualization_start
        total_time = time.time() - start_time
        
        print(f"   ✅ 可视化完成 ({visualization_time:.3f}s)")
        
        # 构建完整结果
        result = {
            'image_name': image_name,
            'image_shape': img.shape,
            'success': len(poses) > 0,
            'num_objects': len(detections),
            'num_poses': len(poses),
            
            # 处理结果
            'detections': detections,
            'masks': masks,
            'poses': poses,
            'visualization': visualization_result,
            
            # 技术参数
            'camera_K': self.camera_K.tolist(),
            'pipeline_config': {
                'detection_model': self.config['detection']['model_name'],
                'segmentation_model': 'Mask R-CNN',
                'pose_model': 'GDR-Net (模拟)',
                'visualization': '增强3D可视化'
            },
            
            # 性能统计
            'processing_times': {
                'detection': detection_time,
                'segmentation': segmentation_time,
                'pose_estimation': pose_time,
                'visualization': visualization_time,
                'total': total_time
            },
            
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat()
        }
        
        # 更新性能统计
        self._update_performance_stats(result['processing_times'])
        
        # 显示结果摘要
        self._print_result_summary(result)
        
        # 显示可视化
        if show_visualization and 'annotated_image' in visualization_result:
            cv2.imshow(f"完整6D姿态估计 - {image_name}", visualization_result['annotated_image'])
            print("\n⌨️ 按任意键继续...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result
    
    def process_batch(self, image_paths: List[str], 
                     save_results: bool = True,
                     show_progress: bool = True) -> List[Dict]:
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            save_results: 是否保存结果
            show_progress: 是否显示进度
            
        Returns:
            处理结果列表
        """
        results = []
        total_images = len(image_paths)
        
        print(f"\n🚀 开始批量处理: {total_images} 张图像")
        print("=" * 60)
        
        for i, image_path in enumerate(image_paths):
            if show_progress:
                print(f"\n📊 进度: {i+1}/{total_images} ({(i+1)/total_images*100:.1f}%)")
            
            try:
                result = self.process_image(image_path, save_results=save_results)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"❌ 处理图像失败 {image_path}: {e}")
                results.append({
                    'image_name': Path(image_path).stem,
                    'success': False,
                    'error': str(e)
                })
        
        # 批量统计
        successful = sum(1 for r in results if r.get('success', False))
        total_objects = sum(r.get('num_objects', 0) for r in results)
        total_poses = sum(r.get('num_poses', 0) for r in results)
        
        print(f"\n📊 批量处理完成:")
        print(f"   ✅ 成功处理: {successful}/{total_images} 张图像")
        print(f"   🎯 总检测物体: {total_objects} 个")
        print(f"   📐 总姿态估计: {total_poses} 个")
        print(f"   ⏱️ 平均处理时间: {self.performance_stats['average_time']:.3f}s/张")
        
        return results
    
    def _create_empty_result(self, image_name: str, image_shape: Tuple, start_time: float) -> Dict:
        """创建空结果"""
        total_time = time.time() - start_time
        
        return {
            'image_name': image_name,
            'image_shape': image_shape,
            'success': False,
            'num_objects': 0,
            'num_poses': 0,
            'detections': [],
            'masks': [],
            'poses': [],
            'message': '未检测到物体',
            'processing_times': {'total': total_time},
            'timestamp': time.time()
        }
    
    def _update_performance_stats(self, processing_times: Dict):
        """更新性能统计"""
        self.performance_stats['total_processed'] += 1
        self.performance_stats['total_time'] += processing_times['total']
        self.performance_stats['detection_time'] += processing_times['detection']
        self.performance_stats['segmentation_time'] += processing_times['segmentation']
        self.performance_stats['pose_estimation_time'] += processing_times['pose_estimation']
        self.performance_stats['visualization_time'] += processing_times['visualization']
        
        # 计算平均时间
        count = self.performance_stats['total_processed']
        self.performance_stats['average_time'] = self.performance_stats['total_time'] / count
    
    def _print_result_summary(self, result: Dict):
        """打印结果摘要"""
        print(f"\n📋 处理结果摘要:")
        print(f"   🎯 检测物体: {result['num_objects']} 个")
        print(f"   📐 姿态估计: {result['num_poses']} 个")
        print(f"   ⏱️ 总耗时: {result['processing_times']['total']:.3f}s")
        
        # 详细时间分解
        times = result['processing_times']
        print(f"   📊 时间分解:")
        print(f"      - 检测: {times['detection']:.3f}s ({times['detection']/times['total']*100:.1f}%)")
        print(f"      - 分割: {times['segmentation']:.3f}s ({times['segmentation']/times['total']*100:.1f}%)")
        print(f"      - 姿态: {times['pose_estimation']:.3f}s ({times['pose_estimation']/times['total']*100:.1f}%)")
        print(f"      - 可视化: {times['visualization']:.3f}s ({times['visualization']/times['total']*100:.1f}%)")
        
        if result['success']:
            print(f"   ✅ 处理成功!")
        else:
            print(f"   ❌ 处理失败: {result.get('message', '未知错误')}")
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        stats = self.performance_stats.copy()
        
        if stats['total_processed'] > 0:
            count = stats['total_processed']
            stats['average_detection_time'] = stats['detection_time'] / count
            stats['average_segmentation_time'] = stats['segmentation_time'] / count
            stats['average_pose_estimation_time'] = stats['pose_estimation_time'] / count
            stats['average_visualization_time'] = stats['visualization_time'] / count
        
        return stats
    
    def print_performance_report(self):
        """打印性能报告"""
        stats = self.get_performance_report()
        
        print("\n📊 Pipeline性能报告")
        print("=" * 40)
        print(f"总处理图像: {stats['total_processed']} 张")
        print(f"总处理时间: {stats['total_time']:.2f}s")
        print(f"平均处理时间: {stats['average_time']:.3f}s/张")
        print(f"处理速度: {1/stats['average_time']:.1f} FPS")
        
        if stats['total_processed'] > 0:
            print(f"\n各模块平均耗时:")
            print(f"  🔍 检测: {stats['average_detection_time']:.3f}s")
            print(f"  🎯 分割: {stats['average_segmentation_time']:.3f}s")
            print(f"  📐 姿态: {stats['average_pose_estimation_time']:.3f}s")
            print(f"  🎨 可视化: {stats['average_visualization_time']:.3f}s")
    
    def update_config(self, new_config: Dict):
        """更新配置"""
        self.config.update(new_config)
        
        # 更新各模块配置
        if 'detection' in new_config:
            self.detector.update_config(new_config['detection'])
        
        if 'camera' in new_config:
            self.camera_K = np.array(new_config['camera']['K'], dtype=np.float32)
        
        self.logger.info("🔄 Pipeline配置已更新")
    
    def get_supported_objects(self) -> List[str]:
        """获取支持的物体类别"""
        return list(self.config['pose_estimation']['objects'].keys())
    
    def get_pipeline_info(self) -> Dict:
        """获取Pipeline信息"""
        return {
            'version': '1.0.0',
            'modules': {
                'detector': 'YOLO',
                'segmentor': 'Mask R-CNN',
                'pose_estimator': 'GDR-Net',
                'visualizer': 'Enhanced 3D'
            },
            'supported_objects': self.get_supported_objects(),
            'camera_resolution': f"{self.config['camera']['width']}x{self.config['camera']['height']}",
            'performance_stats': self.performance_stats
        }

def create_demo_pipeline() -> Complete6DPosePipeline:
    """创建演示用的Pipeline"""
    config = {
        'detection': {
            'model_name': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'device': 'cpu',
            'class_mapping': {
                41: 'can',      # cup -> can
                67: 'phone',    # cell phone -> phone
                62: 'lamp',     # tv -> lamp
                64: 'ape',      # mouse -> ape (for demo)
            }
        },
        'segmentation': {
            'confidence_threshold': 0.7,
            'device': 'cpu',
            'use_detection_prompts': True,
            'smooth_mask': True
        },
        'pose_estimation': {
            'confidence_threshold': 0.6,
            'device': 'cpu',
            'use_segmentation_mask': True,
            'objects': {
                'can': {'id': 5, 'diameter': 202.0},
                'phone': {'id': 15, 'diameter': 213.0},
                'ape': {'id': 1, 'diameter': 102.0},
                'lamp': {'id': 14, 'diameter': 284.0}
            }
        },
        'camera': {
            'K': [[572.4114, 0.0, 325.2611],
                  [0.0, 573.57043, 242.04899], 
                  [0.0, 0.0, 1.0]],
            'width': 640,
            'height': 480
        },
        'visualization': {
            'save_results': True,
            'save_format': 'jpg',
            'save_quality': 95
        }
    }
    
    return Complete6DPosePipeline(config_dict=config)

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🧪 测试完整6D姿态估计Pipeline")
    
    # 创建Pipeline
    pipeline = create_demo_pipeline()
    
    # 创建测试图像
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 120
    
    # 添加一些模拟物体
    cv2.rectangle(test_image, (100, 100), (200, 200), (140, 160, 120), -1)
    cv2.circle(test_image, (350, 150), 60, (120, 140, 180), -1)
    cv2.ellipse(test_image, (500, 300), (50, 80), 0, 0, 360, (160, 120, 140), -1)
    
    # 处理图像
    result = pipeline.process_image(test_image, show_visualization=True)
    
    # 打印性能报告
    pipeline.print_performance_report()
    
    # 打印Pipeline信息
    info = pipeline.get_pipeline_info()
    print(f"\n📋 Pipeline信息:")
    print(f"   版本: {info['version']}")
    print(f"   支持物体: {', '.join(info['supported_objects'])}")
    print(f"   相机分辨率: {info['camera_resolution']}")
    
    print("\n🎉 Pipeline测试完成!")
