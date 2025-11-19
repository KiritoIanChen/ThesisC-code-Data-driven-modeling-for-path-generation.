#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LineMOD 6D Pose Estimation V2 - Base Pipeline
Pipeline基类

Date: 2025-10-10
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Union, List, Optional
from abc import ABC, abstractmethod

from .config_manager_20251010 import ConfigManager
from .data_structures_20251010 import PipelineResult


class BasePipeline(ABC):
    """Pipeline基类"""
    
    def __init__(self, config: Union[str, Path, ConfigManager, dict]):
        """
        初始化Pipeline
        
        Args:
            config: 配置文件路径、ConfigManager对象或配置字典
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 加载配置
        if isinstance(config, ConfigManager):
            self.config = config
        elif isinstance(config, dict):
            self.config = ConfigManager()
            self.config.config = config
        else:
            self.config = ConfigManager(config)
        
        # 初始化组件
        self.detector = None
        self.segmentor = None
        self.pose_estimator = None
        self.visualizer = None
        
        # 性能统计
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
        }
        
        self.logger.info(f"🚀 初始化 {self.__class__.__name__}")
        
        # 子类实现具体的模块初始化
        self._initialize_modules()
    
    @abstractmethod
    def _initialize_modules(self):
        """初始化各个模块（子类实现）"""
        pass
    
    def process_image(self, image: Union[str, Path, np.ndarray], 
                     save_result: bool = None) -> PipelineResult:
        """
        处理单张图像
        
        Args:
            image: 图像路径或numpy数组
            save_result: 是否保存结果（覆盖配置）
            
        Returns:
            PipelineResult
        """
        start_time = time.time()
        
        # 加载图像
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"无法加载图像: {image_path}")
            image_name = image_path.stem
        else:
            img = image.copy()
            image_name = f"image_{int(time.time())}"
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"🖼️ 处理图像: {image_name} ({img.shape[1]}x{img.shape[0]})")
        
        # 执行处理流程（子类实现）
        result = self._process(img, image_name)
        
        # 记录总时间
        total_time = time.time() - start_time
        result.processing_times['total'] = total_time
        
        # 更新统计
        self._update_stats(total_time)
        
        # 打印摘要
        self.logger.info(f"\n{result.summary()}")
        self.logger.info(f"{'='*60}")
        
        # 保存结果
        if save_result or (save_result is None and self.config.get('visualization.save_results', True)):
            self._save_result(result)
        
        return result
    
    @abstractmethod
    def _process(self, image: np.ndarray, image_name: str) -> PipelineResult:
        """处理流程（子类实现）"""
        pass
    
    def process_batch(self, images: List[Union[str, Path, np.ndarray]], 
                     show_progress: bool = True) -> List[PipelineResult]:
        """
        批量处理图像
        
        Args:
            images: 图像列表
            show_progress: 是否显示进度
            
        Returns:
            结果列表
        """
        results = []
        total = len(images)
        
        self.logger.info(f"\n🚀 开始批量处理: {total} 张图像")
        self.logger.info("=" * 60)
        
        for i, image in enumerate(images):
            if show_progress:
                self.logger.info(f"\n📊 进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
            
            try:
                result = self.process_image(image)
                results.append(result)
            except Exception as e:
                self.logger.error(f"❌ 处理失败: {e}")
                # 创建失败结果
                if isinstance(image, (str, Path)):
                    image_name = Path(image).stem
                else:
                    image_name = f"image_{i}"
                
                failed_result = PipelineResult(
                    image_name=image_name,
                    image_shape=(0, 0, 0),
                    success=False,
                    metadata={'error': str(e)}
                )
                results.append(failed_result)
        
        # 打印批量统计
        self._print_batch_stats(results)
        
        return results
    
    def _save_result(self, result: PipelineResult):
        """保存处理结果"""
        output_dir = Path(self.config.get('visualization.output_dir', './results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存可视化图像
        if result.annotated_image is not None:
            output_format = self.config.get('visualization.output_format', 'jpg')
            output_path = output_dir / f"{result.image_name}_result.{output_format}"
            
            if output_format == 'jpg':
                quality = self.config.get('visualization.output_quality', 95)
                cv2.imwrite(str(output_path), result.annotated_image, 
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(str(output_path), result.annotated_image)
            
            self.logger.info(f"💾 结果已保存: {output_path}")
        
        # 保存JSON数据
        import json
        json_path = output_dir / f"{result.image_name}_result.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        self.logger.debug(f"💾 JSON已保存: {json_path}")
    
    def _update_stats(self, processing_time: float):
        """更新性能统计"""
        self.stats['total_processed'] += 1
        self.stats['total_time'] += processing_time
        self.stats['avg_time'] = self.stats['total_time'] / self.stats['total_processed']
    
    def _print_batch_stats(self, results: List[PipelineResult]):
        """打印批量处理统计"""
        successful = sum(1 for r in results if r.success)
        total_detections = sum(len(r.detections) for r in results)
        total_poses = sum(len(r.poses) for r in results)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("📊 批量处理统计:")
        self.logger.info(f"  总图像数: {len(results)}")
        self.logger.info(f"  成功处理: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        self.logger.info(f"  总检测数: {total_detections}")
        self.logger.info(f"  总姿态数: {total_poses}")
        self.logger.info(f"  平均时间: {self.stats['avg_time']:.3f}s/张")
        self.logger.info(f"  平均速度: {1/self.stats['avg_time']:.1f} FPS")
        self.logger.info(f"{'='*60}")
    
    def get_stats(self) -> dict:
        """获取性能统计"""
        return self.stats.copy()
    
    def reset_stats(self):
        """重置性能统计"""
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
        }
        self.logger.info("📊 性能统计已重置")
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"{self.__class__.__name__}("
                f"detector={self.detector is not None}, "
                f"segmentor={self.segmentor is not None}, "
                f"pose_estimator={self.pose_estimator is not None})")


class DummyPipeline(BasePipeline):
    """用于测试的虚拟Pipeline"""
    
    def _initialize_modules(self):
        """不初始化任何模块"""
        self.logger.info("  ℹ️ Dummy Pipeline - 无模块初始化")
    
    def _process(self, image: np.ndarray, image_name: str) -> PipelineResult:
        """简单返回结果"""
        import time
        time.sleep(0.1)  # 模拟处理时间
        
        return PipelineResult(
            image_name=image_name,
            image_shape=image.shape,
            success=True,
            annotated_image=image.copy()
        )


if __name__ == "__main__":
    # 测试Pipeline基类
    print("🧪 测试Pipeline基类")
    
    import logging
    logging.basicConfig(level=logging.INFO,
                       format='%(levelname)s - %(message)s')
    
    # 创建测试pipeline
    pipeline = DummyPipeline({})
    
    # 创建测试图像
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # 处理单张图像
    result = pipeline.process_image(test_image)
    print(f"\n结果摘要:\n{result.summary()}")
    
    # 批量处理
    test_images = [test_image] * 3
    results = pipeline.process_batch(test_images)
    
    print("\n✅ Pipeline基类测试完成")

