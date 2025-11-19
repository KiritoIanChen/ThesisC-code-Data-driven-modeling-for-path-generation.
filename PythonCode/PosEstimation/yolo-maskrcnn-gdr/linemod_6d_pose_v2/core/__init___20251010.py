#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LineMOD 6D Pose Estimation V2 - Core Module
核心模块初始化

Date: 2025-10-10
"""

from .config_manager_20251010 import ConfigManager
from .pipeline_20251010 import BasePipeline
from .data_structures_20251010 import Detection, Mask, Pose6D, PipelineResult

__version__ = "2.0.0"

__all__ = [
    'ConfigManager',
    'BasePipeline',
    'Detection',
    'Mask',
    'Pose6D',
    'PipelineResult',
]

