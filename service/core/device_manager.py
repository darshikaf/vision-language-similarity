"""
Device management utilities for optimal hardware utilization
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection and optimization"""
    
    @staticmethod
    def get_optimal_device(device: Optional[str] = None) -> torch.device:
        """Auto-detect optimal device if not specified"""
        if device:
            return torch.device(device)
        
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_obj = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders")
        else:
            device_obj = torch.device("cpu")
            logger.info("Using CPU")
        
        return device_obj
    
    @staticmethod
    def get_optimal_precision(device: torch.device) -> str:
        """Get optimal precision for device"""
        return 'fp16' if device.type == 'cuda' else 'fp32'