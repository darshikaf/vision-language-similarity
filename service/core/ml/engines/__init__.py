"""ML Engines module - Core evaluator and inference logic."""

from .base_evaluator import AbstractEvaluator
from .model_manager import ModelManager, get_model_manager
from .openclip_evaluator import OpenCLIPEvaluator

__all__ = [
    "AbstractEvaluator",
    "OpenCLIPEvaluator",
    "ModelManager",
    "get_model_manager",
]
