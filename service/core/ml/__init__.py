"""ML module - Machine learning related capabilities."""

from .engines.evaluator import MinimalOpenCLIPEvaluator
from .models.similarity_models import SimilarityModelFactory
from .preprocessing.image_loader import ImageLoader
# DeviceManager imported from parent: service.core.device_manager

__all__ = [
    "MinimalOpenCLIPEvaluator",
    "SimilarityModelFactory",
    "ImageLoader",
]