"""ML module - Machine learning related capabilities."""

from .engines.evaluator import MinimalOpenCLIPEvaluator
from .models.similarity_models import SimilarityModelFactory
from .utils.image_loader import ImageLoader
from .utils.image_processor import ImageProcessor
from .utils.metrics_recorder import MetricsRecorder
from .utils.result_builder import ResultBuilder
# DeviceManager imported from parent: service.core.device_manager

__all__ = [
    "MinimalOpenCLIPEvaluator",
    "SimilarityModelFactory",
    "ImageLoader",
    "ImageProcessor",
    "MetricsRecorder", 
    "ResultBuilder",
]