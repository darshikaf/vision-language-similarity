"""ML module - Machine learning related capabilities."""

from .engines.openclip_evaluator import MinimalOpenCLIPEvaluator
from .models import SimilarityModelFactory
from .utils.image_loader import ImageLoader
from .utils.image_processor import ImageProcessor
from .utils.metrics_recorder import MetricsRecorder
from .utils.result_builder import ResultBuilder

__all__ = [
    "MinimalOpenCLIPEvaluator",
    "SimilarityModelFactory",
    "ImageLoader",
    "ImageProcessor",
    "MetricsRecorder",
    "ResultBuilder",
]
