from abc import ABC, abstractmethod

from PIL import Image

from service.core.device_manager import DeviceManager


class SimilarityModel(ABC):
    """
    Abstract base class for vision-language similarity models.

    This allows for pluggable similarity computation backends while maintaining
    a consistent API.
    """

    def __init__(self, model_name: str, device: str | None = None, model_config: str = "fast"):
        self.model_name = model_name
        self.device = DeviceManager.get_optimal_device(device)
        self.model_config = model_config
        self._model_loaded = False

    @abstractmethod
    async def compute_similarity(self, image: Image.Image, text_prompt: str) -> tuple[float, float]:
        """
        Compute similarity score between image and text.

        Args:
            image: PIL Image in RGB format
            text_prompt: Text description to compare

        Returns:
            Tuple of (similarity_score, processing_time_ms)
        """
        pass

    @abstractmethod
    async def compute_batch_similarity(
        self, images: list[Image.Image], text_prompts: list[str]
    ) -> tuple[list[float], float]:
        """
        Compute similarity scores for batch of image-text pairs.

        Args:
            images: List of PIL Images in RGB format
            text_prompts: List of text descriptions to compare

        Returns:
            Tuple of (similarity_scores, total_processing_time_ms)
        """
        pass

    def is_available(self) -> bool:
        """Check if model is loaded and ready for inference"""
        return self._model_loaded

    def get_model_info(self) -> dict:
        """Get model metadata for diagnostics"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "model_config": self.model_config,
            "is_loaded": self._model_loaded,
        }
