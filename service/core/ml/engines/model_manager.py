from typing import Any

from service.core.ml.models import SimilarityModelFactory
from service.core.ml.models.base import SimilarityModel
from service.log import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Manages similarity model lifecycle and caching.

    - Create models on demand
    - Cache model instances
    - Handle model configuration
    - Manage model cleanup (future)
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._models: dict[str, SimilarityModel] = {}
            self._initialized = True

    def get_model(self, config_name: str, device: str | None = None, **model_kwargs: Any) -> SimilarityModel:
        """
        Get or create a similarity model for the given configuration.

        Args:
            config_name: Model configuration name
            device: Device for computation
            **model_kwargs: Additional model arguments

        Returns:
            SimilarityModel instance
        """
        # Create cache key from config and device
        cache_key = f"{config_name}:{device or 'auto'}"

        if cache_key not in self._models:
            logger.info(f"Creating new model for config: {config_name}")
            self._models[cache_key] = SimilarityModelFactory.create_model(config_name, device=device, **model_kwargs)
            logger.info(f"Model created: {self._models[cache_key].model_name}")

        return self._models[cache_key]

    def has_model(self, config_name: str, device: str | None = None) -> bool:
        """Check if model exists in cache."""
        cache_key = f"{config_name}:{device or 'auto'}"
        return cache_key in self._models

    def get_cached_models(self) -> dict[str, SimilarityModel]:
        """Get all cached models."""
        return self._models.copy()

    def clear_cache(self) -> None:
        """Clear all cached models."""
        logger.info(f"Clearing {len(self._models)} cached models")
        self._models.clear()

    def remove_model(self, config_name: str, device: str | None = None) -> bool:
        """Remove specific model from cache."""
        cache_key = f"{config_name}:{device or 'auto'}"
        if cache_key in self._models:
            logger.info(f"Removing cached model: {cache_key}")
            del self._models[cache_key]
            return True
        return False


# Global singleton instance access
def get_model_manager() -> ModelManager:
    """Get the global ModelManager singleton instance."""
    return ModelManager()
