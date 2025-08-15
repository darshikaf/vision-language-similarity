from typing import Any

from service.core.config import model_registry
from service.core.ml.models.similarity_models.base import SimilarityModel
from service.core.ml.models.similarity_models.openclip_model import OpenCLIPSimilarityModel


class SimilarityModelFactory:
    """Factory for creating similarity model instances"""

    # Model types registry
    _MODEL_REGISTRY: dict[str, type] = {
        "openclip": OpenCLIPSimilarityModel,
    }

    @classmethod
    def create_model(cls, config_name_or_dict: str | dict[str, Any], **kwargs) -> SimilarityModel:
        """
        Create a similarity model from configuration.

        Args:
            config_name_or_dict: Predefined config type or
                a model configuration
            **kwargs: Additional arguments to override config values

        Returns:
            Configured similarity model instance

        Raises:
            ValueError: If config type is unknown or model type is not registered
        """
        if isinstance(config_name_or_dict, str):
            config_name = config_name_or_dict
            try:
                spec = model_registry.get_model_spec(config_name)
            except KeyError:
                _ = list(model_registry.list_available_models().keys())
                raise

            # CLIPModelSpec object conversion to dict
            config = {
                "type": "openclip",
                "model_name": spec.model_name,
                "pretrained": spec.pretrained,
                "model_config": config_name,
            }
        else:
            config = config_name_or_dict.copy()

        # Override config with any provided kwargs
        config.update(kwargs)

        # Extract model type
        model_type = config.pop("type", "openclip")

        if model_type not in cls._MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._MODEL_REGISTRY.keys())}")

        model_class = cls._MODEL_REGISTRY[model_type]
        return model_class(**config)

    @classmethod
    def create_fast_model(cls, **kwargs) -> SimilarityModel:
        """Create a model optimized for speed"""
        return cls.create_model("fast", **kwargs)

    @classmethod
    def create_accurate_model(cls, **kwargs) -> SimilarityModel:
        """Create a model optimized for accuracy"""
        return cls.create_model("accurate", **kwargs)
