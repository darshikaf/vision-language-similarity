from typing import Any

from service.core.similarity_models.base import SimilarityModel
from service.core.similarity_models.openclip_model import OpenCLIPSimilarityModel


class SimilarityModelFactory:
    """Factory for creating similarity model instances"""

    # Model types registry
    _MODEL_REGISTRY: dict[str, type] = {
        "openclip": OpenCLIPSimilarityModel,
    }

    # Predefined model configurations
    _MODEL_CONFIGS: dict[str, dict[str, Any]] = {
        "fast": {
            "type": "openclip",
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "model_config": "fast",
        },
        "accurate": {
            "type": "openclip",
            "model_name": "ViT-L-14",
            "pretrained": "laion2b_s32b_b82k",
            "model_config": "accurate",
        },
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
            if config_name not in cls._MODEL_CONFIGS:
                raise ValueError(f"Unknown model config: {config_name}. Available: {list(cls._MODEL_CONFIGS.keys())}")
            config = cls._MODEL_CONFIGS[config_name].copy()
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

    @classmethod
    def register_model_type(cls, type_name: str, model_class: type):
        """
        Register a new model type for future extensibility.

        Args:
            type_name: Name to identify the model type (e.g., "blip", "align")
            model_class: Class implementing SimilarityModel interface
        """
        if not issubclass(model_class, SimilarityModel):
            raise ValueError("Model class must inherit from SimilarityModel")
        cls._MODEL_REGISTRY[type_name] = model_class

    @classmethod
    def add_model_config(cls, config_name: str, config: dict[str, Any]):
        """
        Add a new predefined model configuration.

        Args:
            config_name: Name for the configuration
            config: Dictionary with model parameters
        """
        cls._MODEL_CONFIGS[config_name] = config

    @classmethod
    def get_available_configs(cls) -> dict[str, dict[str, Any]]:
        """Get all available model configurations"""
        return cls._MODEL_CONFIGS.copy()

    @classmethod
    def get_available_model_types(cls) -> list[str]:
        """Get all registered model types"""
        return list(cls._MODEL_REGISTRY.keys())
