from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from service.log import get_logger

logger = get_logger(__name__)


@dataclass
class CLIPModelSpec:
    """CLIP model specification"""

    model_name: str
    pretrained: str
    description: str
    enabled: bool = True


class DynamicModelRegistry:
    """
    Registry that can load model configurations from multiple sources:
    1. Built-in configurations (fallback)
    2. JSON file
    3. Environment variables
    """

    # Built-in default configurations (always available)
    DEFAULT_MODELS = {
        "fast": CLIPModelSpec(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            description="Faster inference, good performance",
        ),
        "accurate": CLIPModelSpec(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            description="Higher accuracy, slower inference",
        ),
    }

    def __init__(self, config_file_path: str = "/app/config/models.json"):
        self.config_file_path = config_file_path
        self._models: dict[str, CLIPModelSpec] = {}
        self._load_configurations()

    def _load_configurations(self):
        """Load configurations from multiple sources in priority order"""
        # Start with defaults
        self._models = self.DEFAULT_MODELS.copy()

        # Override with file-based config
        self._load_from_file()

        # Override with environment variables
        self._load_from_env()

        logger.info(f"Loaded {len(self._models)} model configurations")

    def _load_from_file(self):
        """Load from JSON file"""
        try:
            config_path = Path(self.config_file_path)
            if config_path.exists():
                with open(config_path) as f:
                    file_config = json.load(f)

                for name, spec_dict in file_config.get("models", {}).items():
                    try:
                        spec = CLIPModelSpec(**spec_dict)
                        self._models[name] = spec
                        logger.info(f"Loaded model config from file: {name}")
                    except Exception as e:
                        logger.error(f"Invalid model spec for {name}: {e}")

        except Exception as e:
            logger.warning(f"Could not load config from {self.config_file_path}: {e}")

    def _load_from_env(self):
        """Load from environment variables"""
        # Look for MODEL_CONFIG_<NAME> environment variables
        for env_var in os.environ:
            if env_var.startswith("MODEL_CONFIG_"):
                try:
                    env_config = json.loads(os.environ[env_var])

                    for name, spec_dict in env_config.get("models", {}).items():
                        try:
                            spec = CLIPModelSpec(**spec_dict)
                            self._models[name] = spec
                            logger.info(f"Loaded model config from env {env_var}: {name}")
                        except Exception as e:
                            logger.error(f"Invalid model spec for {name} in {env_var}: {e}")

                except Exception as e:
                    logger.error(f"Invalid JSON or model spec in {env_var}: {e}")

    def get_model_spec(self, config_name: str) -> CLIPModelSpec:
        """Get model specification by name"""
        if config_name not in self._models:
            raise ValueError(f"Unknown model config: {config_name}. Available: {list(self._models.keys())}")

        spec = self._models[config_name]
        if not spec.enabled:
            raise ValueError(f"Model config {config_name} is disabled")

        return spec

    def list_available_models(self) -> dict[str, dict[str, Any]]:
        """List all available and enabled model configurations"""
        return {name: asdict(spec) for name, spec in self._models.items() if spec.enabled}

    def list_all_models(self) -> dict[str, dict[str, Any]]:
        """List all model configurations (including disabled)"""
        return {name: asdict(spec) for name, spec in self._models.items()}


# Global registry instance
model_registry = DynamicModelRegistry()
