import asyncio
import logging
from typing import Any

from service.config.model_configs import model_registry
from service.config.settings import settings
from service.core import MinimalOpenCLIPEvaluator

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Model management class that handles:
    - Model loading and caching
    - Health checking
    - Resource cleanup
    - Thread-safe operations
    """

    def __init__(self):
        self._models: dict[str, MinimalOpenCLIPEvaluator] = {}

    async def _is_healthy(self, model: MinimalOpenCLIPEvaluator) -> bool:
        """Simple health check for model"""
        try:
            # Basic health check - verify evaluator has similarity model
            if not hasattr(model, "similarity_model"):
                return False

            similarity_model = model.similarity_model

            # Check underlying similarity model
            if not hasattr(similarity_model, "model") or not hasattr(similarity_model, "device"):
                return False

            # Check if model is in evaluation mode (for PyTorch models)
            if hasattr(similarity_model.model, "training"):
                if similarity_model.model.training:
                    logger.warning(f"Model {similarity_model.model_name} is in training mode")
                    return False

            return True

        except Exception as e:
            similarity_model = getattr(model, "similarity_model", None)
            model_name = getattr(similarity_model, "model_name", "unknown") if similarity_model else "unknown"
            logger.error(f"Health check failed for model {model_name}: {e}")
            return False

    async def get_model_info(self, config_name: str) -> dict[str, Any]:
        """Get detailed model information"""
        spec = model_registry.get_model_spec(config_name)

        loaded = config_name in self._models
        health_status = await self._get_health_status(config_name) if loaded else {"healthy": False}
        runtime_info = await self._get_runtime_info(config_name) if loaded else {}

        return {
            "config_name": config_name,
            "spec": self._build_spec_info(spec),
            "loaded": loaded,
            "health_status": health_status,
            "runtime_info": runtime_info,
        }

    def _build_spec_info(self, spec) -> dict[str, Any]:
        """Build specification information dictionary"""
        return {
            "model_name": spec.model_name,
            "pretrained": spec.pretrained,
            "description": spec.description,
            "memory_gb": spec.memory_gb,
            "avg_inference_time_ms": spec.avg_inference_time_ms,
            "accuracy_score": spec.accuracy_score,
            "enabled": spec.enabled,
        }

    async def _get_health_status(self, config_name: str) -> dict[str, Any]:
        """Get health status for a loaded model"""
        try:
            model = self._models[config_name]
            is_healthy = await self._is_healthy(model)

            return {
                "healthy": is_healthy,
                "model_name": self._get_model_name(model),
                "model_loaded": self._is_model_loaded(model),
            }
        except Exception as e:
            logger.warning(f"Could not get health status for {config_name}: {e}")
            return {"healthy": False}

    def _get_model_name(self, model: MinimalOpenCLIPEvaluator) -> str:
        """Extract model name from evaluator instance"""
        if not hasattr(model, "similarity_model"):
            return "unknown"
        return getattr(model.similarity_model, "model_name", "unknown")

    def _is_model_loaded(self, model: MinimalOpenCLIPEvaluator) -> bool:
        """Check if the underlying model is loaded"""
        if not hasattr(model, "similarity_model"):
            return False

        similarity_model = model.similarity_model
        return hasattr(similarity_model, "model") and similarity_model.model is not None

    async def _get_runtime_info(self, config_name: str) -> dict[str, Any]:
        """Get runtime information for a loaded model"""
        try:
            model = self._models[config_name]
            if hasattr(model, "get_model_info"):
                return model.get_model_info()
            return {}
        except Exception as e:
            logger.warning(f"Could not get runtime info for {config_name}: {e}")
            return {}

    def get_status(self) -> dict[str, Any]:
        """Get overall model manager status"""
        return {
            "cached_models": list(self._models.keys()),
            "loaded_models": list(self._models.keys()),
            "available_configs": list(model_registry.list_available_models().keys()),
        }


# Global model manager for resource sharing and performance
model_manager = ModelManager()
