import asyncio
from contextlib import asynccontextmanager
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
        self._lock = asyncio.Lock()

    async def get_model(self, config_name: str) -> MinimalOpenCLIPEvaluator:
        """
        Get or create a model instance with thread-safe initialization

        Args:
            config_name: Model configuration name

        Returns:
            Initialized evaluator instance
        """
        # Fast path: return cached model if healthy
        if config_name in self._models:
            model = self._models[config_name]
            if await self._is_healthy(model):
                return model
            else:
                logger.warning(f"Model {config_name} failed health check, reloading...")
                await self._cleanup_model(config_name)

        # Slow path: load model with lock
        async with self._lock:
            # Double-check pattern
            if config_name in self._models:
                return self._models[config_name]

            # Create new model
            model = await self._create_model(config_name)
            self._models[config_name] = model

            logger.info(f"Loaded and cached model: {config_name}")
            return model

    async def _create_model(self, config_name: str) -> MinimalOpenCLIPEvaluator:
        """Create a new model instance - direct creation logic"""
        logger.info(f"Creating model: {config_name}")

        # Verify config exists
        try:
            _ = model_registry.get_model_spec(config_name)
        except KeyError as e:
            raise ValueError(f"Failed to load model configuration: {config_name}") from e

        # Create evaluator using factory methods for known configs, direct constructor for custom
        model_kwargs = {
            "device": None,  # Auto-detect
            "cache_dir": settings.model.cache_dir,
            "max_workers": settings.model.max_workers,
        }

        if config_name == "fast":
            return MinimalOpenCLIPEvaluator.create_fast_evaluator(**model_kwargs)
        elif config_name == "accurate":
            return MinimalOpenCLIPEvaluator.create_accurate_evaluator(**model_kwargs)
        else:
            # For custom configs, use direct constructor
            return MinimalOpenCLIPEvaluator(model_config_name=config_name, **model_kwargs)

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

    async def preload_models(self, config_names: list[str]) -> None:
        """Preload models for faster first access"""
        tasks = [self.get_model(config_name) for config_name in config_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any failures
        for config_name, result in zip(config_names, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Failed to preload model {config_name}: {result}")

    @asynccontextmanager
    async def model_context(self, config_name: str):
        """Context manager for temporary model usage"""
        model = await self.get_model(config_name)
        try:
            yield model
        finally:
            # Could implement reference counting here if needed
            pass

    def get_status(self) -> dict[str, Any]:
        """Get overall model manager status"""
        return {
            "cached_models": list(self._models.keys()),
            "loaded_models": list(self._models.keys()),
            "available_configs": list(model_registry.list_available_models().keys()),
        }


# Global model manager instance
model_manager = ModelManager()
