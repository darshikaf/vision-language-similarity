import logging
from typing import Any

from service.core.config import model_registry

logger = logging.getLogger(__name__)


def get_model_info(config_name: str) -> dict[str, Any]:
    """
    Get detailed model information for a specific configuration.
    
    Args:
        config_name: Model configuration name
        
    Returns:
        Dictionary containing model specification and status information
    """
    spec = model_registry.get_model_spec(config_name)
    
    return {
        "config_name": config_name,
        "spec": {
            "model_name": spec.model_name,
            "pretrained": spec.pretrained,
            "description": spec.description,
            "memory_gb": spec.memory_gb,
            "avg_inference_time_ms": spec.avg_inference_time_ms,
            "accuracy_score": spec.accuracy_score,
            "enabled": spec.enabled,
        },
        "loaded": False,  # Models are created on-demand, not pre-loaded
        "health_status": {"healthy": False},  # Models are healthy when created
        "runtime_info": {},  # No persistent runtime info since no caching
    }


def get_system_status() -> dict[str, Any]:
    """
    Get overall system status for model management.
    
    Returns:
        Dictionary containing system status information
    """
    available_configs = list(model_registry.list_available_models().keys())
    
    return {
        "cached_models": [],  # No caching in simplified approach
        "loaded_models": [],  # Models created on-demand
        "available_configs": available_configs,
    }