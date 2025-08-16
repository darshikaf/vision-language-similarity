import logging
from typing import Any

from fastapi import APIRouter

from service.core.config import model_registry
from service.core.exception_handler import common_exception_handler
from service.system.handler import get_model_info, get_system_status

logger = logging.getLogger(__name__)

SYSTEM_PREFIX = "/v1/system"

router = APIRouter(prefix=SYSTEM_PREFIX, tags=["system"])


# Model Management Endpoints
@router.get(
    "/models",
    summary="Get available model configurations",
    description="List available model configurations and their details",
)
async def get_available_models() -> dict[str, Any]:
    available_models = model_registry.list_available_models()
    return {"available_configs": available_models, "default_config": "fast", "total_available": len(available_models)}


@router.get(
    "/models/specs",
    summary="Get detailed model specifications",
    description="Get detailed specifications for all available models including performance metrics",
)
async def get_model_specifications() -> dict[str, Any]:
    return model_registry.list_available_models()


@router.get(
    "/models/{config_name}/info",
    summary="Get runtime information for a specific model",
    description="Get detailed runtime information including health status for a specific model",
)
@common_exception_handler
async def get_model_runtime_info(config_name: str) -> dict[str, Any]:
    return get_model_info(config_name)


@router.get(
    "/models/all",
    summary="Get all model configurations",
    description="Get all model configurations including disabled ones (admin endpoint)",
)
async def get_all_model_configs() -> dict[str, Any]:
    return {"all_models": model_registry.list_all_models(), "available_models": model_registry.list_available_models()}


# Administrative Endpoints
@router.get(
    "/status",
    summary="Get system status",
    description="Get detailed status of the model management system (admin endpoint)",
)
async def get_model_manager_status() -> dict[str, Any]:
    """Get system status"""
    return get_system_status()
