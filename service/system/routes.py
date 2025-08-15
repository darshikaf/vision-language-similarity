from functools import wraps
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from service.config.model_configs import model_registry
from service.core.exceptions import ServiceError
from service.model_management import model_manager

logger = logging.getLogger(__name__)

SYSTEM_PREFIX = "/v1/system"

router = APIRouter(prefix=SYSTEM_PREFIX, tags=["system"])


def common_exception_handler(func):
    @wraps(func)
    async def inner_function(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
        except ServiceError as e:
            # Map custom service exceptions to HTTP status codes
            raise HTTPException(e.http_status, str(e)) from e
        except ValueError as e:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Invalid input: {e}",
            ) from e
        except FileNotFoundError as e:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Resource not found: {e}") from e
        except HTTPException:
            # Re-raise HTTPExceptions as-is
            raise
        except Exception as e:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Internal server error: {e}") from e
        return result

    return inner_function


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
async def get_model_info(config_name: str) -> dict[str, Any]:
    return await model_manager.get_model_info(config_name)


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
    summary="Get model manager status",
    description="Get detailed status of the model management system (admin endpoint)",
)
async def get_model_manager_status() -> dict[str, Any]:
    """Get model manager status"""
    return model_manager.get_status()
