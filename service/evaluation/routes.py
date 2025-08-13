from functools import wraps
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from service.evaluation.handler import EvaluationHandler, get_handler
from service.evaluation.schema import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
)

EVALUATION_PREFIX = "/v1/evaluation"

router = APIRouter(prefix=EVALUATION_PREFIX, tags=["evaluation"])


def common_exception_handler(func):
    """Common exception handler decorator"""

    @wraps(func)
    async def inner_function(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
        except ValueError as e:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Invalid input: {e}",
            ) from e
        except FileNotFoundError as e:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Resource not found: {e}") from e
        except HTTPException:
            # Re-raise HTTPExceptions as-is (they're already properly formatted)
            raise
        except Exception as e:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Internal server error: {e}") from e
        return result

    return inner_function


@router.post(
    "/single",
    response_model=EvaluationResponse,
    response_model_exclude_none=True,
    summary="Evaluate single image-text pair",
    description="Evaluate the similarity between a single image and text description using CLIP models",
)
@common_exception_handler
async def evaluate_single(
    request: EvaluationRequest, handler: EvaluationHandler = Depends(get_handler)
) -> EvaluationResponse:
    """Evaluate single image-text pair"""
    return await handler.evaluate_single(request)


@router.post(
    "/batch",
    response_model=BatchEvaluationResponse,
    response_model_exclude_none=True,
    summary="Evaluate multiple image-text pairs",
    description="Efficiently evaluate multiple image-text pairs in batch with progress tracking",
)
@common_exception_handler
async def evaluate_batch(
    request: BatchEvaluationRequest, handler: EvaluationHandler = Depends(get_handler)
) -> BatchEvaluationResponse:
    """Evaluate multiple image-text pairs in batch"""
    if not request.evaluations:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Batch request must contain at least one evaluation")

    return await handler.evaluate_batch(request)


@router.get(
    "/models",
    summary="Get available model configurations",
    description="List available model configurations and their details",
)
async def get_available_models() -> dict[str, Any]:
    """Get available model configurations"""
    return {
        "available_configs": {
            "fast": {
                "model_name": "ViT-B-32",
                "pretrained": "laion2b_s34b_b79k",
                "description": "Faster inference, good performance",
            },
            "accurate": {
                "model_name": "ViT-L-14",
                "pretrained": "laion2b_s32b_b82k",
                "description": "Higher accuracy, slower inference",
            },
        },
        "default_config": "fast",
    }


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health and model availability",
)
@common_exception_handler
async def health_check(handler: EvaluationHandler = Depends(get_handler)) -> HealthResponse:
    """Health check endpoint"""
    return await handler.health_check()
