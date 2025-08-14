from functools import wraps
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from service.core.exceptions import ServiceError
from service.evaluation.handler import EvaluationHandler, get_handler
from service.evaluation.schema import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

EVALUATION_PREFIX = "/v1/evaluation"

router = APIRouter(prefix=EVALUATION_PREFIX, tags=["evaluation"])


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


@router.post(
    "/single",
    response_model=EvaluationResponse,
    response_model_exclude_none=True,
    summary="Evaluate single image-text pair",
)
@common_exception_handler
async def evaluate_single(
    request: EvaluationRequest, handler: EvaluationHandler = Depends(get_handler)
) -> EvaluationResponse:
    return await handler.evaluate_single(request)


@router.post(
    "/batch",
    response_model=BatchEvaluationResponse,
    response_model_exclude_none=True,
    summary="Evaluate multiple image-text pairs",
)
@common_exception_handler
async def evaluate_batch(
    request: BatchEvaluationRequest, handler: EvaluationHandler = Depends(get_handler)
) -> BatchEvaluationResponse:
    if not request.evaluations:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Batch request must contain at least one evaluation")

    return await handler.evaluate_batch(request)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
@common_exception_handler
async def health_check(handler: EvaluationHandler = Depends(get_handler)) -> HealthResponse:
    return await handler.health_check()
