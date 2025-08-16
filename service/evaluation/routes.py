from fastapi import APIRouter, Depends, HTTPException, status

from service.core.exception_handler import common_exception_handler
from service.evaluation.handler import EvaluationHandler, get_handler
from service.evaluation.schema import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
)
from service.log import get_logger

logger = get_logger(__name__)

EVALUATION_PREFIX = "/v1/evaluation"

router = APIRouter(prefix=EVALUATION_PREFIX, tags=["evaluation"])


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
