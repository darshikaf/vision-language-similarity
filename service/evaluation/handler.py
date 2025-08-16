import asyncio
from collections import defaultdict
import logging
import time

from service.core import EvaluationResult, OpenCLIPEvaluator
from service.core.config import model_registry
from service.core.exceptions import ServiceError
from service.core.observability import get_metrics_middleware

from .schema import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)


class EvaluationHandler:
    """Async handler for evaluation requests"""

    def __init__(self):
        self._evaluators: dict[str, OpenCLIPEvaluator] = {}

    def _get_evaluator(self, model_config: str) -> OpenCLIPEvaluator:
        """Get or create evaluator for given config"""
        if model_config not in self._evaluators:
            try:
                # Validate config exists in registry
                model_registry.get_model_spec(model_config)
                # Create evaluator
                self._evaluators[model_config] = OpenCLIPEvaluator(model_config_name=model_config)
            except ValueError as e:
                available_configs = list(model_registry.list_available_models().keys())
                raise ValueError(f"Unknown model configuration: {model_config}. Available: {available_configs}") from e

        return self._evaluators[model_config]

    def _evaluation_result_to_response(self, result: EvaluationResult, model_config: str) -> EvaluationResponse:
        """Convert EvaluationResult to EvaluationResponse"""
        return EvaluationResponse(
            image_input=result.image_path,
            text_prompt=result.text_prompt,
            clip_score=result.clip_score,
            processing_time_ms=result.processing_time_ms,
            error=result.error,
            model_used=model_config,
        )

    async def evaluate_single(self, request: EvaluationRequest) -> EvaluationResponse:
        """Handle single evaluation request"""
        model_config = request.model_config_name or "fast"
        evaluator = self._get_evaluator(model_config)

        # Perform evaluation
        result = await evaluator.evaluate_single(request.image_input, request.text_prompt)

        try:
            metrics = get_metrics_middleware()

            # Record CLIP score distribution
            if result.clip_score is not None and result.error is None:
                metrics.record_clip_score(result.clip_score, model_config)

            # Record evaluation errors
            if result.error is not None:
                # Extract error type from exception
                error_type = getattr(result, "error_type", "evaluation_error")
                metrics.record_evaluation_error(error_type, model_config)

        except RuntimeError:
            # Metrics middleware not initialized - continue without metrics
            logger.debug("Metrics middleware not available")

        return self._evaluation_result_to_response(result, model_config)

    async def evaluate_batch(self, request: BatchEvaluationRequest) -> BatchEvaluationResponse:
        """
        Handle batch evaluation request
        TODO:
        # Batch optimization per model - group requests by model config
        # Address multiple loading of same model concurrently
        # Use evaluator's native batch processing

        FIXME: using native batch processing still cause a memory leak
        """
        start_time = time.time()

        tasks = [self.evaluate_single(eval_req) for eval_req in request.evaluations]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred during processing
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                eval_req = request.evaluations[i]
                model_config = eval_req.model_config_name or "fast"

                failed_response = EvaluationResponse(
                    image_input=eval_req.image_input,
                    text_prompt=eval_req.text_prompt,
                    clip_score=0.0,
                    processing_time_ms=0.0,
                    error=str(result),
                    model_used=model_config,
                )
                final_results.append(failed_response)
            else:
                final_results.append(result)

        # Calculate summary statistics
        total_processing_time = (time.time() - start_time) * 1000
        successful_results = [r for r in final_results if r.error is None]
        failed_results = [r for r in final_results if r.error is not None]

        # Record batch metrics
        try:
            metrics = get_metrics_middleware()
            metrics.record_batch_size(len(request.evaluations))
        except RuntimeError:
            logger.debug("Metrics middleware not available")

        return BatchEvaluationResponse(
            results=final_results,
            total_processed=len(final_results),
            total_successful=len(successful_results),
            total_failed=len(failed_results),
            total_processing_time_ms=total_processing_time,
        )

    async def health_check(self) -> HealthResponse:
        """Health check for model availability"""
        try:
            evaluator = self._get_evaluator("fast")
            model_loaded = (
                evaluator is not None
                and hasattr(evaluator, "similarity_model")
                and hasattr(evaluator.similarity_model, "model")
            )
        except (ServiceError, ValueError):
            # Re-raise service and validation errors to be handled by route exception handler
            raise
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            model_loaded = False

        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            available_configs=list(model_registry.list_available_models().keys()),
        )


_handler: EvaluationHandler | None = None


def get_handler() -> EvaluationHandler:
    global _handler  # noqa: PLW0603
    if _handler is None:
        _handler = EvaluationHandler()
    return _handler
