import asyncio
import logging
import time

from service.config.model_configs import model_registry
from service.core import EvaluationResult
from service.model_management import model_manager
from service.observability.prometheus_middleware import get_metrics_middleware

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

    DEFAULT_MODEL_CONFIG = "fast"

    def __init__(self):
        # Model management handled by ModelManager
        pass

    def _get_model_config(self, requested_config: str | None) -> str:
        """Get model config name, applying default if needed"""
        return requested_config or self.DEFAULT_MODEL_CONFIG

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

    def _record_evaluation_metrics(self, result: EvaluationResult, model_config: str) -> None:
        """Record metrics for an evaluation result (safe to call)"""
        try:
            metrics = get_metrics_middleware()

            if result.clip_score is not None and result.error is None:
                metrics.record_clip_score(result.clip_score, model_config)

            if result.error is not None:
                error_type = getattr(result, "error_type", "evaluation_error")
                metrics.record_evaluation_error(error_type, model_config)

        except RuntimeError:
            # Metrics middleware not initialized - continue without metrics
            logger.debug("Metrics middleware not available")
        except Exception as e:
            # Log unexpected metrics errors but don't fail the evaluation
            logger.warning(f"Failed to record metrics: {e}")

    def _record_batch_metrics(self, batch_size: int) -> None:
        """Record batch-specific metrics (safe to call)"""
        try:
            metrics = get_metrics_middleware()
            metrics.record_batch_size(batch_size)
        except RuntimeError:
            logger.debug("Metrics middleware not available")
        except Exception as e:
            logger.warning(f"Failed to record batch metrics: {e}")

    async def evaluate_single(self, request: EvaluationRequest) -> EvaluationResponse:
        """Handle single evaluation request with clean separation of concerns"""
        model_config = self._get_model_config(request.model_config_name)

        # Perform evaluation using model manager
        async with model_manager.model_context(model_config) as evaluator:
            result = await evaluator.evaluate_single(request.image_input, request.text_prompt)

        # Record metrics (safe operation, won't fail evaluation)
        self._record_evaluation_metrics(result, model_config)

        return self._evaluation_result_to_response(result, model_config)

    def _group_requests_by_model(self, requests: list[EvaluationRequest]) -> dict[str, list[EvaluationRequest]]:
        """Group evaluation requests by model configuration for batch optimization"""
        groups = {}
        for req in requests:
            model_config = self._get_model_config(req.model_config_name)
            if model_config not in groups:
                groups[model_config] = []
            groups[model_config].append(req)
        return groups

    def _create_error_response(
        self, request: EvaluationRequest, error: Exception, model_config: str
    ) -> EvaluationResponse:
        """Create an error response for a failed evaluation request"""
        failed_result = EvaluationResult(
            image_path=request.image_input,
            text_prompt=request.text_prompt,
            clip_score=0.0,
            processing_time_ms=0.0,
            error=str(error),
        )
        return self._evaluation_result_to_response(failed_result, model_config)

    def _calculate_batch_statistics(self, results: list[EvaluationResponse], processing_time_ms: float) -> dict:
        """Calculate summary statistics for batch evaluation results"""
        successful_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]

        return {
            "results": results,
            "total_processed": len(results),
            "total_successful": len(successful_results),
            "total_failed": len(failed_results),
            "total_processing_time_ms": processing_time_ms,
        }

    async def _evaluate_batch_for_model(
        self, requests: list[EvaluationRequest], model_config: str
    ) -> list[EvaluationResponse]:
        """Evaluate a batch of requests using the same model configuration"""
        responses = []

        # Use single model context for all requests with same config
        async with model_manager.model_context(model_config) as evaluator:
            # Process each request, handling exceptions individually
            for request in requests:
                try:
                    result = await evaluator.evaluate_single(request.image_input, request.text_prompt)
                    response = self._evaluation_result_to_response(result, model_config)
                    self._record_evaluation_metrics(result, model_config)
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Evaluation failed for image {request.image_input}: {e}")
                    error_response = self._create_error_response(request, e, model_config)
                    responses.append(error_response)

        return responses

    async def evaluate_batch(self, request: BatchEvaluationRequest) -> BatchEvaluationResponse:
        """
        Handle batch evaluation request with optimized model loading
        Groups requests by model configuration to minimize model loading overhead
        """
        start_time = time.time()

        # Group requests by model configuration for optimization
        model_groups = self._group_requests_by_model(request.evaluations)

        # Process each model group concurrently
        batch_tasks = [
            self._evaluate_batch_for_model(requests, model_config) for model_config, requests in model_groups.items()
        ]

        # Execute all model groups in parallel
        group_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Flatten results and handle any group-level exceptions
        all_results = []
        for i, group_result in enumerate(group_results):
            if isinstance(group_result, Exception):
                # Handle entire group failure - create error responses for all requests in group
                model_config = list(model_groups.keys())[i]
                failed_requests = model_groups[model_config]
                logger.error(f"Batch evaluation failed for model {model_config}: {group_result}")

                for failed_request in failed_requests:
                    error_response = self._create_error_response(failed_request, group_result, model_config)
                    all_results.append(error_response)
            else:
                all_results.extend(group_result)

        # Calculate summary statistics
        total_processing_time_ms = (time.time() - start_time) * 1000
        statistics = self._calculate_batch_statistics(all_results, total_processing_time_ms)

        # Record batch metrics
        self._record_batch_metrics(len(request.evaluations))

        return BatchEvaluationResponse(**statistics)

    async def health_check(self) -> HealthResponse:
        """Health check for model availability"""
        try:
            # Check default model through model manager
            model_info = await model_manager.get_model_info("fast")
            model_loaded = model_info["health_status"]["healthy"]
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
