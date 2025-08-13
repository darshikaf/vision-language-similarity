import asyncio
import logging
import time

from service.core import EvaluationResult, MinimalOpenCLIPEvaluator
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

    def __init__(self):
        self._evaluators: dict[str, MinimalOpenCLIPEvaluator] = {}
        self._model_configs = {
            "fast": {"model_name": "ViT-B-32", "pretrained": "laion2b_s34b_b79k"},
            "accurate": {"model_name": "ViT-L-14", "pretrained": "laion2b_s32b_b82k"},
        }

    def _get_evaluator(self, model_config: str) -> MinimalOpenCLIPEvaluator:
        """Get or create evaluator for given config"""
        if model_config not in self._evaluators:
            if model_config == "fast":
                self._evaluators[model_config] = MinimalOpenCLIPEvaluator.create_fast_evaluator()
            elif model_config == "accurate":
                self._evaluators[model_config] = MinimalOpenCLIPEvaluator.create_accurate_evaluator()
            else:
                raise ValueError(f"Unknown model configuration: {model_config}")

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
                error_type = "evaluation_error"
                # Try to extract specific error type from error message
                if "network" in result.error.lower() or "url" in result.error.lower():
                    error_type = "network_error"
                elif "image" in result.error.lower():
                    error_type = "image_processing_error"
                elif "model" in result.error.lower():
                    error_type = "model_error"

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
                failed_result = EvaluationResult(
                    image_path=eval_req.image_input,
                    text_prompt=eval_req.text_prompt,
                    clip_score=0.0,
                    processing_time_ms=0.0,
                    error=str(result),
                )
                final_results.append(self._evaluation_result_to_response(failed_result, model_config))
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
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            model_loaded = False

        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            available_configs=list(self._model_configs.keys()),
        )


_handler: EvaluationHandler | None = None


def get_handler() -> EvaluationHandler:
    global _handler  # noqa: PLW0603
    if _handler is None:
        _handler = EvaluationHandler()
    return _handler
