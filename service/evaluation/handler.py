import base64
import io
import tempfile
import time
from pathlib import Path

from PIL import Image

from service.core import EvaluationResult, MinimalOpenCLIPEvaluator

from .schema import (
    BatchEvaluationRequest,
    BatchEvaluationResponse,
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
)


class EvaluationHandler:
    """Handler for evaluation requests"""

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

    def _process_image_input(self, image_input: str) -> str:
        """Process image input - handle base64 or return URL/path as-is"""
        if image_input.startswith("data:image/"):
            # Base64 encoded image
            try:
                header, encoded = image_input.split(",", 1)
                image_data = base64.b64decode(encoded)
                image = Image.open(io.BytesIO(image_data))

                # Save to secure temp file 
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_path = temp_file.name
                    image.save(temp_path)
                return temp_path
            except Exception as e:
                raise ValueError(f"Failed to process base64 image: {e}") from e

        return image_input

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

        # Process image input (handle base64 if needed)
        processed_image_input = self._process_image_input(request.image_input)

        # Perform evaluation
        result = evaluator.evaluate_single(processed_image_input, request.text_prompt)

        return self._evaluation_result_to_response(result, model_config)

    async def evaluate_batch(self, request: BatchEvaluationRequest) -> BatchEvaluationResponse:
        """Handle batch evaluation request"""
        start_time = time.time()

        # Group requests by model config
        config_groups = {}
        for i, eval_req in enumerate(request.evaluations):
            model_config = eval_req.model_config_name or "fast"
            if model_config not in config_groups:
                config_groups[model_config] = []
            config_groups[model_config].append((i, eval_req))

        # Process each config group
        all_results = [None] * len(request.evaluations)

        for model_config, eval_requests in config_groups.items():
            evaluator = self._get_evaluator(model_config)

            # Prepare batch inputs
            indices = [idx for idx, _ in eval_requests]
            image_inputs = []
            text_prompts = []

            for _, eval_req in eval_requests:
                processed_image = self._process_image_input(eval_req.image_input)
                image_inputs.append(processed_image)
                text_prompts.append(eval_req.text_prompt)

            # Perform batch evaluation
            results = evaluator.evaluate_batch(
                image_inputs=image_inputs,
                text_prompts=text_prompts,
                batch_size=request.batch_size,
                show_progress=request.show_progress,
            )

            # Map results back to original positions
            for idx, result in zip(indices, results, strict=False):
                all_results[idx] = self._evaluation_result_to_response(result, model_config)

        # Calculate summary statistics
        total_processing_time = (time.time() - start_time) * 1000
        successful_results = [r for r in all_results if r.error is None]
        failed_results = [r for r in all_results if r.error is not None]

        return BatchEvaluationResponse(
            results=all_results,
            total_processed=len(all_results),
            total_successful=len(successful_results),
            total_failed=len(failed_results),
            total_processing_time_ms=total_processing_time,
        )

    async def health_check(self) -> HealthResponse:
        """Health check endpoint"""
        try:
            # Try to load fast evaluator to verify models are accessible
            self._get_evaluator("fast")
            model_loaded = True
        except Exception:
            model_loaded = False

        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            available_configs=list(self._model_configs.keys()),
        )


# Global handler instance - using module-level singleton pattern
_handler: EvaluationHandler | None = None


def get_handler() -> EvaluationHandler:
    """Get global handler instance using singleton pattern"""
    # Using module-level singleton instead of global statement
    global _handler  # noqa: PLW0603
    if _handler is None:
        _handler = EvaluationHandler()
    return _handler
