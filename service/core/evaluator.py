import asyncio
import logging
from pathlib import Path
import time

from PIL import Image

from service.core.exceptions import ServiceError, ValidationError
from service.core.image_loader import ImageLoader
from service.core.models import EvaluationResult
from service.core.similarity_models import SimilarityModelFactory
from service.observability.prometheus_middleware import get_metrics_middleware

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MinimalOpenCLIPEvaluator:
    """
    OpenCLIP evaluator with essential features:
    - Single model configuration with caching
    - CLIP standard scoring only
    - Robust error handling with async execution
    - Efficient batch processing with concurrency

    TODO:
    # Add thread pool cleanup
    # Include GPU memory management
    """

    def __init__(
        self, model_config_name: str = "fast", device: str | None = None, max_concurrent_loads: int = 10, **model_kwargs
    ):
        """
        Initialize evaluator with pluggable similarity model.

        Args:
            model_config_name: Model configuration name ("fast", "accurate", or custom config)
            device: Device for computation (auto-detected if None)
            max_concurrent_loads: Max concurrent image loading operations
            **model_kwargs: Additional arguments passed to the similarity model
        """
        self.model_config_name = model_config_name
        self.max_concurrent_loads = max_concurrent_loads

        self.similarity_model = SimilarityModelFactory.create_model(model_config_name, device=device, **model_kwargs)

        logger.info(f"Initialized evaluator with {self.similarity_model.model_name} model ({model_config_name} config)")

    @property
    def device(self):
        """Get device from underlying similarity model"""
        return self.similarity_model.device

    @property
    def model_config(self):
        """Get model config from underlying similarity model"""
        return self.similarity_model.model_config

    def _create_failed_result(
        self, image_input: str | Image.Image | Path, text_prompt: str, error_message: str, error_type: str | None = None
    ) -> EvaluationResult:
        """Create result for failed evaluation"""
        return EvaluationResult(
            image_path=str(image_input),
            text_prompt=text_prompt,
            clip_score=0.0,
            processing_time_ms=0.0,
            error=error_message,
            error_type=error_type,
        )
    
    async def _load_image(self, image_input: str | Image.Image | Path, image_loader: ImageLoader | None) -> Image.Image:
        """Load image from various input sources"""
        if isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif image_loader:
            image = await image_loader.load_image(image_input)
        else:
            async with ImageLoader() as loader:
                image = await loader.load_image(image_input)
        return image
    

    def _determine_source_type(self, image_input: str | Image.Image | Path) -> str:
        """Determine source type based on input"""
        if isinstance(image_input, str):
            if image_input.startswith(("http://", "https://")):
                source_type = "url"
            elif image_input.startswith("data:image/"):
                source_type = "base64"
            else:
                source_type = "file"
        else:
            source_type = "pil_image"
        return source_type


    async def _record_success_metrics(
            self, clip_score: float, inference_time: float, total_time: float, source_type: str
        ) -> None:
        """Record metrics for successful evaluation"""
        try:
            metrics = get_metrics_middleware()
            if metrics:
                metrics.record_inference_time(
                    inference_time / 1000,
                    self.model_config_name,
                    self.device.type,
                    self.similarity_model.model_name,
                )

                metrics.record_clip_score(clip_score, self.model_config_name, self.similarity_model.model_name)

                image_processing_time = (total_time - inference_time) / 1000
                metrics.record_image_processing_time(image_processing_time, source_type)

        except Exception as e:
            logger.debug(f"Metrics recording failed: {e}")
            pass

    
    def _extract_error_type(self, exception: Exception) -> str | None:
        """Extract error type from exception for metrics"""
        return getattr(exception, "error_type", None) if isinstance(exception, ServiceError) else None
    
    def _record_error_metrics(
        self, exception: Exception, error_type: str | None = None
    ) -> None:
        """Record metrics for evaluation errors"""
        try:
            metrics = get_metrics_middleware()
            if not metrics:
                return
            error_name = error_type or type(exception).__name__ 
            metrics.record_model_error(error_name, self.model_config_name, self.similarity_model.model_name)
            metrics.record_error_pattern(error_name, "evaluation", self.model_config_name)
        except Exception as metrics_exception:
            logger.debug(f"Metrics recording failed: {metrics_exception}")

    async def evaluate_single(
        self, image_input: str | Image.Image | Path, text_prompt: str, image_loader: ImageLoader | None = None
    ) -> EvaluationResult:
        """
        Evaluate single image-text pair asynchronously

        Args:
            image_input: Image source (URL, file path, or PIL Image)
            text_prompt: Text description to compare
            image_loader: Optional async image loader instance

        Returns:
            EvaluationResult with CLIP score
        """
        start_time = time.time()

        try:
            # Load and prepare image
            image = await self._load_image(image_input, image_loader)

            # Determine source type for metrics
            source_type = self._determine_source_type(image_input)

            # Use similarity model for inference
            clip_score, inference_time = await self.similarity_model.compute_similarity(image, text_prompt)
            total_time = (time.time() - start_time) * 1000

            # Record success metrics
            await self._record_success_metrics(clip_score, inference_time, total_time, source_type)

            return EvaluationResult(
                image_path=str(image_input),
                text_prompt=text_prompt,
                clip_score=clip_score,
                processing_time_ms=total_time,
                error=None,
            )

        except Exception as main_exception:
            logger.error(f"Evaluation failed for {image_input}: {main_exception}")
            error_type = self._extract_error_type(main_exception)

            # Record error metrics
            self._record_error_metrics(main_exception, error_type)

            return self._create_failed_result(image_input, text_prompt, str(main_exception), error_type)

    async def evaluate_batch(
        self,
        image_inputs: list[str | Image.Image | Path],
        text_prompts: list[str],
        batch_size: int = 8,
    ) -> list[EvaluationResult]:
        """
        Simplified batch evaluation using native batch processing
        
        Args:
            image_inputs: List of image sources
            text_prompts: List of text descriptions
            batch_size: GPU batch size (kept for API compatibility, not used yet)
            
        Returns:
            List of EvaluationResult objects
            
        TODO: Add chunking if GPU memory issues arise with large batches
        """
        if len(image_inputs) != len(text_prompts):
            raise ValidationError(f"Mismatch: {len(image_inputs)} images vs {len(text_prompts)} prompts")

        start_time = time.time()
        
        try:
            # Load all images concurrently using existing helper
            async with ImageLoader() as image_loader:
                image_tasks = [self._load_image(img_input, image_loader) for img_input in image_inputs]
                loaded_images = await asyncio.gather(*image_tasks, return_exceptions=True)
            
            # Separate successful and failed image loads
            valid_images = []
            valid_prompts = []
            results = []
            
            for i, (image_result, prompt) in enumerate(zip(loaded_images, text_prompts)):
                if isinstance(image_result, Exception):
                    # Create failed result for image loading error
                    failed_result = self._create_failed_result(
                        image_inputs[i], prompt, f"Image loading failed: {image_result}"
                    )
                    results.append(failed_result)
                else:
                    valid_images.append(image_result)
                    valid_prompts.append(prompt)
            
            # Process valid images with native batch processing
            if valid_images:
                try:
                    clip_scores, inference_time = await self.similarity_model.compute_batch_similarity(
                        valid_images, valid_prompts
                    )
                    
                    # Create results for successful evaluations
                    avg_time_per_item = ((time.time() - start_time) * 1000) / len(valid_images)
                    
                    valid_idx = 0
                    for i, (image_result, prompt) in enumerate(zip(loaded_images, text_prompts)):
                        if not isinstance(image_result, Exception):
                            results.insert(i, EvaluationResult(
                                image_path=str(image_inputs[i]),
                                text_prompt=prompt,
                                clip_score=clip_scores[valid_idx],
                                processing_time_ms=avg_time_per_item,
                                error=None,
                            ))
                            valid_idx += 1
                            
                except Exception as batch_error:
                    logger.error(f"Batch inference failed: {batch_error}")
                    # Create error results for all valid images
                    valid_idx = 0
                    for i, (image_result, prompt) in enumerate(zip(loaded_images, text_prompts)):
                        if not isinstance(image_result, Exception):
                            failed_result = self._create_failed_result(
                                image_inputs[i], prompt, f"Batch inference failed: {batch_error}"
                            )
                            results.insert(i, failed_result)
                            valid_idx += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            # Return error results for all inputs
            return [
                self._create_failed_result(img_input, prompt, f"Batch evaluation failed: {e}")
                for img_input, prompt in zip(image_inputs, text_prompts)
            ]

    @classmethod
    def create_fast_evaluator(cls, device: str | None = None, **kwargs) -> "MinimalOpenCLIPEvaluator":
        """Create evaluator optimized for speed"""
        return cls(model_config_name="fast", device=device, **kwargs)

    @classmethod
    def create_accurate_evaluator(cls, device: str | None = None, **kwargs) -> "MinimalOpenCLIPEvaluator":
        """Create evaluator optimized for accuracy"""
        return cls(model_config_name="accurate", device=device, **kwargs)
