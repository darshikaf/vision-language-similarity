import asyncio
import logging
from pathlib import Path
import time

from PIL import Image
from tqdm import tqdm

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
            # Load image asynchronously if needed
            if isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            elif image_loader:
                image = await image_loader.load_image(image_input)
            else:
                async with ImageLoader() as loader:
                    image = await loader.load_image(image_input)

            # -- Metrics: Determine source type
            if isinstance(image_input, str):
                if image_input.startswith(("http://", "https://")):
                    source_type = "url"
                elif image_input.startswith("data:image/"):
                    source_type = "base64"
                else:
                    source_type = "file"
            else:
                source_type = "pil_image"

            # Use similarity model for inference
            clip_score, inference_time = await self.similarity_model.compute_similarity(image, text_prompt)
            total_time = (time.time() - start_time) * 1000

            # -- Metrics
            try:
                metrics = get_metrics_middleware()
                if metrics:
                    metrics.record_inference_time(
                        inference_time / 1000, self.model_config_name, self.device.type, self.similarity_model.model_name
                    )

                    metrics.record_clip_score(clip_score, self.model_config_name, self.similarity_model.model_name)

                    image_processing_time = (total_time - inference_time) / 1000
                    metrics.record_image_processing_time(image_processing_time, source_type)

            except Exception as e:
                logger.debug(f"Metrics recording failed: {e}")
                pass

            return EvaluationResult(
                image_path=str(image_input),
                text_prompt=text_prompt,
                clip_score=clip_score,
                processing_time_ms=total_time,
                error=None,
            )

        except Exception as e:
            logger.error(f"Async evaluation failed for {image_input}: {e}")
            error_type = getattr(e, "error_type", None) if isinstance(e, ServiceError) else None

            # -- Metrics: model-specific errors
            try:
                metrics = get_metrics_middleware()
                if metrics:
                    error_name = error_type or type(e).__name__
                    metrics.record_model_error(error_name, self.model_config_name, self.similarity_model.model_name)
                    metrics.record_error_pattern(error_name, "evaluation", self.model_config_name)
            except Exception as e:
                logger.debug(f"Metrics recording failed: {e}")

            return self._create_failed_result(image_input, text_prompt, str(e), error_type)

    async def evaluate_batch(  # noqa: C901
        self,
        image_inputs: list[str | Image.Image | Path],
        text_prompts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        max_concurrent_loads: int | None = None,
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple image-text pairs using efficient async batch processing

        Args:
            image_inputs: List of image sources
            text_prompts: List of text descriptions
            batch_size: Batch size for PyTorch processing
            show_progress: Show progress bar
            max_concurrent_loads: Max concurrent image loads

        Returns:
            List of EvaluationResult objects
        """
        if len(image_inputs) != len(text_prompts):
            raise ValidationError(f"Mismatch: {len(image_inputs)} images vs {len(text_prompts)} prompts")

        results = []
        total_batches = (len(image_inputs) + batch_size - 1) // batch_size

        # Use semaphore to limit concurrent image loading
        concurrent_loads = max_concurrent_loads or self.max_concurrent_loads
        load_semaphore = asyncio.Semaphore(concurrent_loads)

        async with ImageLoader() as image_loader:
            iterator = range(0, len(image_inputs), batch_size)
            if show_progress:
                iterator = tqdm(iterator, total=total_batches, desc="Processing async batches")

            for batch_start in iterator:
                batch_end = min(batch_start + batch_size, len(image_inputs))
                batch_images_raw = image_inputs[batch_start:batch_end]
                batch_prompts = text_prompts[batch_start:batch_end]

                # Load all images in batch concurrently with semaphore
                async def load_with_semaphore(img_input, idx):
                    async with load_semaphore:
                        try:
                            if isinstance(img_input, Image.Image):
                                return img_input.convert("RGB"), idx, None
                            else:
                                image = await image_loader.load_image(img_input)
                                return image, idx, None # (image_result, idx, error_message)
                        except Exception as e:
                            logger.error(f"Failed to load image {img_input}: {e}")
                            error_type = getattr(e, "error_type", None) if isinstance(e, ServiceError) else None
                            return None, idx, str(e), error_type

                # Load all images in batch concurrently
                load_tasks = [load_with_semaphore(img_input, idx) for idx, img_input in enumerate(batch_images_raw)]

                loaded_results = await asyncio.gather(*load_tasks, return_exceptions=True)

                # Separate successful and failed loads
                batch_images = []
                valid_indices = []
                failed_results = []

                for result in loaded_results:
                    if len(result) == 4:  # image_result, original_idx, error, error_type
                        image_result, original_idx, error, error_type = result
                    else:
                        image_result, original_idx, error = result
                        error_type = None

                    if error or image_result is None:
                        failed_results.append(
                            self._create_failed_result(
                                batch_images_raw[original_idx],
                                batch_prompts[original_idx],
                                error or "Image loading failed",
                                error_type,
                            )
                        )
                    else:
                        batch_images.append(image_result)
                        valid_indices.append(original_idx)

                # Process valid images in PyTorch batch
                if batch_images:
                    try:
                        valid_prompts = [batch_prompts[i] for i in valid_indices]

                        # Use similarity model for batch inference
                        clip_scores, batch_time = await self.similarity_model.compute_batch_similarity(
                            batch_images, valid_prompts
                        )

                        avg_time_per_item = batch_time / len(batch_images)

                        # Create results for valid images
                        for _, (valid_idx, clip_score) in enumerate(zip(valid_indices, clip_scores, strict=False)):
                            results.append(
                                EvaluationResult(
                                    image_path=str(batch_images_raw[valid_idx]),
                                    text_prompt=batch_prompts[valid_idx],
                                    clip_score=clip_score,
                                    processing_time_ms=avg_time_per_item,
                                    error=None,
                                )
                            )

                    except Exception as e:
                        logger.error(f"Batch PyTorch processing failed: {e}")
                        # Fall back to individual async processing
                        individual_tasks = [
                            self.evaluate_single(batch_images_raw[valid_idx], batch_prompts[valid_idx], image_loader)
                            for valid_idx in valid_indices
                        ]
                        individual_results = await asyncio.gather(*individual_tasks)
                        results.extend(individual_results)

                # Add failed results
                results.extend(failed_results)

        return results

    @classmethod
    def create_fast_evaluator(cls, device: str | None = None, **kwargs) -> "MinimalOpenCLIPEvaluator":
        """Create evaluator optimized for speed"""
        return cls(model_config_name="fast", device=device, **kwargs)

    @classmethod
    def create_accurate_evaluator(cls, device: str | None = None, **kwargs) -> "MinimalOpenCLIPEvaluator":
        """Create evaluator optimized for accuracy"""
        return cls(model_config_name="accurate", device=device, **kwargs)
