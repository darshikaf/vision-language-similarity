import asyncio
import logging
from pathlib import Path
import time
from typing import Optional

from PIL import Image

from service.core.exceptions import ValidationError
from service.core.types import EvaluationResult
from service.core.ml.models.similarity_models import SimilarityModelFactory
from service.core.ml.utils.image_processor import ImageProcessor
from service.core.ml.utils.metrics_recorder import MetricsRecorder
from service.core.ml.utils.result_builder import ResultBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MinimalOpenCLIPEvaluator:
    """
    OpenCLIP evaluator focused on evaluation orchestration.
    
    Responsibilities:
    - Orchestrate evaluation workflow
    - Coordinate between image processing, model inference, and result creation
    - Handle high-level async operations
    - Manage similarity model lifecycle
    
    Uses dependency injection for:
    - Image processing (ImageProcessor)
    - Metrics recording (MetricsRecorder) 
    - Result building (ResultBuilder)
    """

    def __init__(
        self, 
        model_config_name: str = "fast", 
        device: Optional[str] = None,
        image_processor: Optional[ImageProcessor] = None,
        metrics_recorder: Optional[MetricsRecorder] = None,
        result_builder: Optional[ResultBuilder] = None,
        **model_kwargs
    ):
        """
        Initialize evaluator with pluggable components.

        Args:
            model_config_name: Model configuration name ("fast", "accurate", or custom config)
            device: Device for computation (auto-detected if None)
            image_processor: Optional image processor (creates default if None)
            metrics_recorder: Optional metrics recorder (creates default if None)
            result_builder: Optional result builder (creates default if None)
            **model_kwargs: Additional arguments passed to the similarity model
        """
        self.model_config_name = model_config_name
        
        # Initialize similarity model
        self.similarity_model = SimilarityModelFactory.create_model(model_config_name, device=device, **model_kwargs)
        
        # Initialize utility components with defaults if not provided
        self.image_processor = image_processor or ImageProcessor()
        self.metrics_recorder = metrics_recorder or MetricsRecorder()
        self.result_builder = result_builder or ResultBuilder()

        logger.info(f"Initialized evaluator with {self.similarity_model.model_name} model ({model_config_name} config)")

    @property
    def device(self):
        """Get device from underlying similarity model"""
        return self.similarity_model.device

    @property
    def model_config(self):
        """Get model config from underlying similarity model"""
        return self.similarity_model.model_config


    async def evaluate_single(
        self, image_input: str | Image.Image | Path, text_prompt: str, image_loader = None
    ) -> EvaluationResult:
        """
        Evaluate single image-text pair asynchronously

        Args:
            image_input: Image source (URL, file path, or PIL Image)
            text_prompt: Text description to compare
            image_loader: Optional async image loader instance (deprecated, use image_processor)

        Returns:
            EvaluationResult with CLIP score
        """
        start_time = time.time()

        try:
            # Load and prepare image using image processor
            image = await self.image_processor.load_image(image_input, image_loader)

            # Determine source type for metrics
            source_type = self.image_processor.determine_source_type(image_input)

            # Use similarity model for inference
            clip_score, inference_time = await self.similarity_model.compute_similarity(image, text_prompt)
            total_time = (time.time() - start_time) * 1000

            # Record success metrics
            await self.metrics_recorder.record_success_metrics(
                clip_score, inference_time, total_time, source_type,
                self.model_config_name, self.device.type, self.similarity_model.model_name
            )

            # Create success result
            return self.result_builder.create_success_result(
                image_input, text_prompt, clip_score, total_time
            )

        except Exception as main_exception:
            logger.error(f"Evaluation failed for {image_input}: {main_exception}")
            error_type = self.metrics_recorder.extract_error_type(main_exception)

            # Record error metrics
            self.metrics_recorder.record_error_metrics(
                main_exception, self.model_config_name, self.similarity_model.model_name, error_type
            )

            # Create failed result
            return self.result_builder.create_failed_result(image_input, text_prompt, str(main_exception), error_type)

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
            # Load all images concurrently using image processor
            loaded_images = await self.image_processor.load_images_batch(image_inputs)
            
            # Separate successful and failed image loads
            valid_images = []
            valid_prompts = []
            valid_inputs = []
            results = []
            
            for i, (image_result, prompt) in enumerate(zip(loaded_images, text_prompts)):
                if isinstance(image_result, Exception):
                    # Create failed result for image loading error
                    failed_result = self.result_builder.create_failed_result(
                        image_inputs[i], prompt, f"Image loading failed: {image_result}"
                    )
                    results.append(failed_result)
                else:
                    valid_images.append(image_result)
                    valid_prompts.append(prompt)
                    valid_inputs.append(image_inputs[i])
            
            # Process valid images with native batch processing
            if valid_images:
                try:
                    clip_scores, inference_time = await self.similarity_model.compute_batch_similarity(
                        valid_images, valid_prompts
                    )
                    
                    # Create results for successful evaluations
                    avg_time_per_item = ((time.time() - start_time) * 1000) / len(valid_images)
                    
                    # Create success results using result builder
                    success_results = self.result_builder.create_batch_results(
                        valid_inputs, valid_prompts, clip_scores, avg_time_per_item
                    )
                    
                    # Merge results back in correct order
                    valid_idx = 0
                    for i, image_result in enumerate(loaded_images):
                        if not isinstance(image_result, Exception):
                            results.insert(i, success_results[valid_idx])
                            valid_idx += 1
                            
                except Exception as batch_error:
                    logger.error(f"Batch inference failed: {batch_error}")
                    # Create error results for all valid images
                    valid_idx = 0
                    for i, (image_result, prompt) in enumerate(zip(loaded_images, text_prompts)):
                        if not isinstance(image_result, Exception):
                            failed_result = self.result_builder.create_failed_result(
                                image_inputs[i], prompt, f"Batch inference failed: {batch_error}"
                            )
                            results.insert(i, failed_result)
                            valid_idx += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            # Return error results for all inputs using result builder
            return self.result_builder.create_batch_error_results(
                image_inputs, text_prompts, f"Batch evaluation failed: {e}"
            )

    @classmethod
    def create_fast_evaluator(cls, device: Optional[str] = None, **kwargs) -> "MinimalOpenCLIPEvaluator":
        """Create evaluator optimized for speed"""
        return cls(model_config_name="fast", device=device, **kwargs)

    @classmethod
    def create_accurate_evaluator(cls, device: Optional[str] = None, **kwargs) -> "MinimalOpenCLIPEvaluator":
        """Create evaluator optimized for accuracy"""
        return cls(model_config_name="accurate", device=device, **kwargs)
