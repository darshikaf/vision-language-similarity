import time
from pathlib import Path
from typing import Any, List

from PIL import Image

from service.core.exceptions import ValidationError
from service.log import get_logger
from service.core.ml.utils.image_processor import ImageProcessor
from service.core.ml.utils.metrics_recorder import MetricsRecorder
from service.core.ml.utils.result_builder import ResultBuilder
from service.core.types import EvaluationResult

from .base_evaluator import AbstractEvaluator
from .model_manager import ModelManager

logger = get_logger(__name__)


class OpenCLIPEvaluator(AbstractEvaluator):
    """
    OpenCLIP evaluator for single and batch evaluation operations.
    
    Evaluates image-text similarity using OpenCLIP models.
    """

    def __init__(
        self,
        model_config_name: str = "fast",
        device: str | None = None,
        image_processor: ImageProcessor | None = None,
        metrics_recorder: MetricsRecorder | None = None,
        result_builder: ResultBuilder | None = None,
        **model_kwargs: Any,
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
        self.device_preference = device
        self.model_kwargs = model_kwargs

        # Initialize shared resources
        self.model_manager = ModelManager()
        self.image_processor = image_processor or ImageProcessor()
        self.metrics_recorder = metrics_recorder or MetricsRecorder()
        self.result_builder = result_builder or ResultBuilder()
        
        # Pre-load model for property access
        self.similarity_model = self.model_manager.get_model(
            model_config_name, device=device, **model_kwargs
        )

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
        self, 
        image_input: str | Image.Image | Path, 
        text_prompt: str, 
        image_loader=None,
        **kwargs: Any
    ) -> EvaluationResult:
        """
        Evaluate single image-text pair.

        Args:
            image_input: Image source (URL, file path, or PIL Image)
            text_prompt: Text description to compare
            image_loader: Optional async image loader instance (deprecated)
            **kwargs: Additional arguments (unused, for interface compatibility)

        Returns:
            EvaluationResult with evaluation outcome
        """
        start_time = time.time()
        
        try:
            # Step 1: Load and prepare image
            image = await self.image_processor.load_image(image_input, image_loader)
            source_type = self.image_processor.determine_source_type(image_input)
            
            # Step 2: Run model inference
            clip_score, inference_time = await self.similarity_model.compute_similarity(image, text_prompt)
            total_time = (time.time() - start_time) * 1000
            
            # Step 3: Record success metrics
            await self.metrics_recorder.record_success_metrics(
                clip_score,
                inference_time,
                total_time,
                source_type,
                self.model_config_name,
                self.device.type,
                self.similarity_model.model_name,
            )
            
            # Step 4: Build success result
            return self.result_builder.create_success_result(
                image_input, text_prompt, clip_score, total_time
            )
            
        except Exception as error:
            logger.error(f"Single evaluation failed for {image_input}: {error}")
            
            # Handle error case
            error_type = self.metrics_recorder.extract_error_type(error)
            
            # Record error metrics
            self.metrics_recorder.record_error_metrics(
                error, self.model_config_name, self.similarity_model.model_name, error_type
            )
            
            # Build error result
            return self.result_builder.create_failed_result(
                image_input, text_prompt, str(error), error_type
            )

    async def evaluate_batch(
        self,
        image_inputs: List[str | Image.Image | Path],
        text_prompts: List[str],
        batch_size: int = 8,
        **kwargs: Any
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple image-text pairs with batch optimization.

        Args:
            image_inputs: List of image sources
            text_prompts: List of text descriptions
            batch_size: GPU batch size for optimization
            **kwargs: Additional arguments (unused, for interface compatibility)

        Returns:
            List of EvaluationResult objects
        """
        # Validation
        if len(image_inputs) != len(text_prompts):
            raise ValidationError(
                f"Mismatch: {len(image_inputs)} images vs {len(text_prompts)} prompts"
            )
        
        if not image_inputs:
            return []
        
        start_time = time.time()
        
        try:
            # Step 1: Load all images concurrently
            loaded_images = await self.image_processor.load_images_batch(image_inputs)
            
            # Step 2: Separate successful and failed image loads
            valid_images, valid_prompts, valid_inputs, results = self._separate_valid_invalid_images(
                loaded_images, text_prompts, image_inputs
            )
            
            # Step 3: Process valid images with batch optimization
            if valid_images:
                success_results = await self._process_valid_batch(
                    valid_images, valid_prompts, valid_inputs, start_time
                )
                
                # Step 4: Merge results back in correct order
                results = self._merge_results_in_order(
                    loaded_images, results, success_results
                )
            
            return results
            
        except Exception as error:
            logger.error(f"Batch evaluation failed: {error}")
            # Return error results for all inputs
            return self.result_builder.create_batch_error_results(
                image_inputs, text_prompts, f"Batch evaluation failed: {error}"
            )
    
    def _separate_valid_invalid_images(
        self,
        loaded_images: List[Image.Image | Exception],
        text_prompts: List[str],
        image_inputs: List[str | Image.Image | Path],
    ) -> tuple[List[Image.Image], List[str], List[str | Image.Image | Path], List[EvaluationResult]]:
        """Separate successfully loaded images from failed ones."""
        valid_images = []
        valid_prompts = []
        valid_inputs = []
        results = []
        
        for i, (image_result, prompt) in enumerate(zip(loaded_images, text_prompts, strict=False)):
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
        
        return valid_images, valid_prompts, valid_inputs, results
    
    async def _process_valid_batch(
        self,
        valid_images: List[Image.Image],
        valid_prompts: List[str],
        valid_inputs: List[str | Image.Image | Path],
        start_time: float,
    ) -> List[EvaluationResult]:
        """Process batch of valid images."""
        try:
            # Native batch processing
            clip_scores, inference_time = await self.similarity_model.compute_batch_similarity(
                valid_images, valid_prompts
            )
            
            # Calculate average time per item
            avg_time_per_item = ((time.time() - start_time) * 1000) / len(valid_images)
            
            # Create success results
            return self.result_builder.create_batch_results(
                valid_inputs, valid_prompts, clip_scores, avg_time_per_item
            )
            
        except Exception as batch_error:
            logger.error(f"Batch inference failed: {batch_error}")
            # Create error results for all valid images
            return [
                self.result_builder.create_failed_result(
                    image_input, prompt, f"Batch inference failed: {batch_error}"
                )
                for image_input, prompt in zip(valid_inputs, valid_prompts, strict=False)
            ]
    
    def _merge_results_in_order(
        self,
        loaded_images: List[Image.Image | Exception],
        failed_results: List[EvaluationResult],
        success_results: List[EvaluationResult],
    ) -> List[EvaluationResult]:
        """Merge success and failed results back in original order."""
        final_results = []
        success_idx = 0
        failed_idx = 0
        
        for image_result in loaded_images:
            if isinstance(image_result, Exception):
                final_results.append(failed_results[failed_idx])
                failed_idx += 1
            else:
                final_results.append(success_results[success_idx])
                success_idx += 1
        
        return final_results

