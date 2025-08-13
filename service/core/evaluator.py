"""
OpenCLIP evaluator with thread pool execution for PyTorch operations
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import time

import open_clip
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .device_manager import DeviceManager
from .image_loader import ImageLoader
from .models import EvaluationResult
from ..observability.prometheus_middleware import get_metrics_middleware

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
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        cache_dir: str | None = None,
        max_workers: int | None = None,
    ):
        """
        Initialize evaluator

        Args:
            model_name: OpenCLIP model architecture
            pretrained: Pretrained weights
            device: Device for computation (auto-detected if None)
            cache_dir: Directory for caching models
            max_workers: Max workers for thread pool (defaults to CPU count)
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "openclip")
        
        # Determine model config for metrics
        if model_name == "ViT-B-32":
            self.model_config = "fast"
        elif model_name == "ViT-L-14":
            self.model_config = "accurate"
        else:
            self.model_config = "custom"

        # Setup device and precision
        self.device = DeviceManager.get_optimal_device(device)
        self.precision = DeviceManager.get_optimal_precision(self.device)

        # Thread pool for PyTorch operations
        self.max_workers = max_workers or 4  # Simple default, configurable via constructor
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="torch_")

        # Initialize components
        self.model, self.preprocess = self._load_model()

        logger.info(f"Initialized async {model_name}/{pretrained} on {self.device} with {self.max_workers} workers")

    def _load_model(self):
        """Load OpenCLIP model with error handling"""
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
                precision=self.precision,
                cache_dir=self.cache_dir,
            )
            model.eval()

            # Apply mixed precision optimization
            if self.precision == "fp16" and self.device.type == "cuda":
                model = model.half()

            return model, preprocess

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}/{self.pretrained}: {e}")
            raise

    def _calculate_clip_score(self, raw_cosine: float) -> float:
        """Convert to CLIP standard score: max(100 * cosine, 0)"""
        return max(100 * raw_cosine, 0)

    def _create_failed_result(
        self, image_input: str | Image.Image | Path, text_prompt: str, error_message: str
    ) -> EvaluationResult:
        """Create result for failed evaluation"""
        return EvaluationResult(
            image_path=str(image_input),
            text_prompt=text_prompt,
            clip_score=0.0,
            processing_time_ms=0.0,
            error=error_message,
        )

    def _run_inference_sync(self, image: Image.Image, text_prompt: str) -> tuple[float, float]:
        """Run PyTorch inference synchronously in thread pool"""
        with torch.no_grad():
            start_time = time.time()

            # Preprocess image
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # Tokenize text
            text_tokens = open_clip.tokenize([text_prompt]).to(self.device)

            # Extract and normalize features
            image_features = F.normalize(self.model.encode_image(image_tensor), p=2, dim=-1)
            text_features = F.normalize(self.model.encode_text(text_tokens), p=2, dim=-1)

            # Calculate similarity
            raw_cosine = torch.cosine_similarity(image_features, text_features, dim=-1).item()

            processing_time = (time.time() - start_time) * 1000
            return raw_cosine, processing_time

    def _run_batch_inference_sync(
        self, images: list[Image.Image], text_prompts: list[str]
    ) -> tuple[list[float], float]:
        """Run batch PyTorch inference synchronously in thread pool"""
        with torch.no_grad():
            start_time = time.time()

            # Stack preprocessed images into a single tensor
            image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)

            # Tokenize all texts at once
            text_tokens = open_clip.tokenize(text_prompts).to(self.device)

            # Extract features in batch
            image_features = F.normalize(self.model.encode_image(image_tensors), p=2, dim=-1)
            text_features = F.normalize(self.model.encode_text(text_tokens), p=2, dim=-1)

            # Calculate similarities for entire batch
            cosine_similarities = torch.sum(image_features * text_features, dim=-1)
            raw_cosines = [sim.item() for sim in cosine_similarities]

            processing_time = (time.time() - start_time) * 1000
            return raw_cosines, processing_time

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

            # Determine source type for metrics
            if isinstance(image_input, str):
                if image_input.startswith(("http://", "https://")):
                    source_type = "url"
                elif image_input.startswith("data:image/"):
                    source_type = "base64"
                else:
                    source_type = "file"
            else:
                source_type = "pil_image"

            # Run PyTorch inference in thread pool
            loop = asyncio.get_event_loop()
            raw_cosine, inference_time = await loop.run_in_executor(
                self._executor, self._run_inference_sync, image, text_prompt
            )

            clip_score = self._calculate_clip_score(raw_cosine)
            total_time = (time.time() - start_time) * 1000

            # Record metrics (safely handle missing middleware)
            try:
                metrics = get_metrics_middleware()
                
                # Record inference timing (using inference_time in seconds)
                metrics.record_inference_time(inference_time / 1000, self.model_config, self.device.type)
                
                # Record image processing time (total_time - inference_time)
                image_processing_time = (total_time - inference_time) / 1000
                metrics.record_image_processing_time(image_processing_time, source_type)
                
            except (ImportError, RuntimeError):
                # Metrics middleware not available - continue without metrics
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
            return self._create_failed_result(image_input, text_prompt, str(e))

    async def evaluate_batch(
        self,
        image_inputs: list[str | Image.Image | Path],
        text_prompts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        max_concurrent_loads: int = 10,
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
            raise ValueError(f"Mismatch: {len(image_inputs)} images vs {len(text_prompts)} prompts")

        results = []
        total_batches = (len(image_inputs) + batch_size - 1) // batch_size

        # Use semaphore to limit concurrent image loading
        load_semaphore = asyncio.Semaphore(max_concurrent_loads)

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
                                return image, idx, None
                        except Exception as e:
                            logger.error(f"Failed to load image {img_input}: {e}")
                            return None, idx, str(e)

                # Load all images in batch concurrently
                load_tasks = [load_with_semaphore(img_input, idx) for idx, img_input in enumerate(batch_images_raw)]

                loaded_results = await asyncio.gather(*load_tasks, return_exceptions=True)

                # Separate successful and failed loads
                batch_images = []
                valid_indices = []
                failed_results = []

                for image_result, original_idx, error in loaded_results:
                    if error or image_result is None:
                        failed_results.append(
                            self._create_failed_result(
                                batch_images_raw[original_idx],
                                batch_prompts[original_idx],
                                error or "Image loading failed",
                            )
                        )
                    else:
                        batch_images.append(image_result)
                        valid_indices.append(original_idx)

                # Process valid images in PyTorch batch
                if batch_images:
                    try:
                        valid_prompts = [batch_prompts[i] for i in valid_indices]

                        # Run batch inference in thread pool
                        loop = asyncio.get_event_loop()
                        raw_cosines, batch_time = await loop.run_in_executor(
                            self._executor, self._run_batch_inference_sync, batch_images, valid_prompts
                        )

                        avg_time_per_item = batch_time / len(batch_images)

                        # Create results for valid images
                        for idx, (valid_idx, raw_cosine) in enumerate(zip(valid_indices, raw_cosines, strict=False)):
                            clip_score = self._calculate_clip_score(raw_cosine)

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
        return cls(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=device, **kwargs)

    @classmethod
    def create_accurate_evaluator(cls, device: str | None = None, **kwargs) -> "MinimalOpenCLIPEvaluator":
        """Create evaluator optimized for accuracy"""
        return cls(model_name="ViT-L-14", pretrained="laion2b_s32b_b82k", device=device, **kwargs)
