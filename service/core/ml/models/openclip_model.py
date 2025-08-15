import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import time

import open_clip
from PIL import Image
import torch
import torch.nn.functional as F  # noqa: F401, N812

from service.core.device_manager import DeviceManager
from service.core.exceptions import ModelError, ValidationError
from service.core.ml.models.base import SimilarityModel
from service.core.observability import get_metrics_middleware

logger = logging.getLogger(__name__)


class OpenCLIPSimilarityModel(SimilarityModel):
    """
    Provides OpenCLIP-based vision-language similarity computation with optimizations
    for batch processing and mixed precision inference.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        cache_dir: str | None = None,
        max_workers: int | None = None,
        model_config: str = "fast",
    ):
        super().__init__(model_name, device, model_config)
        self.pretrained = pretrained
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "openclip")
        self.precision = DeviceManager.get_optimal_precision(self.device)

        # Thread pool for PyTorch operations
        self.max_workers = max_workers or 4
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="torch_")

        # Initialize components with timing
        load_start_time = time.time()

        # Model loading is done at runtime to allow dynamic model selection
        self.model, self.preprocess = self._load_model_sync()
        load_duration = time.time() - load_start_time
        self._model_loaded = True

        # Record model loading time
        try:
            metrics = get_metrics_middleware()
            if metrics:
                metrics.record_model_load_time(load_duration, model_config, self.model_name)
                logger.info(f"Model {self.model_name} loaded in {load_duration:.2f}s")
        except Exception:
            # Metrics not available - continue without metrics
            logger.info(f"Model {self.model_name} loaded in {load_duration:.2f}s (metrics unavailable)")

    def _load_model_sync(self):
        """Synchronous model loading"""
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

            # Record model loading error
            try:
                metrics = get_metrics_middleware()
                if metrics:
                    metrics.record_model_error("model_load_error", self.model_config, self.model_name)
            except Exception as metrics_error:
                logger.debug(f"Metrics recording failed: {metrics_error}")

            raise ModelError(f"Failed to load model {self.model_name}/{self.pretrained}: {e}") from e

    def _calculate_clip_score(self, raw_cosine: float) -> float:
        """Convert to CLIP standard score: max(100 * cosine, 0)"""
        return max(100 * raw_cosine, 0)

    async def compute_similarity(self, image: Image.Image, text_prompt: str) -> tuple[float, float]:
        """
        Compute similarity between single image and text.

        Args:
            image: PIL Image in RGB format
            text_prompt: Text description to compare

        Returns:
            Tuple of (clip_score, processing_time_ms)
        """
        # Offload synchronous PyTorch operations to thread pool
        loop = asyncio.get_event_loop()
        raw_cosine, processing_time = await loop.run_in_executor(
            self._executor, self._run_inference_sync, image, text_prompt
        )

        clip_score = self._calculate_clip_score(raw_cosine)
        return clip_score, processing_time

    async def compute_batch_similarity(
        self, images: list[Image.Image], text_prompts: list[str]
    ) -> tuple[list[float], float]:
        """
        Compute similarity for batch of image-text pairs.

        Args:
            images: List of PIL Images in RGB format
            text_prompts: List of text descriptions

        Returns:
            Tuple of (clip_scores, total_processing_time_ms)
        """
        if len(images) != len(text_prompts):
            raise ValidationError(f"Mismatch: {len(images)} images vs {len(text_prompts)} prompts")

        # Run batch inference in thread pool
        loop = asyncio.get_event_loop()
        raw_cosines, processing_time = await loop.run_in_executor(
            self._executor, self._run_batch_inference_sync, images, text_prompts
        )

        # Convert to CLIP scores
        clip_scores = [self._calculate_clip_score(cosine) for cosine in raw_cosines]

        # Record batch efficiency (estimate single operation time for comparison)
        try:
            metrics = get_metrics_middleware()
            if metrics:
                # Hypothesis: Batch should be more efficient than individual calls
                estimated_single_time = processing_time / len(images) * 1.2  # Add 20% overhead estimate
                total_single_time = estimated_single_time * len(images)
                efficiency_ratio = processing_time / total_single_time if total_single_time > 0 else 1.0

                metrics.record_batch_efficiency(efficiency_ratio, self.model_config, self.model_name)

                logger.debug(f"Batch efficiency: {efficiency_ratio:.3f} for {len(images)} items")
        except Exception as e:
            logger.debug(f"Metrics recording failed: {e}")

        return clip_scores, processing_time

    def _run_inference_sync(self, image: Image.Image, text_prompt: str) -> tuple[float, float]:
        """
        Run PyTorch inference synchronously in thread pool

        """
        try:
            with torch.no_grad():
                start_time = time.time()

                # 1. Preprocess image
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

                # 2. Tokenize text
                text_tokens = open_clip.tokenize([text_prompt]).to(self.device)

                # 3. Extract features
                image_features = F.normalize(self.model.encode_image(image_tensor), p=2, dim=-1)

                # 4. Normalize text features
                text_features = F.normalize(self.model.encode_text(text_tokens), p=2, dim=-1)

                # 5. Calculate cosine similarity
                raw_cosine = torch.cosine_similarity(image_features, text_features, dim=-1).item()
                # TODO: Profile if below implementation is faster because L2 normalization is already applied
                # raw_cosine = torch.sum(image_features * text_features, dim=-1).item()

                processing_time = (time.time() - start_time) * 1000
                return raw_cosine, processing_time
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM during inference: {e}")
            raise ModelError("Out of memory during inference. Try reducing batch size or image resolution.") from e
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise ModelError(f"Failed to compute similarity: {e}") from e

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

    def get_model_info(self) -> dict:
        """Get OpenCLIP-specific model information"""
        base_info = super().get_model_info()
        base_info.update(
            {
                "pretrained": self.pretrained,
                "precision": self.precision,
                "cache_dir": self.cache_dir,
                "max_workers": self.max_workers,
            }
        )
        return base_info

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup"""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources including thread pool with graceful shutdown"""
        try:
            if hasattr(self, "_executor") and self._executor:
                # Graceful shutdown - wait for current tasks to complete
                self._executor.shutdown(wait=True)
                self._executor = None
                logger.debug(f"Cleaned up thread pool for model {self.model_name}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
            # Force shutdown if graceful shutdown failed
            if hasattr(self, "_executor") and self._executor:
                try:
                    self._executor.shutdown(wait=False)
                    self._executor = None
                except Exception:
                    pass  # Ignore force shutdown errors

    def __del__(self):
        """Minimal cleanup on object deletion"""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors in __del__ to avoid issues
