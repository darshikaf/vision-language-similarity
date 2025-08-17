from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

from service.core.ml.utils.types import EvaluationResult


class AbstractEvaluator(ABC):
    """
    Abstract base class for all evaluation implementations.

    Provides a consistent interface for different evaluator types while
    allowing each implementation to handle the specifics of their model type.

    Extension pattern:
    1. Inherit from AbstractEvaluator
    2. Implement evaluate_single and evaluate_batch methods
    3. Add any model-specific initialization and utilities
    """

    @abstractmethod
    async def evaluate_single(
        self, image_input: str | Image.Image | Path, text_prompt: str, **kwargs: Any
    ) -> EvaluationResult:
        """
        Evaluate a single image-text pair.

        Args:
            image_input: Image source (URL, file path, or PIL Image)
            text_prompt: Text description to compare
            **kwargs: Implementation-specific arguments

        Returns:
            EvaluationResult with evaluation outcome
        """
        pass

    @abstractmethod
    async def evaluate_batch(
        self, image_inputs: list[str | Image.Image | Path], text_prompts: list[str], **kwargs: Any
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple image-text pairs.

        Args:
            image_inputs: List of image sources
            text_prompts: List of text descriptions
            **kwargs: Implementation-specific arguments

        Returns:
            List of EvaluationResult objects
        """
        pass

    @property
    @abstractmethod
    def device(self):
        """Get the device used by this evaluator."""
        pass

    @property
    @abstractmethod
    def model_config(self):
        """Get the model configuration used by this evaluator."""
        pass
