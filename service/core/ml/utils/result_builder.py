from pathlib import Path

from PIL import Image

from service.core.types import EvaluationResult


class ResultBuilder:
    """
    Handles creation and formatting of evaluation results.

    Responsibilities:
    - Create successful evaluation results
    - Create failed evaluation results with consistent error handling
    - Standardize result formatting across the application
    - Handle error type categorization
    """

    @staticmethod
    def create_success_result(
        image_input: str | Image.Image | Path, text_prompt: str, clip_score: float, processing_time_ms: float
    ) -> EvaluationResult:
        """
        Create result for successful evaluation.

        Args:
            image_input: Original image input
            text_prompt: Text prompt used for evaluation
            clip_score: Computed CLIP similarity score
            processing_time_ms: Total processing time in milliseconds

        Returns:
            EvaluationResult with success data
        """
        return EvaluationResult(
            image_path=str(image_input),
            text_prompt=text_prompt,
            clip_score=clip_score,
            processing_time_ms=processing_time_ms,
            error=None,
        )

    @staticmethod
    def create_failed_result(
        image_input: str | Image.Image | Path, text_prompt: str, error_message: str, error_type: str | None = None
    ) -> EvaluationResult:
        """
        Create result for failed evaluation.

        Args:
            image_input: Original image input
            text_prompt: Text prompt used for evaluation
            error_message: Error description
            error_type: Optional categorized error type

        Returns:
            EvaluationResult with error information
        """
        return EvaluationResult(
            image_path=str(image_input),
            text_prompt=text_prompt,
            clip_score=0.0,
            processing_time_ms=0.0,
            error=error_message,
            error_type=error_type,
        )

    @staticmethod
    def create_batch_results(
        image_inputs: list[str | Image.Image | Path],
        text_prompts: list[str],
        clip_scores: list[float],
        avg_processing_time_ms: float,
    ) -> list[EvaluationResult]:
        """
        Create results for successful batch evaluation.

        Args:
            image_inputs: Original image inputs
            text_prompts: Text prompts used for evaluation
            clip_scores: Computed CLIP similarity scores
            avg_processing_time_ms: Average processing time per item

        Returns:
            List of EvaluationResult objects
        """
        return [
            ResultBuilder.create_success_result(image_input, text_prompt, clip_score, avg_processing_time_ms)
            for image_input, text_prompt, clip_score in zip(image_inputs, text_prompts, clip_scores, strict=False)
        ]

    @staticmethod
    def create_batch_error_results(
        image_inputs: list[str | Image.Image | Path],
        text_prompts: list[str],
        error_message: str,
        error_type: str | None = None,
    ) -> list[EvaluationResult]:
        """
        Create error results for failed batch evaluation.

        Args:
            image_inputs: Original image inputs
            text_prompts: Text prompts used for evaluation
            error_message: Error description
            error_type: Optional categorized error type

        Returns:
            List of EvaluationResult objects with error information
        """
        return [
            ResultBuilder.create_failed_result(image_input, text_prompt, error_message, error_type)
            for image_input, text_prompt in zip(image_inputs, text_prompts, strict=False)
        ]
