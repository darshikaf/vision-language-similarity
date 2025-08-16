from service.core.exceptions import ServiceError
from service.core.observability import get_metrics_middleware
from service.log import get_logger

logger = get_logger(__name__)


class MetricsRecorder:
    """
    Handles metrics recording and observability concerns for ML evaluations.

    Responsibilities:
    - Record successful evaluation metrics (inference time, CLIP scores, processing time)
    - Record error metrics and patterns
    - Extract error types from exceptions
    - Provide optional/pluggable metrics recording
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize metrics recorder.

        Args:
            enabled: Whether metrics recording is enabled
        """
        self.enabled = enabled

    async def record_success_metrics(
        self,
        clip_score: float,
        inference_time: float,
        total_time: float,
        source_type: str,
        model_config_name: str,
        device_type: str,
        model_name: str,
    ) -> None:
        """
        Record metrics for successful evaluation.

        Args:
            clip_score: Computed CLIP similarity score
            inference_time: Model inference time in milliseconds
            total_time: Total processing time in milliseconds
            source_type: Type of image source (url, file, etc.)
            model_config_name: Model configuration name
            device_type: Device type (cpu, cuda, mps)
            model_name: Actual model name
        """
        if not self.enabled:
            return

        try:
            metrics = get_metrics_middleware()
            if not metrics:
                return

            # Record inference time in seconds
            metrics.record_inference_time(
                inference_time / 1000,
                model_config_name,
                device_type,
                model_name,
            )

            # Record CLIP score
            metrics.record_clip_score(clip_score, model_config_name, model_name)

            # Record image processing time
            image_processing_time = (total_time - inference_time) / 1000
            metrics.record_image_processing_time(image_processing_time, source_type)

        except Exception as e:
            logger.debug(f"Metrics recording failed: {e}")

    def record_error_metrics(
        self, exception: Exception, model_config_name: str, model_name: str, error_type: str | None = None
    ) -> None:
        """
        Record metrics for evaluation errors.

        Args:
            exception: The exception that occurred
            model_config_name: Model configuration name
            model_name: Actual model name
            error_type: Optional specific error type
        """
        if not self.enabled:
            return

        try:
            metrics = get_metrics_middleware()
            if not metrics:
                return

            error_name = error_type or type(exception).__name__
            metrics.record_model_error(error_name, model_config_name, model_name)
            metrics.record_error_pattern(error_name, "evaluation", model_config_name)

        except Exception as metrics_exception:
            logger.debug(f"Metrics recording failed: {metrics_exception}")

    def extract_error_type(self, exception: Exception) -> str | None:
        """
        Extract error type from exception for consistent metrics.

        Args:
            exception: The exception to analyze

        Returns:
            Error type string if available, None otherwise
        """
        return getattr(exception, "error_type", None) if isinstance(exception, ServiceError) else None
