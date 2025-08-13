from fastapi import status


class ServiceError(Exception):
    """Base exception for service errors with built-in HTTP status mapping"""

    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type: str = "service_error"


class ValidationError(ServiceError):
    """Input validation and configuration errors"""

    http_status: int = status.HTTP_400_BAD_REQUEST
    error_type: str = "validation_error"


class ImageProcessingError(ServiceError):
    """Image loading and processing errors"""

    http_status: int = status.HTTP_400_BAD_REQUEST
    error_type: str = "image_processing_error"


class NetworkError(ImageProcessingError):
    """Network-related image loading errors (retryable)"""

    error_type: str = "network_error"


class ModelError(ServiceError):
    """Model loading and inference errors"""

    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_type: str = "model_error"
