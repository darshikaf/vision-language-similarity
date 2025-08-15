import pytest
from fastapi import status
from fastapi.testclient import TestClient

from service.core.exceptions import (
    ServiceError,
    ValidationError,
    ImageProcessingError,
    NetworkError,
    ModelError,
)
from service.main import app


@pytest.fixture
def test_client():
    """Create test client for API testing"""
    return TestClient(app)


@pytest.fixture
def sample_service_error():
    """Sample ServiceError for testing"""
    return ServiceError("Test service error", 503)


@pytest.fixture
def sample_validation_error():
    """Sample ValidationError for testing"""
    return ValidationError("Test validation error")


@pytest.fixture
def sample_image_processing_error():
    """Sample ImageProcessingError for testing"""
    return ImageProcessingError("Test image processing error")


@pytest.fixture
def sample_network_error():
    """Sample NetworkError for testing"""
    return NetworkError("Test network error")


@pytest.fixture
def sample_model_error():
    """Sample ModelError for testing"""
    return ModelError("Test model error")


class TestServiceExceptionHierarchy:
    """Test service exception hierarchy and inheritance"""

    def test_service_error_base_class(self, sample_service_error):
        """Test ServiceError base class properties"""
        assert sample_service_error.http_status == 503
        assert sample_service_error.error_type == "service_error"
        assert str(sample_service_error) == "Test service error"
        assert isinstance(sample_service_error, Exception)

    def test_service_error_default_status(self):
        """Test ServiceError default HTTP status"""
        error = ServiceError("Default status error")
        assert error.http_status == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert error.error_type == "service_error"

    @pytest.mark.parametrize("exception_class,expected_type,expected_status", [
        (ValidationError, "validation_error", status.HTTP_400_BAD_REQUEST),
        (ImageProcessingError, "image_processing_error", status.HTTP_400_BAD_REQUEST),
        (NetworkError, "network_error", status.HTTP_400_BAD_REQUEST),
        (ModelError, "model_error", status.HTTP_500_INTERNAL_SERVER_ERROR),
    ])
    def test_exception_properties(self, exception_class, expected_type, expected_status):
        """Test each exception has correct error type and HTTP status"""
        error = exception_class("Test error")
        assert error.error_type == expected_type
        assert error.http_status == expected_status
        assert isinstance(error, ServiceError)


class TestAPIExceptionHandling:
    """Test exception handling through FastAPI endpoints"""

    def test_validation_error_api_response(self, test_client):
        """Test ValidationError returns proper API response"""
        response = test_client.post("/evaluator/v1/evaluation/batch", json={"evaluations": []})
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_image_processing_error_in_evaluation(self, test_client):
        """Test ImageProcessingError handling in evaluation endpoints"""
        response = test_client.post("/evaluator/v1/evaluation/single", json={
            "image_input": "non_existent_file.jpg",
            "text_prompt": "A test image"
        })
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert result["error"] is not None
        assert result["clip_score"] == 0.0





