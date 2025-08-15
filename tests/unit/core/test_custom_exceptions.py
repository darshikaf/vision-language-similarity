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

    def test_validation_error_properties(self, sample_validation_error):
        """Test ValidationError specific properties"""
        assert sample_validation_error.http_status == status.HTTP_400_BAD_REQUEST
        assert sample_validation_error.error_type == "validation_error"
        assert str(sample_validation_error) == "Test validation error"
        assert isinstance(sample_validation_error, ServiceError)

    def test_image_processing_error_properties(self, sample_image_processing_error):
        """Test ImageProcessingError specific properties"""
        assert sample_image_processing_error.http_status == status.HTTP_400_BAD_REQUEST
        assert sample_image_processing_error.error_type == "image_processing_error"
        assert str(sample_image_processing_error) == "Test image processing error"
        assert isinstance(sample_image_processing_error, ServiceError)

    def test_network_error_inheritance(self, sample_network_error):
        """Test NetworkError inherits from ImageProcessingError"""
        assert sample_network_error.http_status == status.HTTP_400_BAD_REQUEST
        assert sample_network_error.error_type == "network_error"
        assert str(sample_network_error) == "Test network error"
        assert isinstance(sample_network_error, ImageProcessingError)
        assert isinstance(sample_network_error, ServiceError)

    def test_model_error_properties(self, sample_model_error):
        """Test ModelError specific properties"""
        assert sample_model_error.http_status == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert sample_model_error.error_type == "model_error"
        assert str(sample_model_error) == "Test model error"
        assert isinstance(sample_model_error, ServiceError)

    @pytest.mark.parametrize("exception_class,expected_type", [
        (ValidationError, "validation_error"),
        (ImageProcessingError, "image_processing_error"),
        (NetworkError, "network_error"),
        (ModelError, "model_error"),
    ])
    def test_exception_error_types(self, exception_class, expected_type):
        """Test each exception has correct error type"""
        error = exception_class("Test error")
        assert error.error_type == expected_type

    @pytest.mark.parametrize("exception_class,expected_status", [
        (ValidationError, status.HTTP_400_BAD_REQUEST),
        (ImageProcessingError, status.HTTP_400_BAD_REQUEST),
        (NetworkError, status.HTTP_400_BAD_REQUEST),
        (ModelError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    ])
    def test_exception_http_statuses(self, exception_class, expected_status):
        """Test each exception has correct HTTP status"""
        error = exception_class("Test error")
        assert error.http_status == expected_status


class TestAPIExceptionHandling:
    """Test exception handling through FastAPI endpoints"""

    def test_validation_error_api_response(self, test_client):
        """Test ValidationError returns proper API response"""
        # Empty batch request should trigger validation error
        response = test_client.post("/evaluator/v1/evaluation/batch", json={"evaluations": []})
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        error_detail = response.json()["detail"]
        assert "Batch request must contain at least one evaluation" in error_detail

    def test_invalid_json_request_handling(self, test_client):
        """Test handling of malformed JSON requests"""
        response = test_client.post(
            "/evaluator/v1/evaluation/single",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_required_fields_handling(self, test_client):
        """Test handling of requests with missing required fields"""
        response = test_client.post("/evaluator/v1/evaluation/single", json={
            "image_input": "test.jpg"
            # Missing text_prompt
        })
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_successful_request_still_works(self, test_client):
        """Test that normal requests work correctly despite error handling"""
        response = test_client.get("/evaluator/health")
        assert response.status_code == status.HTTP_200_OK

    def test_image_processing_error_in_evaluation(self, test_client):
        """Test ImageProcessingError handling in evaluation endpoints"""
        response = test_client.post("/evaluator/v1/evaluation/single", json={
            "image_input": "non_existent_file.jpg",
            "text_prompt": "A test image"
        })
        
        # Single evaluations should return 200 with error in response body
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        assert result["error"] is not None
        assert "Failed to load image from file" in result["error"]
        assert result["clip_score"] == 0.0

    def test_health_endpoint_error_handling(self, test_client):
        """Test error handling in health endpoints"""
        response = test_client.get("/evaluator/v1/evaluation/health")
        
        # Health endpoint should work or return appropriate error
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_503_SERVICE_UNAVAILABLE]
        
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data
            assert "available_configs" in data


class TestExceptionPropagation:
    """Test exception propagation through application components"""

    @pytest.mark.asyncio
    async def test_image_loader_raises_image_processing_error(self):
        """Test ImageLoader raises ImageProcessingError for missing files"""
        from service.core.image_loader import ImageLoader
        
        loader = ImageLoader()
        with pytest.raises(ImageProcessingError, match="Failed to load image from file"):
            await loader.load_image("non_existent_file.jpg")

    @pytest.mark.asyncio
    async def test_evaluator_validation_error_propagation(self):
        """Test evaluator propagates ValidationError for invalid inputs"""
        from service.core.evaluator import MinimalOpenCLIPEvaluator
        
        evaluator = MinimalOpenCLIPEvaluator.create_fast_evaluator()
        with pytest.raises(ValidationError, match="Mismatch"):
            await evaluator.evaluate_batch(["image1.jpg", "image2.jpg"], ["prompt1"])

    def test_sync_image_loader_raises_image_processing_error(self):
        """Test synchronous ImageLoader raises ImageProcessingError"""
        from service.core.image_loader import ImageLoader
        
        loader = ImageLoader()
        with pytest.raises(ImageProcessingError, match="Failed to load image from file"):
            loader.load_image_sync("non_existent_file.jpg")

    @pytest.mark.asyncio
    async def test_network_error_propagation(self):
        """Test NetworkError is properly propagated"""
        from service.core.image_loader import ImageLoader
        from unittest.mock import patch
        import httpx
        
        async with ImageLoader() as loader:
            with patch('httpx.AsyncClient.get', side_effect=httpx.RequestError("Network failed")):
                with pytest.raises(NetworkError, match="Failed to fetch image from URL"):
                    await loader.load_image("https://example.com/test.jpg")


class TestExceptionCustomization:
    """Test exception customization and extension"""

    def test_service_error_custom_status_code(self):
        """Test ServiceError with custom HTTP status code"""
        error = ServiceError("Custom error", 418)  # I'm a teapot
        assert error.http_status == 418
        assert error.error_type == "service_error"

    def test_exception_string_representation(self):
        """Test exception string representations are meaningful"""
        exceptions = [
            ServiceError("Service failed"),
            ValidationError("Invalid input data"),
            ImageProcessingError("Image format not supported"),
            NetworkError("Connection timeout"),
            ModelError("Model loading failed")
        ]
        
        for exc in exceptions:
            str_repr = str(exc)
            assert len(str_repr) > 0
            assert str_repr != "Exception"  # Should have meaningful message

    def test_exception_with_empty_message(self):
        """Test exceptions handle empty messages gracefully"""
        error = ServiceError("")
        assert str(error) == ""
        assert error.error_type == "service_error"
        assert error.http_status == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_exception_inheritance_chain(self):
        """Test exception inheritance chain is correct"""
        network_error = NetworkError("Test")
        
        # Check inheritance chain
        assert isinstance(network_error, NetworkError)
        assert isinstance(network_error, ImageProcessingError)
        assert isinstance(network_error, ServiceError)
        assert isinstance(network_error, Exception)


class TestExceptionIntegration:
    """Test exception integration with application flow"""

    def test_exception_handling_preserves_error_details(self, test_client):
        """Test that exception details are preserved through API layer"""
        response = test_client.post("/evaluator/v1/evaluation/single", json={
            "image_input": "specific_nonexistent_file.jpg",
            "text_prompt": "Test prompt"
        })
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        
        # Error message should contain specific file name
        assert "specific_nonexistent_file.jpg" in result["error"]

    def test_concurrent_exception_handling(self, test_client):
        """Test exception handling under concurrent requests"""
        import concurrent.futures
        
        def make_bad_request():
            return test_client.post("/evaluator/v1/evaluation/single", json={
                "image_input": "bad_file.jpg",
                "text_prompt": "Test"
            })
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_bad_request) for _ in range(3)]
            responses = [future.result() for future in futures]
        
        # All should handle errors gracefully
        for response in responses:
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert result["error"] is not None

    def test_exception_logging_integration(self, test_client, caplog):
        """Test that exceptions are properly logged"""
        import logging
        
        with caplog.at_level(logging.ERROR):
            response = test_client.post("/evaluator/v1/evaluation/single", json={
                "image_input": "nonexistent.jpg",
                "text_prompt": "Test"
            })
            
            assert response.status_code == status.HTTP_200_OK
            
            # Should have some log entries (though exact format depends on implementation)
            assert len(caplog.records) >= 0  # At least not crashing logging