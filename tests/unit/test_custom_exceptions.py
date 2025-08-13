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


class TestExceptionHierarchy:
    """Test basic exception hierarchy and attributes"""

    def test_service_error_base_class(self):
        """Test ServiceError base class"""
        error = ServiceError("Base error")
        assert error.http_status == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert error.error_type == "service_error"
        assert str(error) == "Base error"

    def test_validation_error(self):
        """Test ValidationError attributes"""
        error = ValidationError("Invalid input")
        assert error.http_status == status.HTTP_400_BAD_REQUEST
        assert error.error_type == "validation_error"
        assert str(error) == "Invalid input"

    def test_image_processing_error(self):
        """Test ImageProcessingError attributes"""
        error = ImageProcessingError("Image failed")
        assert error.http_status == status.HTTP_400_BAD_REQUEST
        assert error.error_type == "image_processing_error"
        assert str(error) == "Image failed"

    def test_network_error(self):
        """Test NetworkError inherits from ImageProcessingError"""
        error = NetworkError("Network failed")
        assert error.http_status == status.HTTP_400_BAD_REQUEST
        assert error.error_type == "network_error"
        assert isinstance(error, ImageProcessingError)
        assert str(error) == "Network failed"

    def test_model_error(self):
        """Test ModelError attributes"""
        error = ModelError("Model failed")
        assert error.http_status == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert error.error_type == "model_error"
        assert str(error) == "Model failed"


class TestAPIExceptionHandling:
    """Test exception handling through FastAPI endpoints"""

    def test_validation_error_api_mapping(self):
        """Test ValidationError maps to 400 in API"""
        client = TestClient(app)
        
        # Empty batch request triggers ValidationError
        response = client.post("/evaluator/v1/evaluation/batch", json={"evaluations": []})
        
        assert response.status_code == 400
        assert "Batch request must contain at least one evaluation" in response.json()["detail"]

    def test_successful_request_still_works(self):
        """Test normal requests work correctly"""
        client = TestClient(app)
        
        response = client.get("/evaluator/health")
        assert response.status_code == 200

    def test_image_processing_error_in_evaluation(self):
        """Test ImageProcessingError is handled gracefully in evaluation"""
        client = TestClient(app)
        
        response = client.post("/evaluator/v1/evaluation/single", json={
            "image_input": "non_existent_file.jpg",
            "text_prompt": "A test image"
        })
        
        # Individual evaluations return 200 with error in response body
        assert response.status_code == 200
        result = response.json()
        assert result["error"] is not None
        assert "Failed to load image from file" in result["error"]
        assert result["clip_score"] == 0.0


class TestExceptionInComponents:
    """Test exceptions are raised correctly in components"""

    @pytest.mark.asyncio
    async def test_image_loader_file_not_found(self):
        """Test ImageLoader raises ImageProcessingError for missing files"""
        from service.core.image_loader import ImageLoader
        
        loader = ImageLoader()
        with pytest.raises(ImageProcessingError, match="Failed to load image from file"):
            await loader.load_image("non_existent_file.jpg")

    @pytest.mark.asyncio
    async def test_evaluator_batch_validation_error(self):
        """Test evaluator raises ValidationError for mismatched inputs"""
        from service.core.evaluator import MinimalOpenCLIPEvaluator
        
        evaluator = MinimalOpenCLIPEvaluator.create_fast_evaluator()
        with pytest.raises(ValidationError, match="Mismatch"):
            await evaluator.evaluate_batch(["image1.jpg", "image2.jpg"], ["prompt1"])

    def test_openclip_model_validation_error(self):
        """Test OpenCLIP model raises ValidationError for mismatched batch"""
        from service.core.similarity_models.openclip_model import OpenCLIPSimilarityModel
        from PIL import Image
        
        # Create a small test image
        test_image = Image.new('RGB', (10, 10), color='red')
        
        model = OpenCLIPSimilarityModel()
        with pytest.raises(ValidationError, match="Mismatch"):
            # This will be caught during batch processing
            pytest.importorskip("asyncio").run(
                model.compute_batch_similarity([test_image], ["prompt1", "prompt2"])
            )
