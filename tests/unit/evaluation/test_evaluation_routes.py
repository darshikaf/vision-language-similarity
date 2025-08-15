from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, Mock
import pytest

from service.main import app
from service.evaluation.routes import EVALUATION_PREFIX
from service.main import PATH_PREFIX
from service.core.exceptions import ServiceError

client = TestClient(app)
EVALUATION_PATH = PATH_PREFIX + EVALUATION_PREFIX


class TestEvaluationSingleRoute:
    """Test single evaluation endpoint"""

    def test_single_evaluation_validation_errors(self):
        """Test single evaluation validation errors"""
        # Empty fields
        response = client.post(EVALUATION_PATH + "/single", json={"image_input": "", "text_prompt": ""})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Invalid model config
        response = client.post(EVALUATION_PATH + "/single", json={
            "image_input": "test.jpg", "text_prompt": "test prompt", 
            "model_config_name": "invalid_config"
        })
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    def test_single_evaluation_success(self, mock_evaluator_class):
        """Test successful single evaluation"""
        # Mock evaluator response
        mock_evaluator = Mock()
        mock_result = Mock()
        mock_result.image_path = "test.jpg"
        mock_result.text_prompt = "test prompt"
        mock_result.clip_score = 75.5
        mock_result.processing_time_ms = 150.0
        mock_result.error = None
        
        mock_evaluator.evaluate_single = AsyncMock(return_value=mock_result)
        mock_evaluator_class.return_value = mock_evaluator
        
        payload = {
            "image_input": "test.jpg",
            "text_prompt": "test prompt",
            "model_config_name": "fast"
        }
        
        response = client.post(EVALUATION_PATH + "/single", json=payload)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "clip_score" in data
        assert "processing_time_ms" in data
        assert "model_used" in data
        assert data["model_used"] == "fast"



class TestEvaluationBatchRoute:
    """Test batch evaluation endpoint"""

    def test_batch_evaluation_validation_errors(self):
        """Test batch evaluation validation errors"""
        # Empty evaluations
        response = client.post(EVALUATION_PATH + "/batch", json={"evaluations": []})
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        
        # Invalid batch size
        response = client.post(EVALUATION_PATH + "/batch", json={
            "evaluations": [{"image_input": "test.jpg", "text_prompt": "test prompt"}],
            "batch_size": 0
        })
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    def test_batch_evaluation_success(self, mock_evaluator_class):
        """Test successful batch evaluation"""
        mock_evaluator = Mock()
        
        # Mock batch response
        mock_response = Mock()
        mock_response.results = [
            Mock(image_input="test1.jpg", text_prompt="prompt1", clip_score=75.0, processing_time_ms=100.0, error=None, model_used="fast"),
            Mock(image_input="test2.jpg", text_prompt="prompt2", clip_score=82.5, processing_time_ms=110.0, error=None, model_used="fast")
        ]
        mock_response.total_processed = 2
        mock_response.total_successful = 2
        mock_response.total_failed = 0
        mock_response.total_processing_time_ms = 210.0
        
        mock_evaluator.evaluate_batch = AsyncMock(return_value=mock_response)
        mock_evaluator_class.return_value = mock_evaluator
        
        payload = {
            "evaluations": [
                {"image_input": "test1.jpg", "text_prompt": "prompt1"},
                {"image_input": "test2.jpg", "text_prompt": "prompt2"}
            ],
            "batch_size": 10
        }
        
        response = client.post(EVALUATION_PATH + "/batch", json=payload)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "results" in data
        assert "total_processed" in data
        assert "total_successful" in data
        assert "total_failed" in data
        assert data["total_processed"] == 2



class TestEvaluationHealthRoute:
    """Test evaluation health endpoint"""

    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    def test_health_check_success(self, mock_evaluator_class):
        """Test successful health check"""
        mock_evaluator = Mock()
        mock_evaluator.similarity_model = Mock()
        mock_evaluator.similarity_model.model = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        
        response = client.get(EVALUATION_PATH + "/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "available_configs" in data
        assert isinstance(data["available_configs"], list)
        assert len(data["available_configs"]) >= 2






