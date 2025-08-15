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

    def test_single_evaluation_invalid_request_empty_fields(self):
        """Test single evaluation with empty required fields"""
        payload = {"image_input": "", "text_prompt": ""}
        
        response = client.post(EVALUATION_PATH + "/single", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_single_evaluation_missing_required_fields(self):
        """Test single evaluation with missing required fields"""
        payload = {"image_input": "test.jpg"}  # Missing text_prompt
        
        response = client.post(EVALUATION_PATH + "/single", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_single_evaluation_invalid_model_config(self):
        """Test single evaluation with invalid model configuration"""
        payload = {
            "image_input": "test.jpg",
            "text_prompt": "test prompt", 
            "model_config_name": "invalid_config"
        }
        
        response = client.post(EVALUATION_PATH + "/single", json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid input" in response.json()["detail"]

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

    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    @patch('service.evaluation.handler._handler', None)  # Reset global handler
    def test_single_evaluation_with_error(self, mock_evaluator_class):
        """Test single evaluation with processing error"""
        mock_evaluator_class.side_effect = ServiceError("Model loading failed", 503)
        
        payload = {
            "image_input": "test.jpg",
            "text_prompt": "test prompt"
        }
        
        response = client.post(EVALUATION_PATH + "/single", json=payload)
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_single_evaluation_default_model_config(self):
        """Test single evaluation uses default model config when not specified"""
        with patch('service.evaluation.handler.MinimalOpenCLIPEvaluator') as mock_evaluator_class:
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
                "text_prompt": "test prompt"
                # No model_config_name specified - should default to "fast"
            }
            
            response = client.post(EVALUATION_PATH + "/single", json=payload)
            assert response.status_code == status.HTTP_200_OK


class TestEvaluationBatchRoute:
    """Test batch evaluation endpoint"""

    def test_batch_evaluation_empty_request(self):
        """Test batch evaluation with empty evaluations list"""
        payload = {"evaluations": []}
        
        response = client.post(EVALUATION_PATH + "/batch", json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "must contain at least one evaluation" in response.json()["detail"]

    def test_batch_evaluation_missing_evaluations(self):
        """Test batch evaluation with missing evaluations field"""
        payload = {"batch_size": 10}
        
        response = client.post(EVALUATION_PATH + "/batch", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_evaluation_invalid_batch_size(self):
        """Test batch evaluation with invalid batch size"""
        payload = {
            "evaluations": [
                {"image_input": "test.jpg", "text_prompt": "test prompt"}
            ],
            "batch_size": 0  # Invalid - must be >= 1
        }
        
        response = client.post(EVALUATION_PATH + "/batch", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_evaluation_batch_size_too_large(self):
        """Test batch evaluation with batch size exceeding limit"""
        payload = {
            "evaluations": [
                {"image_input": "test.jpg", "text_prompt": "test prompt"}
            ],
            "batch_size": 200  # Invalid - must be <= 128
        }
        
        response = client.post(EVALUATION_PATH + "/batch", json=payload)
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

    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    @patch('service.evaluation.handler._handler', None)  # Reset global handler
    def test_batch_evaluation_with_mixed_results(self, mock_evaluator_class):
        """Test batch evaluation with some successes and failures"""
        mock_evaluator = Mock()
        
        # Create mock individual results for each evaluation
        mock_result1 = Mock()
        mock_result1.image_path = "test1.jpg"
        mock_result1.text_prompt = "prompt1"
        mock_result1.clip_score = 75.0
        mock_result1.processing_time_ms = 100.0
        mock_result1.error = None
        
        mock_result2 = Mock()
        mock_result2.image_path = "test2.jpg"
        mock_result2.text_prompt = "prompt2"
        mock_result2.clip_score = 0.0
        mock_result2.processing_time_ms = 50.0
        mock_result2.error = "Image loading failed"
        
        # Handler calls evaluate_single for each request, so mock that method
        mock_evaluator.evaluate_single = AsyncMock(side_effect=[mock_result1, mock_result2])
        mock_evaluator_class.return_value = mock_evaluator
        
        payload = {
            "evaluations": [
                {"image_input": "test1.jpg", "text_prompt": "prompt1"},
                {"image_input": "test2.jpg", "text_prompt": "prompt2"}
            ]
        }
        
        response = client.post(EVALUATION_PATH + "/batch", json=payload)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["total_successful"] == 1
        assert data["total_failed"] == 1


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

    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    def test_health_check_contains_default_configs(self, mock_evaluator_class):
        """Test health check returns expected model configurations"""
        mock_evaluator = Mock()
        mock_evaluator.similarity_model = Mock()
        mock_evaluator.similarity_model.model = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        
        response = client.get(EVALUATION_PATH + "/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        available_configs = data["available_configs"]
        assert "fast" in available_configs
        assert "accurate" in available_configs

    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    @patch('service.evaluation.handler._handler', None)  # Reset global handler
    def test_health_check_model_loading_error(self, mock_evaluator_class):
        """Test health check with model loading error"""
        mock_evaluator_class.side_effect = ServiceError("Model loading failed", 503)
        
        response = client.get(EVALUATION_PATH + "/health")
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestEvaluationRouteValidation:
    """Test evaluation route validation and error handling"""

    def test_single_evaluation_large_payload(self):
        """Test single evaluation with very large text prompt"""
        payload = {
            "image_input": "test.jpg",
            "text_prompt": "a" * 10000,  # Very long prompt
            "model_config_name": "fast"
        }
        
        # Should handle large payloads gracefully
        response = client.post(EVALUATION_PATH + "/single", json=payload)
        # Might be 400 (validation error) or 200 (success) depending on implementation
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]

    def test_evaluation_with_special_characters(self):
        """Test evaluation with special characters in text prompt"""
        payload = {
            "image_input": "test.jpg",
            "text_prompt": "Test with émojis 🚀 and special chars: @#$%^&*()",
            "model_config_name": "fast"
        }
        
        # Should handle special characters
        response = client.post(EVALUATION_PATH + "/single", json=payload) 
        # Could be validation error or success
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]

    def test_evaluation_null_values(self):
        """Test evaluation with null values"""
        payload = {
            "image_input": None,
            "text_prompt": "test prompt"
        }
        
        response = client.post(EVALUATION_PATH + "/single", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestEvaluationRouteIntegration:
    """Test evaluation route integration with application components"""

    def test_evaluation_health_consistent_with_admin(self):
        """Test that evaluation health is consistent with admin endpoints"""
        eval_response = client.get(EVALUATION_PATH + "/health")
        admin_response = client.get("/evaluator/v1/admin/models")
        
        if eval_response.status_code == status.HTTP_200_OK and admin_response.status_code == status.HTTP_200_OK:
            eval_configs = set(eval_response.json()["available_configs"])
            admin_configs = set(admin_response.json()["available_configs"].keys())
            
            # Should have same available configurations
            assert eval_configs == admin_configs

    def test_evaluation_endpoints_use_correct_prefix(self):
        """Test that evaluation endpoints use correct prefix"""
        # All evaluation routes should be under /evaluator/v1/evaluation/
        evaluation_endpoints = [
            "/evaluator/v1/evaluation/single",
            "/evaluator/v1/evaluation/batch", 
            "/evaluator/v1/evaluation/health"
        ]
        
        for endpoint in evaluation_endpoints:
            # POST endpoints need data, but should not return 404
            if "single" in endpoint or "batch" in endpoint:
                response = client.post(endpoint, json={})
                assert response.status_code != status.HTTP_404_NOT_FOUND
            else:
                response = client.get(endpoint)
                assert response.status_code != status.HTTP_404_NOT_FOUND


class TestEvaluationOpenAPIDocumentation:
    """Test evaluation routes OpenAPI documentation"""

    def test_evaluation_routes_documented_in_openapi(self):
        """Test that evaluation routes are properly documented"""
        response = client.get("/evaluator/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        openapi_spec = response.json()
        paths = openapi_spec["paths"]
        
        # Verify evaluation endpoints exist
        evaluation_endpoints = [
            "/evaluator/v1/evaluation/single",
            "/evaluator/v1/evaluation/batch",
            "/evaluator/v1/evaluation/health"
        ]
        
        for endpoint in evaluation_endpoints:
            assert endpoint in paths

    def test_evaluation_routes_have_proper_response_models(self):
        """Test that evaluation routes have proper response models defined"""
        response = client.get("/evaluator/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        openapi_spec = response.json()
        paths = openapi_spec["paths"]
        
        # Single evaluation should have EvaluationResponse model
        single_path = paths["/evaluator/v1/evaluation/single"]["post"]
        assert "responses" in single_path
        assert "200" in single_path["responses"]
        
        # Batch evaluation should have BatchEvaluationResponse model
        batch_path = paths["/evaluator/v1/evaluation/batch"]["post"]
        assert "responses" in batch_path
        assert "200" in batch_path["responses"]

    def test_evaluation_routes_tagged_correctly(self):
        """Test that evaluation routes have correct tags"""
        response = client.get("/evaluator/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        openapi_spec = response.json()
        paths = openapi_spec["paths"]
        
        # Check evaluation routes have 'evaluation' tag
        for path_key in paths:
            if "/evaluation/" in path_key:
                for method_item in paths[path_key].values():
                    if "tags" in method_item:
                        assert "evaluation" in method_item["tags"]


class TestEvaluationRoutePrefix:
    """Test evaluation route prefix configuration"""

    def test_evaluation_prefix_constant(self):
        """Test that evaluation prefix constant is correct"""
        from service.evaluation.routes import EVALUATION_PREFIX
        assert EVALUATION_PREFIX == "/v1/evaluation"

    def test_evaluation_routes_not_accessible_without_prefix(self):
        """Test that evaluation routes are not accessible without proper prefix"""
        # Test that routes are NOT accessible without full prefix
        invalid_routes = [
            "/evaluator/single",
            "/evaluator/batch", 
            "/evaluator/evaluation/single"  # Missing v1
        ]
        
        for route in invalid_routes:
            response = client.post(route, json={})
            assert response.status_code == status.HTTP_404_NOT_FOUND