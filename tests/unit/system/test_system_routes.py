from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import pytest

from service.main import app
from service.system.routes import SYSTEM_PREFIX
from service.main import PATH_PREFIX

client = TestClient(app)
SYSTEM_PATH = PATH_PREFIX + SYSTEM_PREFIX


class TestSystemModelRoutes:
    """Test system model management endpoints"""

    def test_get_available_models_success(self):
        """Test successful retrieval of available models"""
        response = client.get(SYSTEM_PATH + "/models")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "available_configs" in data
        assert "default_config" in data
        assert "total_available" in data
        assert data["default_config"] == "fast"
        assert data["total_available"] >= 2

    def test_get_model_specifications_success(self):
        """Test successful retrieval of detailed model specifications"""
        response = client.get(SYSTEM_PATH + "/models/specs")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert isinstance(data, dict)
        assert "fast" in data
        assert "accurate" in data
        
        # Verify model spec structure
        fast_spec = data["fast"]
        required_fields = ["model_name", "pretrained", "enabled", "description", "memory_gb"]
        for field in required_fields:
            assert field in fast_spec

    def test_get_all_model_configs_success(self):
        """Test successful retrieval of all model configurations"""
        response = client.get(SYSTEM_PATH + "/models/all")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "all_models" in data
        assert "available_models" in data
        
        # Available should be subset of all
        all_models = set(data["all_models"].keys())
        available_models = set(data["available_models"].keys())
        assert available_models.issubset(all_models)

    @patch('service.system.routes.get_model_info')
    def test_get_model_info_success(self, mock_get_model_info):
        """Test successful retrieval of model runtime information"""
        mock_info = {
            "config_name": "fast",
            "spec": {
                "model_name": "ViT-B-32",
                "pretrained": "laion2b_s34b_b79k",
                "description": "Fast model",
                "memory_gb": 2.0,
                "avg_inference_time_ms": 100.0,
                "accuracy_score": 0.85,
                "enabled": True
            },
            "loaded": False,
            "health_status": {"healthy": False},
            "runtime_info": {}
        }
        mock_get_model_info.return_value = mock_info
        
        response = client.get(SYSTEM_PATH + "/models/fast/info")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == mock_info

    @patch('service.system.routes.get_model_info')
    def test_get_model_info_invalid_config(self, mock_get_model_info):
        """Test model info with invalid configuration name"""
        mock_get_model_info.side_effect = ValueError("Unknown model config: invalid")
        
        response = client.get(SYSTEM_PATH + "/models/invalid/info")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid input" in response.json()["detail"]

    @patch('service.system.routes.get_system_status')
    def test_get_model_manager_status_success(self, mock_get_system_status):
        """Test successful retrieval of system status"""
        mock_status = {
            "cached_models": [],
            "loaded_models": [],
            "available_configs": ["fast", "accurate"]
        }
        mock_get_system_status.return_value = mock_status
        
        response = client.get(SYSTEM_PATH + "/status")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == mock_status


class TestSystemRouteValidation:
    """Test system route validation and error handling"""

    def test_model_info_with_special_characters(self):
        """Test model info endpoint with special characters in config name"""
        response = client.get(SYSTEM_PATH + "/models/test@config/info")
        # Should handle gracefully, likely return 400 or 404
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND]

    def test_model_info_with_empty_config_name(self):
        """Test model info endpoint with empty config name"""
        response = client.get(SYSTEM_PATH + "/models//info")
        # FastAPI should return 404 for empty path parameter
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch('service.system.routes.get_model_info')
    def test_file_not_found_error_handling(self, mock_get_model_info):
        """Test handling of FileNotFoundError"""
        mock_get_model_info.side_effect = FileNotFoundError("Model file not found")
        
        response = client.get(SYSTEM_PATH + "/models/test/info")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Resource not found" in response.json()["detail"]

    @patch('service.system.routes.get_model_info')
    def test_generic_exception_handling(self, mock_get_model_info):
        """Test handling of generic exceptions"""
        mock_get_model_info.side_effect = RuntimeError("Unexpected error")
        
        response = client.get(SYSTEM_PATH + "/models/test/info")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Internal server error" in response.json()["detail"]


class TestSystemRouteIntegration:
    """Test system routes integration with real components"""

    def test_default_models_are_available(self):
        """Test that default models are properly configured and available"""
        response = client.get(SYSTEM_PATH + "/models")
        assert response.status_code == status.HTTP_200_OK
        
        available_configs = response.json()["available_configs"]
        
        # Verify fast model
        assert "fast" in available_configs
        fast_config = available_configs["fast"]
        assert fast_config["model_name"] == "ViT-B-32"
        assert fast_config["enabled"] is True
        
        # Verify accurate model
        assert "accurate" in available_configs
        accurate_config = available_configs["accurate"]
        assert accurate_config["model_name"] == "ViT-L-14"
        assert accurate_config["enabled"] is True

    def test_model_specs_contain_performance_metrics(self):
        """Test that model specifications include performance metrics"""
        response = client.get(SYSTEM_PATH + "/models/specs")
        assert response.status_code == status.HTTP_200_OK
        
        specs = response.json()
        for model_name, spec in specs.items():
            # Each spec should have performance information
            assert "memory_gb" in spec
            assert "avg_inference_time_ms" in spec
            assert "accuracy_score" in spec
            assert isinstance(spec["memory_gb"], (int, float))
            assert isinstance(spec["avg_inference_time_ms"], (int, float))
            assert isinstance(spec["accuracy_score"], (int, float))

    def test_all_models_vs_available_models_consistency(self):
        """Test consistency between all models and available models endpoints"""
        response = client.get(SYSTEM_PATH + "/models/all")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        all_models = data["all_models"]
        available_models = data["available_models"]
        
        # Every available model should exist in all models
        for model_name in available_models:
            assert model_name in all_models
            
        # Available models should only include enabled models
        for model_name, model_spec in available_models.items():
            assert model_spec["enabled"] is True


