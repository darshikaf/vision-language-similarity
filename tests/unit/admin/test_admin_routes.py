from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import pytest

from service.main import app
from service.admin.routes import ADMIN_PREFIX
from service.main import PATH_PREFIX

client = TestClient(app)
ADMIN_PATH = PATH_PREFIX + ADMIN_PREFIX


class TestAdminModelRoutes:
    """Test admin model management endpoints"""

    def test_get_available_models_success(self):
        """Test successful retrieval of available models"""
        response = client.get(ADMIN_PATH + "/models")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "available_configs" in data
        assert "default_config" in data
        assert "total_available" in data
        assert data["default_config"] == "fast"
        assert data["total_available"] >= 2

    def test_get_model_specifications_success(self):
        """Test successful retrieval of detailed model specifications"""
        response = client.get(ADMIN_PATH + "/models/specs")
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
        response = client.get(ADMIN_PATH + "/models/all")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "all_models" in data
        assert "available_models" in data
        
        # Available should be subset of all
        all_models = set(data["all_models"].keys())
        available_models = set(data["available_models"].keys())
        assert available_models.issubset(all_models)

    @patch('service.admin.routes.model_manager')
    def test_get_model_info_success(self, mock_manager):
        """Test successful retrieval of model runtime information"""
        mock_info = {
            "config_name": "fast",
            "status": "healthy",
            "memory_usage": "2.1GB",
            "last_used": "2024-01-15T10:30:00Z"
        }
        mock_manager.get_model_info = AsyncMock(return_value=mock_info)
        
        response = client.get(ADMIN_PATH + "/models/fast/info")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == mock_info

    @patch('service.admin.routes.model_manager')
    def test_get_model_info_invalid_config(self, mock_manager):
        """Test model info with invalid configuration name"""
        mock_manager.get_model_info.side_effect = ValueError("Unknown model config: invalid")
        
        response = client.get(ADMIN_PATH + "/models/invalid/info")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid input" in response.json()["detail"]

    @patch('service.admin.routes.model_manager')
    def test_get_model_manager_status_success(self, mock_manager):
        """Test successful retrieval of model manager status"""
        mock_status = {
            "manager_status": "healthy",
            "loaded_models": ["fast", "accurate"],
            "total_memory_usage": "4.2GB"
        }
        mock_manager.get_status.return_value = mock_status
        
        response = client.get(ADMIN_PATH + "/status")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == mock_status


class TestAdminRouteValidation:
    """Test admin route validation and error handling"""

    def test_model_info_with_special_characters(self):
        """Test model info endpoint with special characters in config name"""
        response = client.get(ADMIN_PATH + "/models/test@config/info")
        # Should handle gracefully, likely return 400 or 404
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND]

    def test_model_info_with_empty_config_name(self):
        """Test model info endpoint with empty config name"""
        response = client.get(ADMIN_PATH + "/models//info")
        # FastAPI should return 404 for empty path parameter
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch('service.admin.routes.model_manager')
    def test_file_not_found_error_handling(self, mock_manager):
        """Test handling of FileNotFoundError"""
        mock_manager.get_model_info.side_effect = FileNotFoundError("Model file not found")
        
        response = client.get(ADMIN_PATH + "/models/test/info")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Resource not found" in response.json()["detail"]

    @patch('service.admin.routes.model_manager')
    def test_generic_exception_handling(self, mock_manager):
        """Test handling of generic exceptions"""
        mock_manager.get_model_info.side_effect = RuntimeError("Unexpected error")
        
        response = client.get(ADMIN_PATH + "/models/test/info")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Internal server error" in response.json()["detail"]


class TestAdminRouteIntegration:
    """Test admin routes integration with real components"""

    def test_default_models_are_available(self):
        """Test that default models are properly configured and available"""
        response = client.get(ADMIN_PATH + "/models")
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
        response = client.get(ADMIN_PATH + "/models/specs")
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
        response = client.get(ADMIN_PATH + "/models/all")
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


class TestAdminOpenAPIDocumentation:
    """Test admin routes OpenAPI documentation"""

    def test_admin_routes_documented_in_openapi(self):
        """Test that admin routes are properly documented"""
        response = client.get("/evaluator/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        openapi_spec = response.json()
        paths = openapi_spec["paths"]
        
        # Verify admin endpoints exist
        admin_endpoints = [
            "/evaluator/v1/admin/models",
            "/evaluator/v1/admin/models/specs",
            "/evaluator/v1/admin/models/all",
            "/evaluator/v1/admin/models/{config_name}/info",
            "/evaluator/v1/admin/status"
        ]
        
        for endpoint in admin_endpoints:
            assert endpoint in paths

    def test_admin_routes_tagged_correctly(self):
        """Test that admin routes have correct tags"""
        response = client.get("/evaluator/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        openapi_spec = response.json()
        paths = openapi_spec["paths"]
        
        # Check admin routes have 'admin' tag
        for path_key, path_item in paths.items():
            if "/admin/" in path_key:
                for method_item in path_item.values():
                    if "tags" in method_item:
                        assert "admin" in method_item["tags"]

    def test_admin_routes_have_descriptions(self):
        """Test that admin routes have proper descriptions"""
        response = client.get("/evaluator/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        openapi_spec = response.json()
        paths = openapi_spec["paths"]
        
        admin_paths = {k: v for k, v in paths.items() if "/admin/" in k}
        
        for path_key, path_item in admin_paths.items():
            for method_key, method_item in path_item.items():
                # Each endpoint should have summary and description
                assert "summary" in method_item
                assert "description" in method_item
                assert len(method_item["summary"]) > 0
                assert len(method_item["description"]) > 0


class TestAdminRoutePrefix:
    """Test admin route prefix configuration"""

    def test_admin_prefix_constant(self):
        """Test that admin prefix constant is correct"""
        from service.admin.routes import ADMIN_PREFIX
        assert ADMIN_PREFIX == "/v1/admin"

    def test_admin_routes_accessible_via_prefix(self):
        """Test that admin routes are accessible via the correct prefix"""
        # Test basic admin route accessibility
        response = client.get("/evaluator/v1/admin/models")
        assert response.status_code == status.HTTP_200_OK
        
        # Test that routes are NOT accessible without prefix
        response = client.get("/evaluator/models")
        assert response.status_code == status.HTTP_404_NOT_FOUND
