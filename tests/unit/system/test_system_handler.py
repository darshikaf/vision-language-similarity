from unittest.mock import patch, Mock
import pytest

from service.system.handler import get_model_info, get_system_status
from service.core.config import CLIPModelSpec


@pytest.fixture
def mock_model_spec():
    """Mock model specification for testing"""
    return CLIPModelSpec(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        description="Test model",
        memory_gb=2.0,
        avg_inference_time_ms=100.0,
        accuracy_score=0.85,
        enabled=True
    )


class TestGetModelInfo:
    """Test get_model_info function"""

    @patch('service.system.handler.model_registry')
    def test_get_model_info_returns_correct_structure(self, mock_registry, mock_model_spec):
        """Test that get_model_info returns the expected structure"""
        mock_registry.get_model_spec.return_value = mock_model_spec
        
        result = get_model_info("test_config")
        
        # Check main structure
        assert "config_name" in result
        assert "spec" in result
        assert "loaded" in result
        assert "health_status" in result
        assert "runtime_info" in result
        
        # Check config name
        assert result["config_name"] == "test_config"
        
        # Check spec structure
        spec = result["spec"]
        expected_spec_keys = [
            "model_name", "pretrained", "description",
            "memory_gb", "avg_inference_time_ms", "accuracy_score", "enabled"
        ]
        for key in expected_spec_keys:
            assert key in spec
        
        # Check spec values
        assert spec["model_name"] == "ViT-B-32"
        assert spec["pretrained"] == "laion2b_s34b_b79k"
        assert spec["description"] == "Test model"
        assert spec["memory_gb"] == 2.0
        assert spec["avg_inference_time_ms"] == 100.0
        assert spec["accuracy_score"] == 0.85
        assert spec["enabled"] is True
        
        # Check default values for simplified approach
        assert result["loaded"] is False
        assert result["health_status"] == {"healthy": False}
        assert result["runtime_info"] == {}
        
        # Verify registry was called correctly
        mock_registry.get_model_spec.assert_called_once_with("test_config")

    @patch('service.system.handler.model_registry')
    def test_get_model_info_with_different_configs(self, mock_registry):
        """Test get_model_info with different model configurations"""
        configs = ["fast", "accurate", "custom_config"]
        
        for config in configs:
            mock_spec = CLIPModelSpec(
                model_name=f"Model-{config}",
                pretrained=f"pretrained-{config}",
                description=f"Description for {config}",
                memory_gb=1.0,
                avg_inference_time_ms=50.0,
                accuracy_score=0.9,
                enabled=True
            )
            mock_registry.get_model_spec.return_value = mock_spec
            
            result = get_model_info(config)
            
            assert result["config_name"] == config
            assert result["spec"]["model_name"] == f"Model-{config}"
            assert result["spec"]["pretrained"] == f"pretrained-{config}"
            assert result["spec"]["description"] == f"Description for {config}"


class TestGetSystemStatus:
    """Test get_system_status function"""

    @patch('service.system.handler.model_registry')
    def test_get_system_status_returns_correct_structure(self, mock_registry):
        """Test that get_system_status returns the expected structure"""
        mock_available_models = {
            "fast": {"model_name": "ViT-B-32"},
            "accurate": {"model_name": "ViT-L-14"}
        }
        mock_registry.list_available_models.return_value = mock_available_models
        
        result = get_system_status()
        
        # Check structure
        assert "cached_models" in result
        assert "loaded_models" in result
        assert "available_configs" in result
        
        # Check values for simplified approach
        assert result["cached_models"] == []
        assert result["loaded_models"] == []
        assert result["available_configs"] == ["fast", "accurate"]
        
        # Verify registry was called
        mock_registry.list_available_models.assert_called_once()

    @patch('service.system.handler.model_registry')
    def test_get_system_status_with_empty_models(self, mock_registry):
        """Test get_system_status when no models are available"""
        mock_registry.list_available_models.return_value = {}
        
        result = get_system_status()
        
        assert result["available_configs"] == []

    @patch('service.system.handler.model_registry')
    def test_get_system_status_with_multiple_models(self, mock_registry):
        """Test get_system_status with multiple available models"""
        mock_available_models = {
            "fast": {"model_name": "ViT-B-32"},
            "accurate": {"model_name": "ViT-L-14"},
            "custom1": {"model_name": "Custom-1"},
            "custom2": {"model_name": "Custom-2"}
        }
        mock_registry.list_available_models.return_value = mock_available_models
        
        result = get_system_status()
        
        expected_configs = ["fast", "accurate", "custom1", "custom2"]
        assert sorted(result["available_configs"]) == sorted(expected_configs)


class TestSystemHandlerIntegration:
    """Integration tests for system handler functions"""

    @patch('service.system.handler.model_registry')
    def test_handler_functions_use_same_registry(self, mock_registry):
        """Test that both handler functions use the same registry"""
        mock_spec = CLIPModelSpec(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            description="Test model",
            memory_gb=2.0,
            avg_inference_time_ms=100.0,
            accuracy_score=0.85,
            enabled=True
        )
        mock_available_models = {"test_config": {"model_name": "ViT-B-32"}}
        
        mock_registry.get_model_spec.return_value = mock_spec
        mock_registry.list_available_models.return_value = mock_available_models
        
        # Call both functions
        model_info = get_model_info("test_config")
        system_status = get_system_status()
        
        # Both should have used the registry
        mock_registry.get_model_spec.assert_called_once_with("test_config")
        mock_registry.list_available_models.assert_called_once()
        
        # Results should be consistent
        assert model_info["config_name"] == "test_config"
        assert "test_config" in system_status["available_configs"]

    def test_simplified_approach_consistency(self):
        """Test that the simplified approach is consistent across functions"""
        # Mock registry calls
        with patch('service.system.handler.model_registry') as mock_registry:
            mock_spec = CLIPModelSpec(
                model_name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                description="Test model",
                memory_gb=2.0,
                avg_inference_time_ms=100.0,
                accuracy_score=0.85,
                enabled=True
            )
            mock_available_models = {"test_config": {"model_name": "ViT-B-32"}}
            
            mock_registry.get_model_spec.return_value = mock_spec
            mock_registry.list_available_models.return_value = mock_available_models
            
            model_info = get_model_info("test_config")
            system_status = get_system_status()
            
            # Both should indicate no caching/loading in simplified approach
            assert model_info["loaded"] is False
            assert model_info["health_status"]["healthy"] is False
            assert model_info["runtime_info"] == {}
            
            assert system_status["cached_models"] == []
            assert system_status["loaded_models"] == []