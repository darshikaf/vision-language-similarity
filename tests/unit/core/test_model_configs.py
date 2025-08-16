import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import pytest

from service.core.config import DynamicModelRegistry, CLIPModelSpec


@pytest.fixture
def registry():
    """Create registry instance for testing"""
    return DynamicModelRegistry()


@pytest.fixture
def sample_configmap_data():
    """Sample ConfigMap data for testing"""
    return {
        "models": {
            "custom_fast": {
                "model_name": "ViT-B-16",
                "pretrained": "laion2b_s34b_b88k",
                "description": "Custom fast model",
                "memory_gb": 3.0,
                "avg_inference_time_ms": 75,
                "accuracy_score": 0.87,
                "enabled": True
            },
            "experimental": {
                "model_name": "ViT-L-16",
                "pretrained": "laion2b_s34b_b79k",
                "description": "Experimental model",
                "memory_gb": 8.0,
                "avg_inference_time_ms": 300,
                "accuracy_score": 0.95,
                "enabled": True
            }
        }
    }


@pytest.fixture
def sample_env_config():
    """Sample environment variable config for testing"""
    return {
        "models": {
            "env_model": {
                "model_name": "ViT-B-32",
                "pretrained": "laion2b_s34b_b88k",
                "description": "Model from environment variable",
                "memory_gb": 2.5,
                "avg_inference_time_ms": 60,
                "accuracy_score": 0.86,
                "enabled": True
            }
        }
    }


@pytest.fixture
def temp_config_file(sample_configmap_data, tmp_path):
    """Create temporary config file for testing"""
    config_file = tmp_path / "test_config.json"
    with open(config_file, 'w') as f:
        json.dump(sample_configmap_data, f, indent=2)
    return str(config_file)


class TestDynamicModelRegistry:
    """Test dynamic model registry functionality"""

    def test_registry_initialization(self, registry):
        """Test registry initializes correctly"""
        assert registry is not None
        assert hasattr(registry, 'list_available_models')
        assert hasattr(registry, 'list_all_models')
        assert hasattr(registry, 'get_model_spec')

    def test_default_models_loaded(self, registry):
        """Test that default models are always available"""
        available = registry.list_available_models()
        
        assert "fast" in available
        assert "accurate" in available
        assert len(available) >= 2

    @pytest.mark.parametrize("model_name,expected_model,expected_pretrained", [
        ("fast", "ViT-B-32", "laion2b_s34b_b79k"),
        ("accurate", "ViT-L-14", "laion2b_s32b_b82k"),
    ])
    def test_default_model_specs(self, registry, model_name, expected_model, expected_pretrained):
        """Test default model specifications"""
        spec = registry.get_model_spec(model_name)
        
        assert spec.model_name == expected_model
        assert spec.pretrained == expected_pretrained
        assert spec.enabled is True
        assert isinstance(spec.memory_gb, (int, float))
        assert isinstance(spec.avg_inference_time_ms, (int, float))
        assert isinstance(spec.accuracy_score, (int, float))

    def test_get_nonexistent_model_raises_error(self, registry):
        """Test error handling for non-existent model specifications"""
        with pytest.raises(ValueError, match="Unknown model config: nonexistent"):
            registry.get_model_spec("nonexistent")

    def test_list_all_vs_available_models(self, registry):
        """Test difference between list_all_models and list_available_models"""
        all_models = registry.list_all_models()
        available_models = registry.list_available_models()
        
        # Available should be subset of all
        for model_name in available_models:
            assert model_name in all_models
            
        # All available models should be enabled
        for model_name in available_models:
            assert all_models[model_name]["enabled"] is True


class TestConfigMapLoading:
    """Test ConfigMap file loading functionality"""

    def test_configmap_file_loading(self, temp_config_file):
        """Test loading configuration from ConfigMap file"""
        registry = DynamicModelRegistry(config_file_path=temp_config_file)
        available = registry.list_available_models()
        
        # Should have defaults + ConfigMap models
        expected_models = {"fast", "accurate", "custom_fast", "experimental"}
        actual_models = set(available.keys())
        assert expected_models == actual_models

    def test_missing_configmap_graceful_handling(self):
        """Test that missing ConfigMap files don't crash the system"""
        registry = DynamicModelRegistry(config_file_path="/non/existent/path/models.json")
        available = registry.list_available_models()
        assert "fast" in available
        assert "accurate" in available
        assert len(available) == 2




class TestEnvironmentVariableConfiguration:
    """Test environment variable configuration loading"""

    def test_environment_variable_configuration(self, sample_env_config):
        """Test model configuration via environment variables"""
        with patch.dict(os.environ, {
            "MODEL_CONFIG_ENV_MODEL": json.dumps(sample_env_config)
        }):
            registry = DynamicModelRegistry()
            available = registry.list_available_models()
            assert "env_model" in available
