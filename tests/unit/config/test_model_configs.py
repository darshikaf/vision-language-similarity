import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import pytest

from service.config.model_configs import DynamicModelRegistry, CLIPModelSpec


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

    def test_default_fast_model_spec(self, registry):
        """Test default fast model specification"""
        fast_spec = registry.get_model_spec("fast")
        
        assert fast_spec.model_name == "ViT-B-32"
        assert fast_spec.pretrained == "laion2b_s34b_b79k"
        assert fast_spec.enabled is True
        assert isinstance(fast_spec.memory_gb, (int, float))
        assert isinstance(fast_spec.avg_inference_time_ms, (int, float))
        assert isinstance(fast_spec.accuracy_score, (int, float))

    def test_default_accurate_model_spec(self, registry):
        """Test default accurate model specification"""
        accurate_spec = registry.get_model_spec("accurate")
        
        assert accurate_spec.model_name == "ViT-L-14"
        assert accurate_spec.pretrained == "laion2b_s32b_b82k"
        assert accurate_spec.enabled is True
        assert isinstance(accurate_spec.memory_gb, (int, float))
        assert isinstance(accurate_spec.avg_inference_time_ms, (int, float))
        assert isinstance(accurate_spec.accuracy_score, (int, float))

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

    def test_configmap_file_loading(self, sample_configmap_data, temp_config_file):
        """Test loading configuration from ConfigMap file"""
        registry = DynamicModelRegistry(config_file_path=temp_config_file)
        
        available = registry.list_available_models()
        
        # Should have defaults + ConfigMap models
        expected_models = {"fast", "accurate", "custom_fast", "experimental"}
        actual_models = set(available.keys())
        assert expected_models == actual_models

    def test_configmap_model_specs_loaded_correctly(self, temp_config_file):
        """Test ConfigMap model specifications are loaded correctly"""
        registry = DynamicModelRegistry(config_file_path=temp_config_file)
        
        custom_spec = registry.get_model_spec("custom_fast")
        assert custom_spec.model_name == "ViT-B-16"
        assert custom_spec.description == "Custom fast model"
        assert custom_spec.memory_gb == 3.0
        
        experimental_spec = registry.get_model_spec("experimental") 
        assert experimental_spec.model_name == "ViT-L-16"
        assert experimental_spec.memory_gb == 8.0

    def test_missing_configmap_graceful_handling(self):
        """Test that missing ConfigMap files don't crash the system"""
        non_existent_path = "/non/existent/path/models.json"
        
        # Should not raise exception
        registry = DynamicModelRegistry(config_file_path=non_existent_path)
        
        # Should still have default models
        available = registry.list_available_models()
        assert "fast" in available
        assert "accurate" in available
        assert len(available) == 2  # Only defaults

    def test_malformed_json_handling(self, tmp_path):
        """Test handling of malformed JSON in ConfigMap"""
        malformed_file = tmp_path / "malformed.json"
        malformed_file.write_text("{ invalid json content }")
        
        # Should not crash, should fall back to defaults
        registry = DynamicModelRegistry(config_file_path=str(malformed_file))
        
        available = registry.list_available_models()
        assert "fast" in available
        assert "accurate" in available
        assert len(available) == 2  # Only defaults

    def test_invalid_model_spec_handling(self, tmp_path):
        """Test handling of invalid model specifications"""
        invalid_config = {
            "models": {
                "valid_model": {
                    "model_name": "ViT-B-16",
                    "pretrained": "laion2b_s34b_b88k",
                    "description": "Valid model",
                    "memory_gb": 3.0,
                    "avg_inference_time_ms": 75,
                    "accuracy_score": 0.87,
                    "enabled": True
                },
                "invalid_model": {
                    "model_name": "ViT-B-16",
                    "pretrained": "laion2b_s34b_b88k",
                    "description": "Invalid model",
                    # Missing required fields
                    "enabled": True
                }
            }
        }
        
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, 'w') as f:
            json.dump(invalid_config, f)
        
        # Should not crash, should load valid models and skip invalid ones
        registry = DynamicModelRegistry(config_file_path=str(invalid_file))
        
        available = registry.list_available_models()
        
        # Valid model should be loaded
        assert "valid_model" in available
        
        # Invalid model should be skipped
        assert "invalid_model" not in available
        
        # Should still have defaults
        assert "fast" in available
        assert "accurate" in available


class TestConfigMapDynamicUpdates:
    """Test ConfigMap dynamic update scenarios"""

    def test_configmap_update_with_restart_simulation(self, sample_configmap_data, tmp_path):
        """Test ConfigMap update followed by registry restart"""
        config_file = tmp_path / "dynamic_config.json"
        
        # Initial configuration
        initial_config = {
            "models": {
                "updatable_model": {
                    "model_name": "ViT-B-16",
                    "pretrained": "laion2b_s34b_b88k",
                    "description": "Initial model",
                    "memory_gb": 3.0,
                    "avg_inference_time_ms": 75,
                    "accuracy_score": 0.87,
                    "enabled": True
                }
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(initial_config, f, indent=2)
        
        # Initial registry
        registry1 = DynamicModelRegistry(config_file_path=str(config_file))
        initial_spec = registry1.get_model_spec("updatable_model")
        assert initial_spec.description == "Initial model"
        assert initial_spec.memory_gb == 3.0
        
        # Update configuration
        updated_config = {
            "models": {
                "updatable_model": {
                    "model_name": "ViT-B-16",
                    "pretrained": "laion2b_s34b_b88k",
                    "description": "UPDATED model with better specs",
                    "memory_gb": 4.0,  # Updated
                    "avg_inference_time_ms": 80,  # Updated
                    "accuracy_score": 0.89,  # Updated
                    "enabled": True
                },
                "new_model": {  # Brand new model
                    "model_name": "ViT-L-32",
                    "pretrained": "laion2b_s32b_b82k",
                    "description": "Newly added model",
                    "memory_gb": 5.0,
                    "avg_inference_time_ms": 150,
                    "accuracy_score": 0.91,
                    "enabled": True
                }
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(updated_config, f, indent=2)
        
        # Simulate restart with new registry instance
        registry2 = DynamicModelRegistry(config_file_path=str(config_file))
        
        # Verify updates
        updated_spec = registry2.get_model_spec("updatable_model")
        assert updated_spec.description == "UPDATED model with better specs"
        assert updated_spec.memory_gb == 4.0
        assert updated_spec.accuracy_score == 0.89
        
        # Verify new model addition
        new_spec = registry2.get_model_spec("new_model")
        assert new_spec.model_name == "ViT-L-32"
        assert new_spec.description == "Newly added model"
        
        available = registry2.list_available_models()
        assert "new_model" in available

    def test_model_removal_and_disabling(self, tmp_path):
        """Test model removal and disabling via ConfigMap updates"""
        config_file = tmp_path / "removal_test.json"
        
        # Initial configuration with multiple models
        initial_config = {
            "models": {
                "model_to_remove": {
                    "model_name": "ViT-B-16",
                    "pretrained": "laion2b_s34b_b88k",
                    "description": "Will be removed",
                    "memory_gb": 3.0,
                    "avg_inference_time_ms": 75,
                    "accuracy_score": 0.87,
                    "enabled": True
                },
                "model_to_disable": {
                    "model_name": "ViT-L-16",
                    "pretrained": "laion2b_s34b_b79k",
                    "description": "Will be disabled",
                    "memory_gb": 5.0,
                    "avg_inference_time_ms": 150,
                    "accuracy_score": 0.91,
                    "enabled": True
                },
                "model_to_keep": {
                    "model_name": "ViT-L-32",
                    "pretrained": "laion2b_s32b_b82k",
                    "description": "Will remain enabled",
                    "memory_gb": 6.0,
                    "avg_inference_time_ms": 200,
                    "accuracy_score": 0.93,
                    "enabled": True
                }
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(initial_config, f, indent=2)
        
        # Initial state
        registry1 = DynamicModelRegistry(config_file_path=str(config_file))
        initial_available = set(registry1.list_available_models().keys())
        initial_all = set(registry1.list_all_models().keys())
        
        # Should have all three custom models plus defaults
        custom_models = initial_available - {"fast", "accurate"}
        assert custom_models == {"model_to_remove", "model_to_disable", "model_to_keep"}
        
        # Update ConfigMap: remove one, disable one, keep one
        updated_config = {
            "models": {
                # model_to_remove is completely removed from ConfigMap
                "model_to_disable": {
                    "model_name": "ViT-L-16",
                    "pretrained": "laion2b_s34b_b79k",
                    "description": "Now disabled",
                    "memory_gb": 5.0,
                    "avg_inference_time_ms": 150,
                    "accuracy_score": 0.91,
                    "enabled": False  # DISABLED
                },
                "model_to_keep": {
                    "model_name": "ViT-L-32",
                    "pretrained": "laion2b_s32b_b82k",
                    "description": "Still enabled",
                    "memory_gb": 6.0,
                    "avg_inference_time_ms": 200,
                    "accuracy_score": 0.93,
                    "enabled": True
                }
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(updated_config, f, indent=2)
        
        # Simulate restart
        registry2 = DynamicModelRegistry(config_file_path=str(config_file))
        final_available = set(registry2.list_available_models().keys())
        final_all = set(registry2.list_all_models().keys())
        
        # Verify removal: model_to_remove should not exist at all
        assert "model_to_remove" not in final_all
        assert "model_to_remove" not in final_available
        
        # Verify disabling: model_to_disable should exist but not be available
        assert "model_to_disable" in final_all
        assert "model_to_disable" not in final_available
        
        # Verify keeping: model_to_keep should be available
        assert "model_to_keep" in final_available
        assert "model_to_keep" in final_all
        
        # Test that disabled model raises error when accessed
        with pytest.raises(ValueError, match="is disabled"):
            registry2.get_model_spec("model_to_disable")


class TestEnvironmentVariableConfiguration:
    """Test environment variable configuration loading"""

    def test_environment_variable_configuration(self, sample_env_config):
        """Test model configuration via environment variables"""
        with patch.dict(os.environ, {
            "MODEL_CONFIG_ENV_MODEL": json.dumps(sample_env_config)
        }):
            registry = DynamicModelRegistry()
            
            # Should have default models + env model
            available = registry.list_available_models()
            assert "env_model" in available
            
            env_spec = registry.get_model_spec("env_model")
            assert env_spec.model_name == "ViT-B-32"
            assert env_spec.description == "Model from environment variable"
            assert env_spec.memory_gb == 2.5

    @pytest.mark.parametrize("env_var_name", [
        "MODEL_CONFIG_CUSTOM_MODEL",
        "MODEL_CONFIG_TEST_MODEL", 
        "MODEL_CONFIG_PRODUCTION_MODEL"
    ])
    def test_multiple_environment_variables(self, env_var_name, sample_env_config):
        """Test loading from different environment variable names"""
        expected_model_name = env_var_name[len("MODEL_CONFIG_"):].lower()
        
        with patch.dict(os.environ, {env_var_name: json.dumps(sample_env_config)}):
            registry = DynamicModelRegistry()
            
            available = registry.list_available_models()
            # The env_model from the config should be loaded
            assert "env_model" in available

    def test_invalid_environment_variable_json(self):
        """Test handling of invalid JSON in environment variables"""
        with patch.dict(os.environ, {
            "MODEL_CONFIG_INVALID": "{ invalid json }"
        }):
            # Should not crash, should load only defaults
            registry = DynamicModelRegistry()
            
            available = registry.list_available_models()
            assert "fast" in available
            assert "accurate" in available
            assert len(available) == 2  # Only defaults

    def test_environment_variable_with_invalid_spec(self):
        """Test handling of invalid model spec in environment variables"""
        invalid_env_config = {
            "models": {
                "invalid_env_model": {
                    "model_name": "ViT-B-32",
                    # Missing required fields
                    "enabled": True
                }
            }
        }
        
        with patch.dict(os.environ, {
            "MODEL_CONFIG_INVALID_SPEC": json.dumps(invalid_env_config)
        }):
            # Should not crash, should skip invalid model
            registry = DynamicModelRegistry()
            
            available = registry.list_available_models()
            assert "invalid_env_model" not in available
            assert "fast" in available
            assert "accurate" in available


class TestCLIPModelSpec:
    """Test CLIPModelSpec data class functionality"""

    def test_clip_model_spec_creation(self):
        """Test CLIPModelSpec creation with all fields"""
        spec = CLIPModelSpec(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            description="Test model",
            memory_gb=2.0,
            avg_inference_time_ms=100.0,
            accuracy_score=0.85,
            enabled=True
        )
        
        assert spec.model_name == "ViT-B-32"
        assert spec.pretrained == "laion2b_s34b_b79k"
        assert spec.description == "Test model"
        assert spec.memory_gb == 2.0
        assert spec.avg_inference_time_ms == 100.0
        assert spec.accuracy_score == 0.85
        assert spec.enabled is True

    def test_clip_model_spec_defaults(self):
        """Test CLIPModelSpec with default values for enabled field"""
        spec = CLIPModelSpec(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            description="Test model",
            memory_gb=1.5,
            avg_inference_time_ms=75.0,
            accuracy_score=0.85
        )
        
        assert spec.model_name == "ViT-B-32"
        assert spec.pretrained == "laion2b_s34b_b79k"
        assert spec.description == "Test model"
        assert spec.memory_gb == 1.5
        assert spec.avg_inference_time_ms == 75.0
        assert spec.accuracy_score == 0.85
        assert spec.enabled is True  # This is the only field with a default

    @pytest.mark.parametrize("field_name,field_value", [
        ("memory_gb", 4.5),
        ("avg_inference_time_ms", 150.0),
        ("accuracy_score", 0.92),
        ("enabled", False),
    ])
    def test_clip_model_spec_field_types(self, field_name, field_value):
        """Test CLIPModelSpec field type handling"""
        kwargs = {
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "description": "Test model description",
            "memory_gb": 2.0,
            "avg_inference_time_ms": 100.0,
            "accuracy_score": 0.88,
            field_name: field_value
        }
        
        spec = CLIPModelSpec(**kwargs)
        assert getattr(spec, field_name) == field_value


class TestModelRegistryIntegration:
    """Test model registry integration scenarios"""

    def test_registry_with_both_file_and_env_configs(self, temp_config_file, sample_env_config):
        """Test registry with both ConfigMap file and environment variables"""
        with patch.dict(os.environ, {
            "MODEL_CONFIG_ENV_TEST": json.dumps(sample_env_config)
        }):
            registry = DynamicModelRegistry(config_file_path=temp_config_file)
            
            available = registry.list_available_models()
            
            # Should have defaults + ConfigMap + env models
            expected_models = {"fast", "accurate", "custom_fast", "experimental", "env_model"}
            actual_models = set(available.keys())
            assert expected_models == actual_models

    def test_registry_consistency_across_instances(self, temp_config_file):
        """Test that multiple registry instances are consistent"""
        registry1 = DynamicModelRegistry(config_file_path=temp_config_file)
        registry2 = DynamicModelRegistry(config_file_path=temp_config_file)
        
        available1 = set(registry1.list_available_models().keys())
        available2 = set(registry2.list_available_models().keys())
        
        assert available1 == available2

    def test_registry_model_spec_consistency(self, registry):
        """Test that model specs are consistent between access methods"""
        all_models = registry.list_all_models()
        available_models = registry.list_available_models()
        
        # Compare specs accessed via different methods
        for model_name in available_models:
            direct_spec = registry.get_model_spec(model_name)
            all_spec = all_models[model_name]
            available_spec = available_models[model_name]
            
            # Should have same core properties
            assert direct_spec.model_name == all_spec["model_name"]
            assert direct_spec.model_name == available_spec["model_name"]
            assert direct_spec.enabled == all_spec["enabled"]
            assert direct_spec.enabled == available_spec["enabled"]