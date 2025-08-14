"""Tests for model configuration management and ConfigMap scenarios."""

import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import pytest

from service.config.model_configs import DynamicModelRegistry, CLIPModelSpec


class TestModelConfigManagement:
    """Test model configuration loading and management scenarios"""

    def test_default_models_loaded(self):
        """Test that default models are always available"""
        registry = DynamicModelRegistry()
        
        available = registry.list_available_models()
        assert "fast" in available
        assert "accurate" in available
        
        # Verify default specs
        fast_spec = registry.get_model_spec("fast")
        assert fast_spec.model_name == "ViT-B-32"
        assert fast_spec.pretrained == "laion2b_s34b_b79k"
        assert fast_spec.enabled is True
        
        accurate_spec = registry.get_model_spec("accurate")
        assert accurate_spec.model_name == "ViT-L-14"
        assert accurate_spec.pretrained == "laion2b_s32b_b82k"
        assert accurate_spec.enabled is True

    def test_configmap_pod_startup_scenario(self):
        """Test ConfigMap loading on pod startup (K8s scenario)"""
        # Create temporary ConfigMap file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            configmap_data = {
                "models": {
                    "custom_fast": {
                        "model_name": "ViT-B-16",
                        "pretrained": "laion2b_s34b_b88k",
                        "description": "Custom fast model from ConfigMap",
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
            json.dump(configmap_data, f, indent=2)
            config_file = f.name

        try:
            # Simulate pod startup with ConfigMap
            registry = DynamicModelRegistry(config_file_path=config_file)
            
            available = registry.list_available_models()
            
            # Should have defaults + ConfigMap models
            expected_models = {"fast", "accurate", "custom_fast", "experimental"}
            actual_models = set(available.keys())
            assert expected_models == actual_models
            
            # Verify ConfigMap models loaded correctly
            custom_spec = registry.get_model_spec("custom_fast")
            assert custom_spec.model_name == "ViT-B-16"
            assert custom_spec.description == "Custom fast model from ConfigMap"
            assert custom_spec.memory_gb == 3.0
            
            experimental_spec = registry.get_model_spec("experimental")
            assert experimental_spec.model_name == "ViT-L-16"
            assert experimental_spec.memory_gb == 8.0
            
        finally:
            os.unlink(config_file)

    def test_configmap_update_with_pod_restart(self):
        """Test ConfigMap update followed by pod restart"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
            json.dump(initial_config, f, indent=2)
            config_file = f.name

        try:
            # Initial pod startup
            registry1 = DynamicModelRegistry(config_file_path=config_file)
            initial_spec = registry1.get_model_spec("updatable_model")
            assert initial_spec.description == "Initial model"
            assert initial_spec.memory_gb == 3.0
            assert initial_spec.accuracy_score == 0.87

            # Simulate ConfigMap update
            with open(config_file, 'w') as f:
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
                json.dump(updated_config, f, indent=2)

            # Simulate pod restart (new registry instance)
            registry2 = DynamicModelRegistry(config_file_path=config_file)
            
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
            
        finally:
            os.unlink(config_file)

    def test_model_removal_and_disabling(self):
        """Test model removal and disabling via ConfigMap"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
            json.dump(initial_config, f, indent=2)
            config_file = f.name

        try:
            # Initial state
            registry1 = DynamicModelRegistry(config_file_path=config_file)
            initial_available = set(registry1.list_available_models().keys())
            initial_all = set(registry1.list_all_models().keys())
            
            # Should have all three custom models plus defaults
            custom_models = initial_available - {"fast", "accurate"}
            assert custom_models == {"model_to_remove", "model_to_disable", "model_to_keep"}

            # Update ConfigMap: remove one, disable one, keep one
            with open(config_file, 'w') as f:
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
                json.dump(updated_config, f, indent=2)

            # Simulate pod restart
            registry2 = DynamicModelRegistry(config_file_path=config_file)
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
                
        finally:
            os.unlink(config_file)


    def test_environment_variable_configuration(self):
        """Test model configuration via environment variables"""
        env_config = {
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b88k",
            "description": "Model from environment variable",
            "memory_gb": 2.5,
            "avg_inference_time_ms": 60,
            "accuracy_score": 0.86,
            "enabled": True
        }
        
        with patch.dict(os.environ, {
            "MODEL_CONFIG_ENV_MODEL": json.dumps(env_config)
        }):
            registry = DynamicModelRegistry()
            
            # Should have default models + env model
            available = registry.list_available_models()
            assert "env_model" in available
            
            env_spec = registry.get_model_spec("env_model")
            assert env_spec.model_name == "ViT-B-32"
            assert env_spec.description == "Model from environment variable"
            assert env_spec.memory_gb == 2.5


    def test_invalid_configuration_handling(self):
        """Test handling of invalid model configurations"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
                        # Missing required fields: memory_gb, avg_inference_time_ms, accuracy_score
                        "enabled": True
                    }
                }
            }
            json.dump(invalid_config, f, indent=2)
            config_file = f.name

        try:
            # Should not crash, should load valid models and skip invalid ones
            registry = DynamicModelRegistry(config_file_path=config_file)
            
            available = registry.list_available_models()
            
            # Valid model should be loaded
            assert "valid_model" in available
            
            # Invalid model should be skipped
            assert "invalid_model" not in available
            
            # Should still have defaults
            assert "fast" in available
            assert "accurate" in available
            
        finally:
            os.unlink(config_file)

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

    def test_malformed_json_handling(self):
        """Test handling of malformed JSON in ConfigMap"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content }")
            config_file = f.name

        try:
            # Should not crash, should fall back to defaults
            registry = DynamicModelRegistry(config_file_path=config_file)
            
            available = registry.list_available_models()
            assert "fast" in available
            assert "accurate" in available
            assert len(available) == 2  # Only defaults
            
        finally:
            os.unlink(config_file)

    def test_get_nonexistent_model_error(self):
        """Test error handling for non-existent model specifications"""
        registry = DynamicModelRegistry()
        
        with pytest.raises(ValueError, match="Unknown model config: nonexistent"):
            registry.get_model_spec("nonexistent")

    def test_list_all_vs_available_models(self):
        """Test difference between list_all_models and list_available_models"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "models": {
                    "enabled_model": {
                        "model_name": "ViT-B-16",
                        "pretrained": "laion2b_s34b_b88k",
                        "description": "Enabled model",
                        "memory_gb": 3.0,
                        "avg_inference_time_ms": 75,
                        "accuracy_score": 0.87,
                        "enabled": True
                    },
                    "disabled_model": {
                        "model_name": "ViT-L-16",
                        "pretrained": "laion2b_s34b_b79k",
                        "description": "Disabled model",
                        "memory_gb": 5.0,
                        "avg_inference_time_ms": 150,
                        "accuracy_score": 0.91,
                        "enabled": False
                    }
                }
            }
            json.dump(config, f, indent=2)
            config_file = f.name

        try:
            registry = DynamicModelRegistry(config_file_path=config_file)
            
            all_models = set(registry.list_all_models().keys())
            available_models = set(registry.list_available_models().keys())
            
            # All models should include everything
            assert "enabled_model" in all_models
            assert "disabled_model" in all_models
            assert "fast" in all_models
            assert "accurate" in all_models
            
            # Available should only include enabled models
            assert "enabled_model" in available_models
            assert "disabled_model" not in available_models  # Disabled
            assert "fast" in available_models
            assert "accurate" in available_models
            
        finally:
            os.unlink(config_file)
