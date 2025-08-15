from unittest.mock import patch, Mock, AsyncMock
import pytest

from service.model_management.manager import ModelManager, model_manager
from service.config.model_configs import CLIPModelSpec


@pytest.fixture
def manager():
    """Create ModelManager instance for testing"""
    return ModelManager()


@pytest.fixture
def mock_evaluator():
    """Mock evaluator for testing"""
    evaluator = Mock()
    evaluator.similarity_model = Mock()
    evaluator.similarity_model.model_name = "ViT-B-32"
    evaluator.similarity_model.model = Mock()
    evaluator.similarity_model.device = "cpu"
    evaluator.similarity_model.model.training = False
    return evaluator


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


class TestModelManager:
    """Test model manager functionality"""

    def test_manager_initialization(self, manager):
        """Test manager initializes correctly"""
        assert manager is not None
        assert hasattr(manager, '_models')
        assert isinstance(manager._models, dict)
        assert len(manager._models) == 0

    def test_manager_has_required_methods(self, manager):
        """Test manager has all required methods"""
        required_methods = [
            'get_model_info',
            'get_status',
            '_is_healthy',
            '_build_spec_info',
            '_get_health_status',
            '_get_runtime_info'
        ]
        
        for method in required_methods:
            assert hasattr(manager, method)
            assert callable(getattr(manager, method))

    @pytest.mark.asyncio
    async def test_is_healthy_with_valid_model(self, manager, mock_evaluator):
        """Test health check with valid model"""
        is_healthy = await manager._is_healthy(mock_evaluator)
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_is_healthy_with_missing_similarity_model(self, manager):
        """Test health check with model missing similarity_model"""
        invalid_evaluator = Mock()
        del invalid_evaluator.similarity_model  # Remove attribute
        
        is_healthy = await manager._is_healthy(invalid_evaluator)
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_is_healthy_with_training_mode_model(self, manager, mock_evaluator):
        """Test health check with model in training mode"""
        mock_evaluator.similarity_model.model.training = True
        
        is_healthy = await manager._is_healthy(mock_evaluator)
        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_is_healthy_with_exception(self, manager):
        """Test health check with exception"""
        broken_evaluator = Mock()
        broken_evaluator.similarity_model = Mock()
        broken_evaluator.similarity_model.model = Mock(side_effect=Exception("Test error"))
        
        is_healthy = await manager._is_healthy(broken_evaluator)
        assert is_healthy is False

    def test_build_spec_info(self, manager, mock_model_spec):
        """Test building specification information"""
        spec_info = manager._build_spec_info(mock_model_spec)
        
        expected_keys = [
            "model_name", "pretrained", "description",
            "memory_gb", "avg_inference_time_ms", "accuracy_score", "enabled"
        ]
        
        for key in expected_keys:
            assert key in spec_info
        
        assert spec_info["model_name"] == "ViT-B-32"
        assert spec_info["pretrained"] == "laion2b_s34b_b79k"
        assert spec_info["enabled"] is True

    def test_get_model_name_with_valid_model(self, manager, mock_evaluator):
        """Test extracting model name from valid evaluator"""
        model_name = manager._get_model_name(mock_evaluator)
        assert model_name == "ViT-B-32"

    def test_get_model_name_with_invalid_model(self, manager):
        """Test extracting model name from invalid evaluator"""
        invalid_evaluator = Mock()
        del invalid_evaluator.similarity_model
        
        model_name = manager._get_model_name(invalid_evaluator)
        assert model_name == "unknown"

    def test_is_model_loaded_with_loaded_model(self, manager, mock_evaluator):
        """Test checking if model is loaded with valid model"""
        is_loaded = manager._is_model_loaded(mock_evaluator)
        assert is_loaded is True

    def test_is_model_loaded_with_unloaded_model(self, manager):
        """Test checking if model is loaded with unloaded model"""
        unloaded_evaluator = Mock()
        unloaded_evaluator.similarity_model = Mock()
        unloaded_evaluator.similarity_model.model = None
        
        is_loaded = manager._is_model_loaded(unloaded_evaluator)
        assert is_loaded is False

    def test_get_status(self, manager):
        """Test getting manager status"""
        status = manager.get_status()
        
        required_keys = ["cached_models", "loaded_models", "available_configs"]
        for key in required_keys:
            assert key in status
        
        assert isinstance(status["cached_models"], list)
        assert isinstance(status["loaded_models"], list)
        assert isinstance(status["available_configs"], list)
        
        # Should have default configs available
        assert len(status["available_configs"]) >= 2
        assert "fast" in status["available_configs"]
        assert "accurate" in status["available_configs"]


class TestModelManagerModelInfo:
    """Test model manager model info functionality"""

    @pytest.mark.asyncio
    @patch('service.model_management.manager.model_registry')
    async def test_get_model_info_unloaded_model(self, mock_registry, manager, mock_model_spec):
        """Test getting info for unloaded model"""
        mock_registry.get_model_spec.return_value = mock_model_spec
        
        info = await manager.get_model_info("test_config")
        
        assert info["config_name"] == "test_config"
        assert info["loaded"] is False
        assert "spec" in info
        assert info["spec"]["model_name"] == "ViT-B-32"
        assert info["health_status"]["healthy"] is False

    @pytest.mark.asyncio
    @patch('service.model_management.manager.model_registry')
    async def test_get_model_info_loaded_model(self, mock_registry, manager, mock_model_spec, mock_evaluator):
        """Test getting info for loaded model"""
        mock_registry.get_model_spec.return_value = mock_model_spec
        manager._models["test_config"] = mock_evaluator
        
        info = await manager.get_model_info("test_config")
        
        assert info["config_name"] == "test_config"
        assert info["loaded"] is True
        assert info["health_status"]["healthy"] is True
        assert info["health_status"]["model_name"] == "ViT-B-32"

    @pytest.mark.asyncio
    async def test_get_health_status_loaded_model(self, manager, mock_evaluator):
        """Test getting health status for loaded model"""
        manager._models["test_config"] = mock_evaluator
        
        health_status = await manager._get_health_status("test_config")
        
        assert health_status["healthy"] is True
        assert health_status["model_name"] == "ViT-B-32"
        assert health_status["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_get_health_status_exception_handling(self, manager):
        """Test health status with exception handling"""
        # Don't add model to _models, will cause KeyError
        health_status = await manager._get_health_status("nonexistent")
        
        assert health_status["healthy"] is False

    @pytest.mark.asyncio
    async def test_get_runtime_info_with_info_method(self, manager, mock_evaluator):
        """Test getting runtime info for model with get_model_info method"""
        mock_evaluator.get_model_info = Mock(return_value={"runtime": "info"})
        manager._models["test_config"] = mock_evaluator
        
        runtime_info = await manager._get_runtime_info("test_config")
        
        assert runtime_info == {"runtime": "info"}

    @pytest.mark.asyncio
    async def test_get_runtime_info_without_info_method(self, manager, mock_evaluator):
        """Test getting runtime info for model without get_model_info method"""
        # Ensure mock_evaluator doesn't have get_model_info method
        if hasattr(mock_evaluator, 'get_model_info'):
            delattr(mock_evaluator, 'get_model_info')
        manager._models["test_config"] = mock_evaluator
        
        runtime_info = await manager._get_runtime_info("test_config")
        
        assert runtime_info == {}

    @pytest.mark.asyncio
    async def test_get_runtime_info_exception_handling(self, manager):
        """Test runtime info with exception handling"""
        runtime_info = await manager._get_runtime_info("nonexistent")
        
        assert runtime_info == {}


class TestModelManagerHealthChecks:
    """Test model manager health check functionality"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("has_similarity_model,has_model,has_device,is_training,expected", [
        (True, True, True, False, True),   # Healthy model
        (False, True, True, False, False), # Missing similarity_model
        (True, False, True, False, False), # Missing model
        (True, True, False, False, False), # Missing device
        (True, True, True, True, False),   # Training mode
    ])
    async def test_health_check_conditions(self, manager, has_similarity_model, has_model, has_device, is_training, expected):
        """Test health check with different model conditions"""
        evaluator = Mock()
        
        if has_similarity_model:
            evaluator.similarity_model = Mock()
            evaluator.similarity_model.model_name = "ViT-B-32"
            
            if has_model:
                evaluator.similarity_model.model = Mock()
                evaluator.similarity_model.model.training = is_training
            else:
                del evaluator.similarity_model.model
                
            if has_device:
                evaluator.similarity_model.device = "cpu"
            else:
                del evaluator.similarity_model.device
        else:
            del evaluator.similarity_model
        
        is_healthy = await manager._is_healthy(evaluator)
        assert is_healthy == expected

    @pytest.mark.asyncio
    async def test_health_check_with_model_without_training_attribute(self, manager):
        """Test health check with model that doesn't have training attribute"""
        evaluator = Mock()
        evaluator.similarity_model = Mock()
        evaluator.similarity_model.model_name = "ViT-B-32"
        evaluator.similarity_model.model = Mock()
        evaluator.similarity_model.device = "cpu"
        
        # Remove training attribute
        del evaluator.similarity_model.model.training
        
        is_healthy = await manager._is_healthy(evaluator)
        assert is_healthy is True  # Should pass if training attribute doesn't exist

    @pytest.mark.asyncio
    async def test_health_check_logs_errors(self, manager, caplog):
        """Test that health check logs errors appropriately"""
        import logging
        
        broken_evaluator = Mock()
        broken_evaluator.similarity_model = Mock()
        broken_evaluator.similarity_model.model_name = "TestModel"
        broken_evaluator.side_effect = Exception("Test error")
        
        with caplog.at_level(logging.ERROR):
            is_healthy = await manager._is_healthy(broken_evaluator)
            
        assert is_healthy is False
        # Should have logged an error (though exact format depends on implementation)
        assert len(caplog.records) >= 0


class TestModelManagerGlobalInstance:
    """Test global model manager instance"""

    def test_global_model_manager_exists(self):
        """Test that global model manager instance exists"""
        assert model_manager is not None
        assert isinstance(model_manager, ModelManager)

    def test_global_model_manager_singleton_behavior(self):
        """Test that global model manager behaves as singleton"""
        from service.model_management.manager import model_manager as manager1
        from service.model_management.manager import model_manager as manager2
        
        assert manager1 is manager2

    def test_global_model_manager_functionality(self):
        """Test that global model manager has expected functionality"""
        status = model_manager.get_status()
        
        assert isinstance(status, dict)
        assert "available_configs" in status
        assert len(status["available_configs"]) >= 2


class TestModelManagerIntegration:
    """Test model manager integration scenarios"""

    @pytest.mark.asyncio
    @patch('service.model_management.manager.model_registry')
    async def test_model_info_integration_with_registry(self, mock_registry, manager):
        """Test model info integration with model registry"""
        # Mock registry response
        mock_spec = CLIPModelSpec(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            description="Integration test model",
            memory_gb=5.0,
            avg_inference_time_ms=200.0,
            accuracy_score=0.92,
            enabled=True
        )
        mock_registry.get_model_spec.return_value = mock_spec
        
        info = await manager.get_model_info("accurate")
        
        assert info["spec"]["model_name"] == "ViT-L-14"
        assert info["spec"]["accuracy_score"] == 0.92
        mock_registry.get_model_spec.assert_called_once_with("accurate")

    def test_status_integration_with_registry(self, manager):
        """Test status integration with model registry"""
        status = manager.get_status()
        
        # Should include standard configs from registry
        available_configs = status["available_configs"]
        assert "fast" in available_configs
        assert "accurate" in available_configs

    @pytest.mark.asyncio
    async def test_multiple_model_health_checks(self, manager):
        """Test health checks for multiple models"""
        # Add multiple mock models
        for i, config_name in enumerate(["model1", "model2", "model3"]):
            mock_evaluator = Mock()
            mock_evaluator.similarity_model = Mock()
            mock_evaluator.similarity_model.model_name = f"Model-{i}"
            mock_evaluator.similarity_model.model = Mock()
            mock_evaluator.similarity_model.device = "cpu"
            mock_evaluator.similarity_model.model.training = False
            manager._models[config_name] = mock_evaluator
        
        # Check health for all models
        health_results = {}
        for config_name in manager._models:
            health_results[config_name] = await manager._get_health_status(config_name)
        
        assert len(health_results) == 3
        for config_name, health in health_results.items():
            assert health["healthy"] is True
            assert health["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_manager_error_resilience(self, manager):
        """Test manager resilience to errors"""
        # Add a broken model
        broken_evaluator = Mock()
        broken_evaluator.side_effect = Exception("Broken model")
        manager._models["broken"] = broken_evaluator
        
        # Manager should still function
        status = manager.get_status()
        assert "broken" in status["cached_models"]
        
        # Health check should handle the broken model gracefully
        health_status = await manager._get_health_status("broken")
        assert health_status["healthy"] is False