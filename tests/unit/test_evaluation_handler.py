import pytest
from unittest.mock import AsyncMock, Mock, patch
from service.evaluation.handler import EvaluationHandler, get_handler
from service.evaluation.schema import EvaluationRequest
from service.core import EvaluationResult


class TestEvaluationHandler:
    """Simplified evaluation handler tests"""

    @pytest.fixture
    def handler(self):
        return EvaluationHandler()

    @pytest.fixture
    def sample_evaluation_result(self):
        return EvaluationResult(
            image_path="test_image.jpg",
            text_prompt="test prompt",
            clip_score=75.5,
            processing_time_ms=150.0,
            error=None
        )

    @pytest.mark.asyncio
    async def test_model_manager_invalid_config(self, handler):
        """Test that invalid model config raises appropriate error"""
        with patch('service.evaluation.handler.model_manager') as mock_manager:
            mock_manager.model_context.side_effect = ValueError("Failed to load model configuration: invalid_config")
            
            request = EvaluationRequest(
                image_input="test.jpg",
                text_prompt="test prompt",
                model_config_name="invalid_config"
            )
            
            with pytest.raises(ValueError, match="Failed to load model configuration"):
                await handler.evaluate_single(request)

    @pytest.mark.asyncio
    async def test_evaluate_single_success(self, handler, sample_evaluation_result, disable_metrics):
        request = EvaluationRequest(
            image_input="test.jpg",
            text_prompt="test prompt",
            model_config_name="fast"
        )
        
        with patch('service.evaluation.handler.model_manager') as mock_manager:
            # Create mock evaluator with evaluate_single method
            mock_evaluator = Mock()
            mock_evaluator.evaluate_single = AsyncMock(return_value=sample_evaluation_result)
            
            # Mock the context manager
            mock_manager.model_context.return_value.__aenter__ = AsyncMock(return_value=mock_evaluator)
            mock_manager.model_context.return_value.__aexit__ = AsyncMock(return_value=None)
            
            response = await handler.evaluate_single(request)
            
            assert response.clip_score == 75.5
            assert response.processing_time_ms == 150.0
            assert response.error is None
            assert response.model_used == "fast"
            
            # Verify model manager was called with correct config
            mock_manager.model_context.assert_called_once_with("fast")

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, handler):
        with patch('service.evaluation.handler.model_manager') as mock_manager:
            # Mock model info with healthy status
            mock_model_info = {
                "health_status": {"healthy": True},
                "loaded": True
            }
            mock_manager.get_model_info = AsyncMock(return_value=mock_model_info)
            
            with patch('service.evaluation.handler.model_registry') as mock_registry:
                mock_registry.list_available_models.return_value = {"fast": {}, "accurate": {}}
                
                response = await handler.health_check()
                
                assert response.status == "healthy"
                assert response.model_loaded is True
                assert "fast" in response.available_configs
                
                # Verify model manager was called for health check
                mock_manager.get_model_info.assert_called_once_with("fast")

    def test_get_handler_singleton(self):
        handler1 = get_handler()
        handler2 = get_handler()
        assert handler1 is handler2
