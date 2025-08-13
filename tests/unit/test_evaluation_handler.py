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

    def test_get_evaluator_invalid_config(self, handler):
        with pytest.raises(ValueError, match="Unknown model configuration"):
            handler._get_evaluator("invalid_config")

    @pytest.mark.asyncio
    async def test_evaluate_single_success(self, handler, sample_evaluation_result, disable_metrics):
        request = EvaluationRequest(
            image_input="test.jpg",
            text_prompt="test prompt",
            model_config_name="fast"
        )
        
        with patch.object(handler, '_get_evaluator') as mock_get_evaluator:
            mock_evaluator = Mock()
            mock_evaluator.evaluate_single = AsyncMock(return_value=sample_evaluation_result)
            mock_get_evaluator.return_value = mock_evaluator
            
            response = await handler.evaluate_single(request)
            
            assert response.clip_score == 75.5
            assert response.processing_time_ms == 150.0
            assert response.error is None
            assert response.model_used == "fast"

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, handler):
        with patch.object(handler, '_get_evaluator') as mock_get_evaluator:
            mock_evaluator = Mock()
            mock_evaluator.similarity_model = Mock()
            mock_evaluator.similarity_model.model = Mock()
            mock_get_evaluator.return_value = mock_evaluator
            
            response = await handler.health_check()
            
            assert response.status == "healthy"
            assert response.model_loaded is True

    def test_get_handler_singleton(self):
        handler1 = get_handler()
        handler2 = get_handler()
        assert handler1 is handler2
