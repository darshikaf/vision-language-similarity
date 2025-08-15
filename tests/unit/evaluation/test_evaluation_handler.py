from unittest.mock import AsyncMock, Mock, patch
import pytest

from service.evaluation.handler import EvaluationHandler, get_handler
from service.evaluation.schema import EvaluationRequest, BatchEvaluationRequest
from service.core.exceptions import ServiceError


@pytest.fixture
def sample_evaluation_request():
    """Sample evaluation request for testing"""
    return EvaluationRequest(
        image_input="test_image.jpg",
        text_prompt="test prompt",
        model_config_name="fast"
    )


@pytest.fixture
def sample_batch_request():
    """Sample batch evaluation request for testing"""
    return BatchEvaluationRequest(
        evaluations=[
            EvaluationRequest(
                image_input="test1.jpg",
                text_prompt="prompt 1",
                model_config_name="fast"
            ),
            EvaluationRequest(
                image_input="test2.jpg", 
                text_prompt="prompt 2",
                model_config_name="accurate"
            )
        ],
        batch_size=10
    )


@pytest.fixture
def handler():
    """Create handler instance for testing"""
    return EvaluationHandler()


@pytest.fixture
def mock_evaluation_result():
    """Mock evaluation result for testing"""
    mock_result = Mock()
    mock_result.image_path = "test_image.jpg"
    mock_result.text_prompt = "test prompt"
    mock_result.clip_score = 85.2
    mock_result.processing_time_ms = 120.5
    mock_result.error = None
    return mock_result


@pytest.fixture
def mock_evaluation_result_with_error():
    """Mock evaluation result with error for testing"""
    mock_result = Mock()
    mock_result.image_path = "test_image.jpg"
    mock_result.text_prompt = "test prompt"
    mock_result.clip_score = 0.0
    mock_result.processing_time_ms = 50.0
    mock_result.error = "Image loading failed"
    return mock_result


class TestEvaluationHandler:
    """Test evaluation handler functionality"""

    def test_handler_initialization(self, handler):
        """Test handler initializes correctly"""
        assert handler is not None
        assert hasattr(handler, 'evaluate_single')
        assert hasattr(handler, 'evaluate_batch')
        assert hasattr(handler, 'health_check')

    @pytest.mark.asyncio
    async def test_evaluate_single_invalid_config(self, handler):
        """Test single evaluation with invalid model configuration"""
        request = EvaluationRequest(
            image_input="test.jpg",
            text_prompt="test prompt",
            model_config_name="invalid_config"
        )
        
        with pytest.raises(ValueError, match="Unknown model config"):
            await handler.evaluate_single(request)

    @pytest.mark.asyncio
    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    async def test_evaluate_single_success(self, mock_evaluator_class, handler, sample_evaluation_request, mock_evaluation_result):
        """Test successful single evaluation"""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_single = AsyncMock(return_value=mock_evaluation_result)
        mock_evaluator_class.return_value = mock_evaluator
        
        response = await handler.evaluate_single(sample_evaluation_request)
        
        assert response.clip_score == 85.2
        assert response.processing_time_ms == 120.5
        assert response.error is None
        assert response.model_used == "fast"
        assert response.image_input == "test_image.jpg"
        assert response.text_prompt == "test prompt"
        
        mock_evaluator_class.assert_called_once_with(model_config_name="fast")

    @pytest.mark.asyncio
    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    async def test_evaluate_single_with_error(self, mock_evaluator_class, handler, sample_evaluation_request, mock_evaluation_result_with_error):
        """Test single evaluation that encounters an error"""
        mock_evaluator = Mock()
        mock_evaluator.evaluate_single = AsyncMock(return_value=mock_evaluation_result_with_error)
        mock_evaluator_class.return_value = mock_evaluator
        
        response = await handler.evaluate_single(sample_evaluation_request)
        
        assert response.clip_score == 0.0
        assert response.error == "Image loading failed"
        assert response.processing_time_ms == 50.0

    @pytest.mark.asyncio
    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    async def test_evaluate_batch_success(self, mock_evaluator_class, handler, sample_batch_request):
        """Test successful batch evaluation"""
        mock_evaluator = Mock()
        
        # Create mock results for individual evaluations
        mock_result1 = Mock()
        mock_result1.image_path = "test1.jpg"
        mock_result1.text_prompt = "prompt 1" 
        mock_result1.clip_score = 78.5
        mock_result1.processing_time_ms = 100.0
        mock_result1.error = None
        
        mock_result2 = Mock()
        mock_result2.image_path = "test2.jpg"
        mock_result2.text_prompt = "prompt 2"
        mock_result2.clip_score = 82.1
        mock_result2.processing_time_ms = 110.0
        mock_result2.error = None
        
        # Handler calls evaluate_single for each request, so mock that method
        mock_evaluator.evaluate_single = AsyncMock(side_effect=[mock_result1, mock_result2])
        mock_evaluator_class.return_value = mock_evaluator
        
        response = await handler.evaluate_batch(sample_batch_request)
        
        assert len(response.results) == 2
        assert response.total_processed == 2
        assert response.total_successful == 2
        assert response.total_failed == 0
        assert response.results[0].clip_score == 78.5
        assert response.results[1].clip_score == 82.1

    @pytest.mark.asyncio
    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    async def test_evaluate_batch_with_mixed_results(self, mock_evaluator_class, handler, sample_batch_request):
        """Test batch evaluation with some successes and failures"""
        mock_evaluator = Mock()
        
        # Create mock mixed results
        mock_result1 = Mock()
        mock_result1.image_path = "test1.jpg"
        mock_result1.text_prompt = "prompt 1"
        mock_result1.clip_score = 78.5
        mock_result1.processing_time_ms = 100.0
        mock_result1.error = None
        
        mock_result2 = Mock()
        mock_result2.image_path = "test2.jpg"
        mock_result2.text_prompt = "prompt 2"
        mock_result2.clip_score = 0.0
        mock_result2.processing_time_ms = 25.0
        mock_result2.error = "Image not found"
        
        # Handler calls evaluate_single for each request, so mock that method
        mock_evaluator.evaluate_single = AsyncMock(side_effect=[mock_result1, mock_result2])
        mock_evaluator_class.return_value = mock_evaluator
        
        response = await handler.evaluate_batch(sample_batch_request)
        
        assert response.total_processed == 2
        assert response.total_successful == 1
        assert response.total_failed == 1
        assert response.results[1].error == "Image not found"

    @pytest.mark.asyncio
    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    async def test_health_check_healthy(self, mock_evaluator_class, handler):
        """Test health check when service is healthy"""
        mock_evaluator = Mock()
        mock_evaluator.similarity_model = Mock()
        mock_evaluator.similarity_model.model = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        
        response = await handler.health_check()
        
        assert response.status == "healthy"
        assert response.model_loaded is True
        assert isinstance(response.available_configs, list)
        assert len(response.available_configs) >= 2
        assert "fast" in response.available_configs
        assert "accurate" in response.available_configs

    @pytest.mark.asyncio
    @patch('service.evaluation.handler.MinimalOpenCLIPEvaluator')
    async def test_health_check_model_loading_error(self, mock_evaluator_class, handler):
        """Test health check when model loading fails"""  
        mock_evaluator_class.side_effect = Exception("Model loading failed")
        
        response = await handler.health_check()
        
        assert response.status == "unhealthy"
        assert response.model_loaded is False
        assert len(response.available_configs) >= 2

    def test_handler_singleton_pattern(self):
        """Test that get_handler returns singleton instance"""
        handler1 = get_handler()
        handler2 = get_handler()
        
        assert handler1 is handler2
        assert isinstance(handler1, EvaluationHandler)


    @pytest.mark.asyncio
    async def test_handler_graceful_error_handling(self, handler):
        """Test handler handles errors gracefully"""
        with patch('service.evaluation.handler.MinimalOpenCLIPEvaluator') as mock_evaluator_class:
            mock_evaluator_class.side_effect = ServiceError("Critical model error", 503)
            
            request = EvaluationRequest(
                image_input="test.jpg",
                text_prompt="test prompt"
            )
            
            with pytest.raises(ServiceError):
                await handler.evaluate_single(request)

    @pytest.mark.asyncio
    async def test_handler_validation_error_handling(self, handler):
        """Test handler handles validation errors appropriately"""
        request = EvaluationRequest(
            image_input="test.jpg",
            text_prompt="test prompt",
            model_config_name="nonexistent_config"
        )
        
        with pytest.raises(ValueError, match="Unknown model config"):
            await handler.evaluate_single(request)





class TestEvaluationHandlerErrorScenarios:
    """Test handler error scenarios and edge cases"""

    @pytest.fixture
    def handler(self):
        return EvaluationHandler()

    @pytest.mark.asyncio
    async def test_evaluate_single_service_error_propagation(self, handler):
        """Test that ServiceErrors are properly propagated"""
        with patch('service.evaluation.handler.MinimalOpenCLIPEvaluator') as mock_evaluator_class:
            mock_evaluator_class.side_effect = ServiceError("Model unavailable", 503)
            
            request = EvaluationRequest(
                image_input="test.jpg",
                text_prompt="test prompt"
            )
            
            with pytest.raises(ServiceError) as exc_info:
                await handler.evaluate_single(request)
            
            assert exc_info.value.http_status == 503
            assert "Model unavailable" in str(exc_info.value)

    @pytest.mark.asyncio  
    async def test_evaluate_batch_partial_failure_handling(self, handler):
        """Test batch evaluation handles partial failures correctly"""
        with patch('service.evaluation.handler.MinimalOpenCLIPEvaluator') as mock_evaluator_class:
            mock_evaluator = Mock()
            
            # Create results with mixed success/failure
            mock_results = []
            for i in range(3):
                mock_result = Mock()
                mock_result.image_path = f"test{i}.jpg"
                mock_result.text_prompt = f"prompt {i}"
                if i == 1:  # Second item fails
                    mock_result.clip_score = 0.0
                    mock_result.error = "Processing failed"
                    mock_result.processing_time_ms = 25.0
                else:
                    mock_result.clip_score = 75.0 + i * 5
                    mock_result.error = None
                    mock_result.processing_time_ms = 100.0 + i * 10
                mock_results.append(mock_result)
            
            # Handler calls evaluate_single for each request, so mock that method
            mock_evaluator.evaluate_single = AsyncMock(side_effect=mock_results)
            mock_evaluator_class.return_value = mock_evaluator
            
            batch_request = BatchEvaluationRequest(
                evaluations=[
                    EvaluationRequest(image_input=f"test{i}.jpg", text_prompt=f"prompt {i}")
                    for i in range(3)
                ]
            )
            
            response = await handler.evaluate_batch(batch_request)
            
            assert response.total_processed == 3
            assert response.total_successful == 2
            assert response.total_failed == 1
            assert response.results[1].error == "Processing failed"
            assert response.results[0].clip_score == 75.0
            assert response.results[2].clip_score == 85.0

    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self, handler):
        """Test health check gracefully handles exceptions"""  
        with patch('service.evaluation.handler.MinimalOpenCLIPEvaluator') as mock_evaluator_class:
            mock_evaluator_class.side_effect = RuntimeError("Unexpected model error")
            
            response = await handler.health_check()
            
            assert response.status == "unhealthy"
            assert response.model_loaded is False
            # Should still return available configs from registry
            assert len(response.available_configs) >= 2