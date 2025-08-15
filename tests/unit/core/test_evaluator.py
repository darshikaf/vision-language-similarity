from unittest.mock import patch, Mock, AsyncMock
import pytest
from PIL import Image

from service.core.ml.engines.openclip_evaluator import MinimalOpenCLIPEvaluator
from service.core.exceptions import ValidationError
from service.core.config import CLIPModelSpec


@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    return Image.new('RGB', (224, 224), color='red')


@pytest.fixture
def sample_text_prompt():
    """Sample text prompt for testing"""
    return "A red colored image"


@pytest.fixture
def sample_batch_images():
    """Create sample batch of images for testing"""
    return [
        Image.new('RGB', (224, 224), color='red'),
        Image.new('RGB', (224, 224), color='blue'),
        Image.new('RGB', (224, 224), color='green')
    ]


@pytest.fixture
def sample_batch_prompts():
    """Sample batch of text prompts for testing"""
    return [
        "A red colored image",
        "A blue colored image", 
        "A green colored image"
    ]


@pytest.fixture
def mock_clip_model_spec():
    """Mock CLIP model specification for testing"""
    return CLIPModelSpec(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        description="Test model",
        memory_gb=2.0,
        avg_inference_time_ms=100.0,
        accuracy_score=0.85,
        enabled=True
    )


class TestMinimalOpenCLIPEvaluator:
    """Test MinimalOpenCLIPEvaluator functionality"""

    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    def test_evaluator_initialization_with_config_name(self, mock_create_model):
        """Test evaluator initialization with model config name"""
        mock_similarity_model = Mock()
        mock_create_model.return_value = mock_similarity_model
        
        evaluator = MinimalOpenCLIPEvaluator(model_config_name="fast")
        
        assert evaluator.similarity_model == mock_similarity_model
        mock_create_model.assert_called_once_with("fast", device=None)

    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    def test_evaluator_initialization_with_spec(self, mock_create_model, mock_clip_model_spec):
        """Test evaluator initialization with model spec"""
        mock_similarity_model = Mock()
        mock_create_model.return_value = mock_similarity_model
        
        # Pass model_spec as a kwarg since evaluator passes **model_kwargs to factory
        evaluator = MinimalOpenCLIPEvaluator(model_config_name="custom", model_spec=mock_clip_model_spec)
        
        assert evaluator.similarity_model == mock_similarity_model
        mock_create_model.assert_called_once_with("custom", device=None, model_spec=mock_clip_model_spec)

    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    def test_evaluator_initialization_requires_config(self, mock_create_model):
        """Test evaluator uses default config when none specified"""
        mock_similarity_model = Mock()
        mock_create_model.return_value = mock_similarity_model
        
        # Evaluator should use default "fast" config
        evaluator = MinimalOpenCLIPEvaluator()
        
        assert evaluator.model_config_name == "fast"
        assert evaluator.similarity_model == mock_similarity_model
        mock_create_model.assert_called_once_with("fast", device=None)


    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_evaluate_single_success(self, mock_create_model, sample_image, sample_text_prompt):
        """Test successful single evaluation"""
        # Mock similarity model
        mock_similarity_model = Mock()
        mock_similarity_model.compute_similarity = AsyncMock(return_value=(0.85, 50.0))  # (score, inference_time)
        mock_similarity_model.model_name = "ViT-B-32"
        mock_create_model.return_value = mock_similarity_model
        
        # Create mock image processor
        mock_image_processor = AsyncMock()
        mock_image_processor.load_image = AsyncMock(return_value=sample_image)
        mock_image_processor.determine_source_type = Mock(return_value="file")
        
        # Create mock metrics recorder (disabled)
        mock_metrics_recorder = Mock()
        mock_metrics_recorder.record_success_metrics = AsyncMock()
        mock_metrics_recorder.enabled = False
        
        evaluator = MinimalOpenCLIPEvaluator(
            model_config_name="fast",
            image_processor=mock_image_processor,
            metrics_recorder=mock_metrics_recorder
        )
        
        result = await evaluator.evaluate_single("test_image.jpg", sample_text_prompt)
        
        assert result.image_path == "test_image.jpg"
        assert result.text_prompt == sample_text_prompt
        assert result.clip_score == 0.85  # Raw similarity score from mock
        assert result.processing_time_ms > 0
        assert result.error is None

    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_evaluate_single_image_loading_error(self, mock_create_model):
        """Test single evaluation with image loading error"""
        # Mock similarity model
        mock_similarity_model = Mock()
        mock_similarity_model.model_name = "ViT-B-32"
        mock_create_model.return_value = mock_similarity_model
        
        # Create mock image processor that raises error
        mock_image_processor = AsyncMock()
        mock_image_processor.load_image = AsyncMock(side_effect=Exception("Image loading failed"))
        
        # Create mock metrics recorder 
        mock_metrics_recorder = Mock()
        mock_metrics_recorder.extract_error_type = Mock(return_value=None)
        mock_metrics_recorder.record_error_metrics = Mock()
        
        evaluator = MinimalOpenCLIPEvaluator(
            model_config_name="fast",
            image_processor=mock_image_processor,
            metrics_recorder=mock_metrics_recorder
        )
        
        result = await evaluator.evaluate_single("bad_image.jpg", "test prompt")
        
        assert result.image_path == "bad_image.jpg"
        assert result.text_prompt == "test prompt"
        assert result.clip_score == 0.0
        assert result.error is not None
        assert "Image loading failed" in result.error

    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_evaluate_single_similarity_computation_error(self, mock_create_model, sample_image):
        """Test single evaluation with similarity computation error"""
        # Mock similarity model to raise error
        mock_similarity_model = Mock()
        mock_similarity_model.compute_similarity = AsyncMock(side_effect=Exception("Model inference failed"))
        mock_similarity_model.model_name = "ViT-B-32"
        mock_create_model.return_value = mock_similarity_model
        
        # Create mock image processor
        mock_image_processor = AsyncMock()
        mock_image_processor.load_image = AsyncMock(return_value=sample_image)
        mock_image_processor.determine_source_type = Mock(return_value="file")
        
        # Create mock metrics recorder 
        mock_metrics_recorder = Mock()
        mock_metrics_recorder.extract_error_type = Mock(return_value=None)
        mock_metrics_recorder.record_error_metrics = Mock()
        
        evaluator = MinimalOpenCLIPEvaluator(
            model_config_name="fast",
            image_processor=mock_image_processor,
            metrics_recorder=mock_metrics_recorder
        )
        
        result = await evaluator.evaluate_single("test_image.jpg", "test prompt")
        
        assert result.clip_score == 0.0
        assert result.error is not None
        assert "Model inference failed" in result.error

    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_evaluate_batch_success(self, mock_create_model, sample_batch_images, sample_batch_prompts):
        """Test successful batch evaluation"""
        # Mock similarity model
        mock_similarity_model = Mock()
        mock_similarity_model.compute_batch_similarity = AsyncMock(return_value=([0.8, 0.7, 0.9], 45.0))  # (scores, inference_time)
        mock_create_model.return_value = mock_similarity_model
        
        evaluator = MinimalOpenCLIPEvaluator(model_config_name="fast")
        
        results = await evaluator.evaluate_batch(sample_batch_images, sample_batch_prompts)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.clip_score in [0.8, 0.7, 0.9]  # Raw similarity scores from mock
            assert result.processing_time_ms > 0
            assert result.error is None

    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_evaluate_batch_length_mismatch(self, mock_create_model, sample_batch_images):
        """Test batch evaluation with mismatched lengths"""
        mock_similarity_model = Mock()
        mock_create_model.return_value = mock_similarity_model
        
        evaluator = MinimalOpenCLIPEvaluator(model_config_name="fast")
        
        # Mismatched lengths: 3 images, 2 prompts
        mismatched_prompts = ["prompt1", "prompt2"]
        
        with pytest.raises(ValidationError, match="Mismatch"):
            await evaluator.evaluate_batch(sample_batch_images, mismatched_prompts)

    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_evaluate_batch_empty_inputs(self, mock_create_model):
        """Test batch evaluation with empty inputs"""
        mock_similarity_model = Mock()
        mock_create_model.return_value = mock_similarity_model
        
        evaluator = MinimalOpenCLIPEvaluator(model_config_name="fast")
        
        results = await evaluator.evaluate_batch([], [])
        
        assert len(results) == 0

    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_evaluate_batch_with_batch_size(self, mock_create_model, sample_batch_images, sample_batch_prompts):
        """Test batch evaluation with custom batch size"""
        # Mock similarity model to track calls
        mock_similarity_model = Mock()
        mock_similarity_model.compute_batch_similarity = AsyncMock(return_value=([0.8, 0.7, 0.9], 45.0))  # (scores, inference_time)
        mock_create_model.return_value = mock_similarity_model
        
        evaluator = MinimalOpenCLIPEvaluator(model_config_name="fast")
        
        results = await evaluator.evaluate_batch(sample_batch_images, sample_batch_prompts, batch_size=2)
        
        assert len(results) == 3
        # Should still process all items regardless of batch size
        for result in results:
            assert result.error is None


class TestMinimalOpenCLIPEvaluatorEdgeCases:
    """Test evaluator edge cases and error scenarios"""

    @pytest.mark.parametrize("model_config", ["fast", "accurate"])
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    def test_evaluator_with_different_configs(self, mock_create_model, model_config):
        """Test evaluator with different model configurations"""
        mock_similarity_model = Mock()
        mock_create_model.return_value = mock_similarity_model
        
        evaluator = MinimalOpenCLIPEvaluator(model_config_name=model_config)
        
        assert evaluator.similarity_model == mock_similarity_model
        mock_create_model.assert_called_once_with(model_config, device=None)

    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_evaluate_batch_partial_failure(self, mock_create_model):
        """Test batch evaluation with partial failures"""
        # Mock similarity model to fail on second item
        mock_similarity_model = Mock()
        mock_similarity_model.compute_batch_similarity = AsyncMock(side_effect=[
            Exception("Batch processing failed")
        ])
        mock_create_model.return_value = mock_similarity_model
        
        evaluator = MinimalOpenCLIPEvaluator(model_config_name="fast")
        
        images = [Image.new('RGB', (50, 50), 'red') for _ in range(2)]
        prompts = ["prompt1", "prompt2"]
        
        results = await evaluator.evaluate_batch(images, prompts)
        
        # Should return results with errors
        assert len(results) == 2
        for result in results:
            assert result.clip_score == 0.0
            assert result.error is not None

    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_evaluate_single_with_pil_image(self, mock_create_model, sample_image):
        """Test single evaluation with PIL Image instead of path"""
        mock_similarity_model = Mock()
        mock_similarity_model.compute_similarity = AsyncMock(return_value=(0.75, 50.0))  # (score, inference_time)
        mock_create_model.return_value = mock_similarity_model
        
        evaluator = MinimalOpenCLIPEvaluator(model_config_name="fast")
        
        result = await evaluator.evaluate_single(sample_image, "test prompt")
        
        # The evaluator converts PIL image to its string representation
        assert result.image_path == str(sample_image)
        assert result.clip_score == 0.75  # Raw similarity score
        assert result.error is None

    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_clip_score_scaling(self, mock_create_model, sample_image):
        """Test CLIP score handling with different similarity values"""
        mock_similarity_model = Mock()
        # Test different similarity values
        test_similarities = [0.0, 0.5, 1.0, -0.1]  # Including negative value
        expected_scores = [0.0, 0.5, 1.0, -0.1]  # Raw values returned as-is
        
        for similarity, expected_score in zip(test_similarities, expected_scores):
            mock_similarity_model.compute_similarity = AsyncMock(return_value=(similarity, 50.0))  # (score, inference_time)
            mock_create_model.return_value = mock_similarity_model
            
            evaluator = MinimalOpenCLIPEvaluator(model_config_name="fast")
            result = await evaluator.evaluate_single(sample_image, "test prompt")
            
            assert result.clip_score == expected_score


class TestMinimalOpenCLIPEvaluatorIntegration:
    """Test evaluator integration scenarios"""

    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_mixed_evaluation_operations(self, mock_create_model, sample_image):
        """Test mixing single and batch operations"""
        mock_similarity_model = Mock()
        mock_similarity_model.compute_similarity = AsyncMock(return_value=(0.8, 50.0))  # (score, inference_time)
        mock_similarity_model.compute_batch_similarity = AsyncMock(return_value=([0.7, 0.9], 45.0))  # (scores, inference_time)
        mock_create_model.return_value = mock_similarity_model
        
        evaluator = MinimalOpenCLIPEvaluator(model_config_name="fast")
        
        # Single evaluation
        single_result = await evaluator.evaluate_single(sample_image, "prompt")
        assert single_result.clip_score == 0.8  # Raw similarity score
        
        # Batch evaluation
        batch_results = await evaluator.evaluate_batch([sample_image, sample_image], ["prompt1", "prompt2"])
        assert len(batch_results) == 2
        assert batch_results[0].clip_score == 0.7  # Raw similarity scores
        assert batch_results[1].clip_score == 0.9

    @pytest.mark.asyncio
    @patch('service.core.ml.models.factory.SimilarityModelFactory.create_model')
    async def test_concurrent_evaluations(self, mock_create_model, sample_image):
        """Test concurrent evaluation operations"""
        import asyncio
        
        mock_similarity_model = Mock()
        mock_similarity_model.compute_similarity = AsyncMock(return_value=(0.8, 50.0))  # (score, inference_time)
        mock_create_model.return_value = mock_similarity_model
        
        evaluator = MinimalOpenCLIPEvaluator(model_config_name="fast")
        
        # Run multiple evaluations concurrently
        tasks = [
            evaluator.evaluate_single(sample_image, f"prompt {i}")
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert result.clip_score == 0.8  # Raw similarity score
            assert result.error is None