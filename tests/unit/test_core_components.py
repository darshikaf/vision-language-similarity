import pytest
import torch
from service.core import DeviceManager, ImageLoader, MinimalOpenCLIPEvaluator, ModelConfig


class TestDeviceManager:
    """Essential device manager tests"""

    def test_get_optimal_device_returns_device(self):
        """Test device selection returns a valid torch device"""
        device = DeviceManager.get_optimal_device()
        assert isinstance(device, torch.device)

    def test_get_optimal_device_explicit_cpu(self):
        """Test explicit CPU device selection"""
        cpu_device = DeviceManager.get_optimal_device("cpu")
        assert cpu_device.type == "cpu"

    def test_get_optimal_precision_cpu(self):
        """Test precision selection for CPU"""
        device = torch.device("cpu")
        precision = DeviceManager.get_optimal_precision(device)
        assert precision == "fp32"
    
    def test_get_optimal_precision_cuda(self):
        """Test precision selection for CUDA"""
        device = torch.device("cuda")
        precision = DeviceManager.get_optimal_precision(device)
        assert precision == "fp16"


class TestImageLoader:
    """Essential image loader tests"""

    def test_is_url_detection(self):
        """Test URL vs path detection"""
        assert ImageLoader._is_url("https://example.com/image.jpg")
        assert ImageLoader._is_url("http://example.com/image.jpg")
        assert not ImageLoader._is_url("/path/to/image.jpg")
        assert not ImageLoader._is_url("local/path/image.jpg")

    def test_load_local_image_file(self, sample_image_path):
        """Test loading local image file from test data"""
        loader = ImageLoader()
        loaded_image = loader.load_image_sync(sample_image_path)
        assert loaded_image.mode == 'RGB'
        assert loaded_image.size[0] > 0

    def test_load_invalid_file_error(self):
        """Test error handling for non-existent files"""
        loader = ImageLoader()
        with pytest.raises(ValueError, match="Failed to load image"):
            loader.load_image_sync("non_existent_file.jpg")


class TestModelConfig:
    """Essential model configuration tests"""

    def test_fast_config(self):
        config = ModelConfig.get_fast_config()
        assert config.model_name == "ViT-B-32"
        assert config.pretrained == "laion2b_s34b_b79k"

    def test_accurate_config(self):
        config = ModelConfig.get_accurate_config()
        assert config.model_name == "ViT-L-14"
        assert config.pretrained == "laion2b_s32b_b82k"


class TestMinimalOpenCLIPEvaluator:
    """Essential evaluator tests"""

    def test_factory_methods_create_correct_models(self):
        fast_eval = MinimalOpenCLIPEvaluator.create_fast_evaluator()
        accurate_eval = MinimalOpenCLIPEvaluator.create_accurate_evaluator()
        
        assert fast_eval.similarity_model.model_name == "ViT-B-32"
        assert accurate_eval.similarity_model.model_name == "ViT-L-14"

    @pytest.mark.asyncio
    async def test_evaluate_single_success(self, fast_evaluator, sample_image_path, sample_prompts):
        result = await fast_evaluator.evaluate_single(sample_image_path, sample_prompts[0])
        
        assert hasattr(result, 'clip_score')
        assert hasattr(result, 'processing_time_ms')
        assert hasattr(result, 'error')
        
        if result.error is None:
            assert result.clip_score >= 0.0
            assert result.processing_time_ms > 0.0

    @pytest.mark.asyncio
    async def test_evaluate_batch_length_validation(self, fast_evaluator):
        images = ["image1.jpg", "image2.jpg"]
        prompts = ["prompt1"]  # Mismatched length
        
        with pytest.raises(ValueError, match="Mismatch"):
            await fast_evaluator.evaluate_batch(images, prompts)
