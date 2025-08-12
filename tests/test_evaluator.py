import pytest
import os
import json
from service.core import MinimalOpenCLIPEvaluator

def test_single_evaluation(fast_evaluator, sample_images, sample_prompts):
    """Test single image-text evaluation"""
    result = fast_evaluator.evaluate_single(
        sample_images[0],
        sample_prompts[0]
    )
    assert result.error is None
    assert result.clip_score > 0.0
    assert result.processing_time_ms > 0.0

def test_batch_evaluation(fast_evaluator, sample_images, sample_prompts):
    """Test batch image-text evaluation"""
    batch_results = fast_evaluator.evaluate_batch(sample_images, sample_prompts)
    assert len(batch_results) == len(sample_images)
    for result in batch_results:
        assert result.error is None
        assert result.clip_score > 0.0

def test_matched_image_caption_pairs(fast_evaluator, image_caption_pairs):
    """Test evaluation with matched image-caption pairs from Leonardo data"""
    if not image_caption_pairs:
        pytest.skip("No image-caption pairs available")
    
    for image_path, caption in image_caption_pairs:
        result = fast_evaluator.evaluate_single(image_path, caption)
        assert result.error is None
        assert result.clip_score > 0.0
        assert result.processing_time_ms > 0.0

def test_error_handling(fast_evaluator):
    """Test error handling for invalid inputs"""
    result = fast_evaluator.evaluate_single(
        "invalid_url_or_path",
        "test prompt"
    )
    assert result.clip_score == 0.0
    assert result.error is not None

def test_batch_length_mismatch(fast_evaluator):
    """Test error handling for mismatched batch lengths"""
    images = ["image1.jpg", "image2.jpg"]
    prompts = ["prompt1"]
    
    with pytest.raises(ValueError, match="Mismatch"):
        fast_evaluator.evaluate_batch(images, prompts)

def test_create_evaluators():
    """Test factory methods for creating evaluators"""
    fast_eval = MinimalOpenCLIPEvaluator.create_fast_evaluator()
    accurate_eval = MinimalOpenCLIPEvaluator.create_accurate_evaluator()
    
    assert fast_eval.model_name == 'ViT-B-32'
    assert accurate_eval.model_name == 'ViT-L-14'
