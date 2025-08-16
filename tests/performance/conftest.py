"""CLAUDE generated: Performance test fixtures and configuration."""

import asyncio
import gc
import time
from typing import Generator, List
import pytest
from pathlib import Path
from PIL import Image

from service.core.ml.engines.openclip_evaluator import OpenCLIPEvaluator


@pytest.fixture
def test_data_dir():
    """Get the path to test data directory"""
    return Path(__file__).parent.parent / "data" / "samples"


@pytest.fixture
def performance_image(test_data_dir) -> str:
    """Load a real test image for performance benchmarks."""
    image_files = [f for f in test_data_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not image_files:
        pytest.skip("No image files found in test data for performance testing")
    return str(image_files[0])


@pytest.fixture
def performance_images_small(test_data_dir) -> List[str]:
    """Load small batch of real test images for performance testing."""
    image_files = [f for f in test_data_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if len(image_files) < 3:
        pytest.skip("Need at least 3 image files for small batch performance testing")
    # Repeat images to get 5 items for consistency with existing tests
    selected_images = image_files[:3]
    return [str(selected_images[i % len(selected_images)]) for i in range(5)]


@pytest.fixture
def performance_images_medium(test_data_dir) -> List[str]:
    """Create medium batch of test image paths for performance testing."""
    image_files = [f for f in test_data_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not image_files:
        pytest.skip("No image files found in test data for performance testing")
    # Repeat available images to get 25 items
    return [str(image_files[i % len(image_files)]) for i in range(25)]


@pytest.fixture
def performance_images_large(test_data_dir) -> List[str]:
    """Create large batch of test image paths for performance testing."""
    image_files = [f for f in test_data_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not image_files:
        pytest.skip("No image files found in test data for performance testing")
    # Repeat available images to get 100 items
    return [str(image_files[i % len(image_files)]) for i in range(100)]


@pytest.fixture
def performance_prompts_small(test_data_dir) -> List[str]:
    """Load real captions for small batch performance testing."""
    import csv
    csv_file = test_data_dir / "challenge_set.csv"
    prompts = []
    
    if csv_file.exists():
        with csv_file.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'caption' in row and row['caption']:
                    prompts.append(row['caption'])
    
    if len(prompts) < 3:
        # Fallback to generic prompts if not enough real data
        prompts = [
            "2 friendly real estate agent standing with curly hair and glasses",
            "A pile of bricks on the floor",
            "Cute koala bear eating bamboo in a jungle",
            "Modern architectural building with glass facade",
            "Vector pattern with pastel colors in kawaii cartoon style"
        ]
    
    # Repeat prompts to get exactly 5 items for consistency
    return [prompts[i % len(prompts)] for i in range(5)]


@pytest.fixture
def performance_prompts_medium(test_data_dir) -> List[str]:
    """Create medium batch of test prompts."""
    import csv
    csv_file = test_data_dir / "challenge_set.csv"
    prompts = []
    
    if csv_file.exists():
        with csv_file.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'caption' in row and row['caption']:
                    prompts.append(row['caption'])
    
    if not prompts:
        prompts = [
            "2 friendly real estate agent standing with curly hair and glasses",
            "A pile of bricks on the floor", 
            "Cute koala bear eating bamboo in a jungle"
        ]
    
    # Repeat prompts to get 25 items
    return [prompts[i % len(prompts)] for i in range(25)]


@pytest.fixture
def performance_prompts_large(test_data_dir) -> List[str]:
    """Create large batch of test prompts."""
    import csv
    csv_file = test_data_dir / "challenge_set.csv"
    prompts = []
    
    if csv_file.exists():
        with csv_file.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'caption' in row and row['caption']:
                    prompts.append(row['caption'])
    
    if not prompts:
        prompts = [
            "2 friendly real estate agent standing with curly hair and glasses",
            "A pile of bricks on the floor",
            "Cute koala bear eating bamboo in a jungle"
        ]
    
    # Repeat prompts to get 100 items
    return [prompts[i % len(prompts)] for i in range(100)]


@pytest.fixture
def fast_evaluator() -> Generator[OpenCLIPEvaluator, None, None]:
    """Create a fast evaluator for performance testing."""
    evaluator = OpenCLIPEvaluator(model_config_name="fast")  
    yield evaluator
    # Cleanup
    del evaluator
    gc.collect()


@pytest.fixture
def accurate_evaluator() -> Generator[OpenCLIPEvaluator, None, None]:
    """Create an accurate evaluator for performance testing."""
    evaluator = OpenCLIPEvaluator(model_config_name="accurate")
    yield evaluator
    # Cleanup
    del evaluator
    gc.collect()


@pytest.fixture
def multiple_fast_evaluators() -> Generator[List[OpenCLIPEvaluator], None, None]:
    """Create multiple evaluators with same config for memory testing."""
    evaluators = []
    for i in range(3):
        evaluators.append(OpenCLIPEvaluator(model_config_name="fast"))
    
    yield evaluators
    
    # Cleanup
    for evaluator in evaluators:
        del evaluator
    evaluators.clear()
    gc.collect()


@pytest.fixture
def mixed_evaluators() -> Generator[List[OpenCLIPEvaluator], None, None]:
    """Create evaluators with different configs for memory testing."""
    evaluators = []
    evaluators.append(OpenCLIPEvaluator(model_config_name="fast"))
    evaluators.append(OpenCLIPEvaluator(model_config_name="accurate"))
    evaluators.append(OpenCLIPEvaluator(model_config_name="fast"))
    
    yield evaluators
    
    # Cleanup
    for evaluator in evaluators:
        del evaluator
    evaluators.clear()
    gc.collect()


@pytest.fixture(autouse=True)
def cleanup_after_performance_test():
    """Ensure cleanup after each performance test."""
    yield
    # Force garbage collection after each test
    gc.collect()
    # Small delay to allow cleanup
    time.sleep(0.1)
