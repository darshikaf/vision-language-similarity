import pytest
import sys
import os
import csv
from pathlib import Path

# Add the service directory to the Python path
service_dir = Path(__file__).parent.parent
sys.path.insert(0, str(service_dir))

from service.core import MinimalOpenCLIPEvaluator

@pytest.fixture(scope="session")
def fast_evaluator():
    """Session-scoped fixture for fast evaluator to avoid reloading model"""
    return MinimalOpenCLIPEvaluator.create_fast_evaluator()

@pytest.fixture(scope="session")
def accurate_evaluator():
    """Session-scoped fixture for accurate evaluator"""
    return MinimalOpenCLIPEvaluator.create_accurate_evaluator()

@pytest.fixture
def test_data_dir():
    """Get the path to test data directory"""
    return os.path.join(os.path.dirname(__file__), "data", "Leonardo")

@pytest.fixture
def sample_image_path(test_data_dir):
    """Get path to a sample image from Leonardo test data"""
    image_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        pytest.skip("No image files found in Leonardo test data")
    return os.path.join(test_data_dir, image_files[0])

@pytest.fixture
def sample_images(test_data_dir):
    """Load sample images from Leonardo test data"""
    image_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        pytest.skip("No image files found in Leonardo test data")
    return [os.path.join(test_data_dir, f) for f in image_files[:3]]  # Use first 3 images

@pytest.fixture
def sample_prompts(test_data_dir):
    """Load sample prompts from Leonardo test data"""
    csv_file = os.path.join(test_data_dir, "challenge_set.csv")
    if os.path.exists(csv_file):
        prompts = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'caption' in row and row['caption']:
                    prompts.append(row['caption'])
        if prompts:
            return prompts[:3]  # Use first 3 prompts
    
    # Fallback to generic prompts if CSV reading fails
    return [
        "2 friendly real estate agent standing. one with curly brown hair and glasses",
        "vector pattern, pastel colors, in style kawai cartoon",
    ]

@pytest.fixture
def image_caption_pairs(test_data_dir):
    """Load matched image-caption pairs from Leonardo test data"""
    csv_file = os.path.join(test_data_dir, "challenge_set.csv")
    if os.path.exists(csv_file):
        pairs = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'url' in row and 'caption' in row and row['url'] and row['caption']:
                    pairs.append((row['url'], row['caption']))
        if pairs:
            return pairs[:3]  # Use first 3 pairs for testing
    
    return []