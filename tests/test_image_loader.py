import pytest
import os
import csv
from PIL import Image
from service.core import ImageLoader

@pytest.fixture
def sample_leonardo_url(test_data_dir):
    """Get a sample Leonardo URL from the CSV"""
    csv_file = os.path.join(test_data_dir, "challenge_set.csv")
    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'url' in row and row['url']:
                    return row['url']
    return None

def test_is_url():
    """Test URL detection"""
    assert ImageLoader._is_url("https://example.com/image.jpg")
    assert ImageLoader._is_url("http://example.com/image.jpg")
    assert not ImageLoader._is_url("/path/to/image.jpg")
    assert not ImageLoader._is_url("image.jpg")

def test_load_pil_image():
    """Test loading PIL Image directly"""
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='red')
    loaded_image = ImageLoader.load_image(test_image)
    
    assert isinstance(loaded_image, Image.Image)
    assert loaded_image.mode == 'RGB'

def test_load_local_file(sample_image_path):
    """Test loading local image file from Leonardo test data"""
    loaded_image = ImageLoader.load_image(sample_image_path)
    
    assert isinstance(loaded_image, Image.Image)
    assert loaded_image.mode == 'RGB'

def test_load_leonardo_url(sample_leonardo_url):
    """Test loading image from Leonardo CDN URL"""
    if not sample_leonardo_url:
        pytest.skip("No Leonardo URL available")
    
    try:
        loaded_image = ImageLoader.load_image(sample_leonardo_url)
        assert isinstance(loaded_image, Image.Image)
        assert loaded_image.mode == 'RGB'
    except ValueError as e:
        # If URL fails due to network issues, that's acceptable for testing
        pytest.skip(f"Network error loading Leonardo URL: {e}")

def test_load_invalid_file():
    """Test loading non-existent file"""
    with pytest.raises(ValueError, match="Failed to load image"):
        ImageLoader.load_image("non_existent_file.jpg")

def test_load_invalid_url():
    """Test loading from invalid URL"""
    with pytest.raises(ValueError, match="Failed to fetch image"):
        ImageLoader.load_image("https://invalid-domain-12345.com/image.jpg")
