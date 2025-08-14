import os
import csv
import pytest
from pathlib import Path
from PIL import Image


@pytest.fixture
def test_data_dir():
    """Get the path to test data directory"""
    return Path(__file__).parent.parent / "data" / "samples"


@pytest.fixture
def sample_image_path(test_data_dir):
    image_files = [f for f in test_data_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not image_files:
        pytest.skip("No image files found in samples test data")
    return str(image_files[0])


@pytest.fixture
def sample_images(test_data_dir):
    image_files = [f for f in test_data_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not image_files:
        pytest.skip("No image files found in samples test data")
    return [str(f) for f in image_files[:3]]  # Use first 3 images


@pytest.fixture
def sample_prompts(test_data_dir):
    """Load sample prompts from samples test data"""
    csv_file = test_data_dir / "challenge_set.csv"
    if csv_file.exists():
        prompts = []
        with csv_file.open('r', encoding='utf-8') as f:
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
        "modern architectural building with glass facade"
    ]


@pytest.fixture
def image_caption_pairs(test_data_dir):
    csv_file = test_data_dir / "challenge_set.csv"
    if csv_file.exists():
        pairs = []
        with csv_file.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'url' in row and 'caption' in row and row['url'] and row['caption']:
                    pairs.append((row['url'], row['caption']))
        if pairs:
            return pairs[:3]  # Use first 3 pairs for testing
    
    return []


def extract_user_id_from_url(url):
    """Extract the user ID (first UUID) from Leonardo AI URL"""
    import re
    uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
    matches = re.findall(uuid_pattern, url, re.IGNORECASE)
    return matches[0] if matches else None


@pytest.fixture
def matched_image_caption_pairs(test_data_dir):
    """Load matched image file paths with their captions using UUID extraction from URLs"""
    csv_file = test_data_dir / "challenge_set.csv"
    pairs = []
    
    if csv_file.exists():
        # Get all available image files
        image_files = {f.stem: str(f) for f in test_data_dir.iterdir() 
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
        
        # Match with captions from CSV using UUID extraction
        with csv_file.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'url' in row and 'caption' in row and row['caption']:
                    # Extract UUID from URL to match with local files
                    user_id = extract_user_id_from_url(row['url'])
                    if user_id and user_id in image_files:
                        pairs.append((image_files[user_id], row['caption']))
    
    return pairs[:5] if pairs else []  # Return first 5 matched pairs


@pytest.fixture
def test_pil_image():
    """Create a simple test PIL image for testing"""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def invalid_image_paths():
    """Common invalid image paths for testing error handling"""
    return [
        "non_existent_file.jpg",
        "/path/that/does/not/exist.png",
        "invalid_format.txt",
        "",
        None
    ]
