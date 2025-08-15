from unittest.mock import patch, Mock, AsyncMock
import asyncio
import pytest
from PIL import Image
import httpx

from service.core.ml.preprocessing.image_loader import ImageLoader
from service.core.exceptions import ImageProcessingError, NetworkError


@pytest.fixture
def image_loader():
    """Create ImageLoader instance for testing"""
    return ImageLoader()


@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing"""
    return Image.new('RGB', (100, 100), color='red')


@pytest.fixture
def sample_rgba_image():
    """Create a sample RGBA image for testing"""
    return Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary image file for testing"""
    image_path = tmp_path / "test_image.jpg"
    image = Image.new('RGB', (50, 50), color='blue')
    image.save(image_path)
    return str(image_path)


class TestImageLoader:
    """Test image loader functionality"""


    @pytest.mark.parametrize("input_str,is_url", [
        ("https://example.com/image.jpg", True),
        ("/path/to/image.jpg", False),
        ("image.jpg", False)
    ])
    def test_url_detection(self, input_str, is_url):
        """Test URL detection"""
        assert ImageLoader._is_url(input_str) == is_url

    def test_load_local_image_success(self, image_loader, sample_image_path):
        """Test loading local image file successfully"""
        loaded_image = image_loader.load_image_sync(sample_image_path)
        assert isinstance(loaded_image, Image.Image)
        assert loaded_image.mode == 'RGB'
        assert loaded_image.size == (50, 50)

    def test_load_nonexistent_file_error(self, image_loader):
        """Test error handling for non-existent files"""
        with pytest.raises(ImageProcessingError, match="Failed to load image from file"):
            image_loader.load_image_sync("non_existent_file.jpg")

    @pytest.mark.asyncio
    async def test_load_local_image_async_success(self, image_loader, sample_image_path):
        """Test async loading of local image file"""
        loaded_image = await image_loader.load_image(sample_image_path)
        assert isinstance(loaded_image, Image.Image)
        assert loaded_image.mode == 'RGB'
        assert loaded_image.size == (50, 50)


    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_load_url_success(self, mock_get, sample_image_path):
        """Test loading image from URL successfully"""
        # Read actual image bytes
        with open(sample_image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.content = image_bytes
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        async with ImageLoader() as loader:
            loaded_image = await loader.load_image("https://example.com/test.jpg")
            
            assert isinstance(loaded_image, Image.Image)
            assert loaded_image.mode == 'RGB'
        mock_get.assert_called_once()


    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_load_url_errors(self, mock_get):
        """Test URL loading error handling"""
        mock_get.side_effect = httpx.RequestError("Connection failed")
        
        async with ImageLoader() as loader:
            with pytest.raises(NetworkError, match="Failed to fetch image from URL"):
                await loader.load_image("https://example.com/test.jpg")

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_load_url_invalid_image_data(self, mock_get):
        """Test handling invalid image data from URL"""
        mock_response = Mock()
        mock_response.content = b"not an image"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        async with ImageLoader() as loader:
            with pytest.raises(ImageProcessingError, match="Failed to process image from URL"):
                await loader.load_image("https://example.com/invalid.jpg")


class TestImageLoaderImageProcessing:
    """Test image loader image processing functionality"""

    @pytest.mark.parametrize("mode,expected_mode", [
        ('RGBA', 'RGB'),
        ('L', 'RGB'),
        ('RGB', 'RGB')
    ])
    def test_image_mode_conversion(self, image_loader, tmp_path, mode, expected_mode):
        """Test images are converted to RGB mode"""
        image = Image.new(mode, (50, 50), color=128 if mode == 'L' else (255, 0, 0, 128) if mode == 'RGBA' else 'red')
        image_path = tmp_path / f"test_{mode.lower()}.png"
        image.save(image_path)
        
        loaded_image = image_loader.load_image_sync(str(image_path))
        assert loaded_image.mode == expected_mode




