from unittest.mock import patch, Mock, AsyncMock
import asyncio
import pytest
from PIL import Image
import httpx

from service.core.image_loader import ImageLoader
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

    def test_initialization(self, image_loader):
        """Test ImageLoader initializes correctly"""
        assert image_loader is not None
        assert hasattr(image_loader, 'load_image')
        assert hasattr(image_loader, 'load_image_sync')

    @pytest.mark.parametrize("url", [
        "https://example.com/image.jpg",
        "http://example.com/image.png", 
        "https://subdomain.domain.com/path/to/image.webp"
    ])
    def test_is_url_detection_valid_urls(self, url):
        """Test URL detection for valid URLs"""
        assert ImageLoader._is_url(url)

    @pytest.mark.parametrize("path", [
        "/path/to/image.jpg",
        "local/path/image.jpg",
        "relative_image.png",
        "C:\\Windows\\image.jpg",
        "image.jpg"
    ])
    def test_is_url_detection_file_paths(self, path):
        """Test URL detection correctly identifies file paths"""
        assert not ImageLoader._is_url(path)

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
    async def test_load_nonexistent_file_async_error(self, image_loader):
        """Test async error handling for non-existent files"""
        with pytest.raises(ImageProcessingError, match="Failed to load image from file"):
            await image_loader.load_image("non_existent_file.jpg")

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
    async def test_load_url_http_error(self, mock_get):
        """Test handling HTTP errors when loading URLs"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.side_effect = httpx.HTTPStatusError("404 Not Found", request=Mock(), response=mock_response)
        
        async with ImageLoader() as loader:
            with pytest.raises(NetworkError, match="Failed to fetch image from URL"):
                await loader.load_image("https://example.com/nonexistent.jpg")

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_load_url_network_error(self, mock_get):
        """Test handling network errors when loading URLs"""
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

    def test_rgba_to_rgb_conversion(self, image_loader, tmp_path):
        """Test RGBA images are converted to RGB"""
        # Create RGBA image
        rgba_image = Image.new('RGBA', (50, 50), color=(255, 0, 0, 128))
        rgba_path = tmp_path / "rgba_image.png"
        rgba_image.save(rgba_path)
        
        loaded_image = image_loader.load_image_sync(str(rgba_path))
        assert loaded_image.mode == 'RGB'
        assert loaded_image.size == (50, 50)

    def test_grayscale_to_rgb_conversion(self, image_loader, tmp_path):
        """Test grayscale images are converted to RGB"""
        # Create grayscale image
        gray_image = Image.new('L', (50, 50), color=128)
        gray_path = tmp_path / "gray_image.png"
        gray_image.save(gray_path)
        
        loaded_image = image_loader.load_image_sync(str(gray_path))
        assert loaded_image.mode == 'RGB'
        assert loaded_image.size == (50, 50)

    def test_rgb_image_unchanged(self, image_loader, sample_image_path):
        """Test RGB images remain unchanged"""
        loaded_image = image_loader.load_image_sync(sample_image_path)
        assert loaded_image.mode == 'RGB'

    @pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
    def test_different_image_formats(self, image_loader, tmp_path, image_format):
        """Test loading different image formats"""
        # Create image in specified format
        image = Image.new('RGB', (30, 30), color='green')
        
        if image_format == "JPEG":
            image_path = tmp_path / "test.jpg"
        elif image_format == "PNG":
            image_path = tmp_path / "test.png"
        elif image_format == "WEBP":
            image_path = tmp_path / "test.webp"
        
        try:
            image.save(image_path, format=image_format)
            loaded_image = image_loader.load_image_sync(str(image_path))
            assert loaded_image.mode == 'RGB'
            assert loaded_image.size == (30, 30)
        except OSError:
            # Skip if format not supported in test environment
            pytest.skip(f"{image_format} format not supported in test environment")


class TestImageLoaderErrorHandling:
    """Test image loader error handling scenarios"""

    def test_corrupted_file_handling(self, image_loader, tmp_path):
        """Test handling of corrupted image files"""
        corrupted_path = tmp_path / "corrupted.jpg"
        corrupted_path.write_bytes(b"not a valid image file")
        
        with pytest.raises(ImageProcessingError, match="Failed to load image from file"):
            image_loader.load_image_sync(str(corrupted_path))

    def test_empty_file_handling(self, image_loader, tmp_path):
        """Test handling of empty files"""
        empty_path = tmp_path / "empty.jpg"
        empty_path.write_bytes(b"")
        
        with pytest.raises(ImageProcessingError, match="Failed to load image from file"):
            image_loader.load_image_sync(str(empty_path))

    def test_permission_denied_handling(self, image_loader):
        """Test handling of permission denied errors"""
        # Use a path that would typically cause permission issues
        restricted_path = "/root/restricted_image.jpg"
        
        with pytest.raises(ImageProcessingError, match="Failed to load image from file"):
            image_loader.load_image_sync(restricted_path)

    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.get')
    async def test_url_timeout_handling(self, mock_get):
        """Test handling of request timeouts"""
        mock_get.side_effect = httpx.TimeoutException("Request timed out")
        
        async with ImageLoader() as loader:
            with pytest.raises(NetworkError, match="Failed to fetch image from URL"):
                await loader.load_image("https://slow-server.com/image.jpg")


class TestImageLoaderIntegration:
    """Test image loader integration scenarios"""

    def test_mixed_load_operations(self, sample_image_path):
        """Test mixing sync and async operations"""
        loader = ImageLoader()
        # Load same image both ways - sync first
        sync_image = loader.load_image_sync(sample_image_path)
        
        # Then async in a separate event loop context
        async def load_async():
            async with ImageLoader() as async_loader:
                return await async_loader.load_image(sample_image_path)
        
        async_image = asyncio.run(load_async())
        
        # Should produce equivalent results
        assert sync_image.mode == async_image.mode
        assert sync_image.size == async_image.size

    def test_multiple_loader_instances(self, sample_image_path):
        """Test multiple ImageLoader instances work independently"""
        loader1 = ImageLoader()
        loader2 = ImageLoader()
        
        image1 = loader1.load_image_sync(sample_image_path)
        image2 = loader2.load_image_sync(sample_image_path)
        
        assert image1.mode == image2.mode
        assert image1.size == image2.size

    @pytest.mark.asyncio
    async def test_concurrent_async_loads(self, image_loader, sample_image_path):
        """Test concurrent async image loading"""
        import asyncio
        
        # Load same image multiple times concurrently
        tasks = [
            image_loader.load_image(sample_image_path)
            for _ in range(3)
        ]
        
        images = await asyncio.gather(*tasks)
        
        # All should succeed and be equivalent
        for image in images:
            assert isinstance(image, Image.Image)
            assert image.mode == 'RGB'
            assert image.size == (50, 50)