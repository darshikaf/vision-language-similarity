import asyncio
import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
import httpx
from PIL import Image

from service.core.exceptions import ImageProcessingError, NetworkError


class ImageLoader:
    """
    Handles loading images from various sources with error handling

    TODO:
    # Timeout and connection limit configuration
    # HTTP/2 support for better performance
    """

    def __init__(self, http_client: httpx.AsyncClient = None):
        """Initialize with optional HTTP client for connection reuse"""
        self._http_client = http_client
        self._own_client = http_client is None

    async def __aenter__(self):
        """Async context manager entry for process pooling"""
        if self._own_client:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                follow_redirects=True,
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._own_client and self._http_client:
            await self._http_client.aclose()

    async def load_image(self, image_input: str | Image.Image | Path) -> Image.Image:
        """Load image from URL, file path, base64, or PIL Image"""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        image_str = str(image_input)

        # Check if base64 data URI
        if image_str.startswith("data:image/"):
            return await self._load_from_base64(image_str)
        # Check if URL
        elif self._is_url(image_str):
            return await self._load_from_url(image_str)
        else:
            return await self._load_from_file(image_str)

    @staticmethod
    def _is_url(path_str: str) -> bool:
        """Check if string is a URL"""
        parsed = urlparse(path_str)
        return parsed.scheme in ("http", "https")

    async def _load_from_url(self, url: str) -> Image.Image:
        """Load image from URL with retry logic"""
        if not self._http_client:
            raise ImageProcessingError("HTTP client not initialized. Use async context manager.")

        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = await self._http_client.get(url)
                response.raise_for_status()

                # Run PIL operations in thread pool to avoid blocking
                image_bytes = response.content
                return await asyncio.to_thread(self._process_image_bytes, image_bytes)

            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx), but retry server errors (5xx)
                if e.response.status_code < 500:
                    raise NetworkError(f"Failed to fetch image from URL {url}: HTTP {e.response.status_code}") from e

            except httpx.RequestError:
                # Network errors - will retry
                pass

            except Exception as e:
                # PIL or other processing errors - don't retry
                raise ImageProcessingError(f"Failed to process image from URL {url}: {e}") from e

            # Retry with exponential backoff
            if attempt < max_retries - 1:
                delay = 1.0 * (2**attempt)
                await asyncio.sleep(delay)

        # All retries exhausted
        raise NetworkError(f"Failed to fetch image from URL {url} after {max_retries} attempts")

    async def _load_from_file(self, file_path: str) -> Image.Image:
        """Load image from local file with async I/O"""
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")

            # Use aiofiles for non-blocking file read
            async with aiofiles.open(file_path, "rb") as f:
                image_bytes = await f.read()

            # Process image in thread pool
            image = await asyncio.to_thread(self._process_image_bytes, image_bytes)
            return image

        except Exception as e:
            raise ImageProcessingError(f"Failed to load image from file {file_path}: {e}") from e

    async def _load_from_base64(self, data_uri: str) -> Image.Image:
        """Load image from base64 data URI"""
        try:
            # Parse data URI: data:image/png;base64,<data>
            header, encoded = data_uri.split(",", 1)
            image_data = await asyncio.to_thread(self._decode_base64, encoded)
            image = await asyncio.to_thread(self._process_image_bytes, image_data)
            return image
        except Exception as e:
            raise ImageProcessingError(f"Failed to process base64 image: {e}") from e

    @staticmethod
    def _decode_base64(encoded_data: str) -> bytes:
        """Decode base64 data - runs in thread pool"""
        return base64.b64decode(encoded_data)

    @staticmethod
    def _process_image_bytes(image_bytes: bytes) -> Image.Image:
        """Process image bytes into PIL Image - runs in thread pool"""
        image = Image.open(BytesIO(image_bytes))
        return image.convert("RGB")
