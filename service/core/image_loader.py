"""
Image loading utilities with support for URLs, files, and PIL Images
"""

from io import BytesIO
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import requests
from PIL import Image


class ImageLoader:
    """Handles loading images from various sources with error handling"""
    
    @staticmethod
    def load_image(image_input: Union[str, Image.Image, Path]) -> Image.Image:
        """Load image from URL, file path, or PIL Image"""
        if isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        
        image_str = str(image_input)
        
        # Check if URL
        if ImageLoader._is_url(image_str):
            return ImageLoader._load_from_url(image_str)
        else:
            return ImageLoader._load_from_file(image_str)
    
    @staticmethod
    def _is_url(path_str: str) -> bool:
        """Check if string is a URL"""
        parsed = urlparse(path_str)
        return parsed.scheme in ('http', 'https')
    
    @staticmethod
    def _load_from_url(url: str, timeout: int = 30) -> Image.Image:
        """Load image from URL with error handling"""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image.convert('RGB')
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch image from URL {url}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to process image from URL {url}: {e}")
    
    @staticmethod
    def _load_from_file(file_path: str) -> Image.Image:
        """Load image from local file"""
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")
            
            image = Image.open(file_path)
            return image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image from file {file_path}: {e}")