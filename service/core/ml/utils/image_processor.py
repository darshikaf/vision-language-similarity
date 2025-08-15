import asyncio
from pathlib import Path
from PIL import Image

from service.core.ml.utils.image_loader import ImageLoader


class ImageProcessor:
    """
    Handles image loading and processing from various input sources.
    
    Responsibilities:
    - Load images from URLs, file paths, or PIL Image objects
    - Determine source type for metrics and logging
    - Handle image format conversion to RGB
    - Manage concurrent image loading operations
    """

    def __init__(self, max_concurrent_loads: int = 10):
        """
        Initialize image processor.
        
        Args:
            max_concurrent_loads: Maximum concurrent image loading operations
        """
        self.max_concurrent_loads = max_concurrent_loads

    async def load_image(self, image_input: str | Image.Image | Path, image_loader: ImageLoader | None = None) -> Image.Image:
        """
        Load image from various input sources.
        
        Args:
            image_input: Image source (URL, file path, or PIL Image)
            image_loader: Optional async image loader instance
            
        Returns:
            PIL Image in RGB format
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif image_loader:
            return await image_loader.load_image(image_input)
        else:
            async with ImageLoader() as loader:
                return await loader.load_image(image_input)

    def determine_source_type(self, image_input: str | Image.Image | Path) -> str:
        """
        Determine source type based on input for metrics categorization.
        
        Args:
            image_input: Image source to categorize
            
        Returns:
            Source type string: "url", "base64", "file", or "pil_image"
        """
        if isinstance(image_input, str):
            if image_input.startswith(("http://", "https://")):
                return "url"
            elif image_input.startswith("data:image/"):
                return "base64"
            else:
                return "file"
        else:
            return "pil_image"

    async def load_images_batch(
        self, 
        image_inputs: list[str | Image.Image | Path]
    ) -> list[Image.Image | Exception]:
        """
        Load multiple images concurrently with error handling.
        
        Args:
            image_inputs: List of image sources to load
            
        Returns:
            List of loaded PIL Images or Exception objects for failed loads
        """
        async with ImageLoader() as image_loader:
            image_tasks = [self.load_image(img_input, image_loader) for img_input in image_inputs]
            return await asyncio.gather(*image_tasks, return_exceptions=True)
