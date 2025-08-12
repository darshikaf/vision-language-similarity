"""
Minimal OpenCLIP evaluator with essential features for vision-language similarity
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .device_manager import DeviceManager
from .image_loader import ImageLoader
from .models import EvaluationResult

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MinimalOpenCLIPEvaluator:
    """
    Minimal OpenCLIP evaluator with essential features:
    - Single model configuration with caching
    - CLIP standard scoring only
    - Robust error handling
    - Efficient batch processing
    """
    
    def __init__(
        self,
        model_name: str = 'ViT-B-32',
        pretrained: str = 'laion2b_s34b_b79k',
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize minimal evaluator
        
        Args:
            model_name: OpenCLIP model architecture
            pretrained: Pretrained weights
            device: Device for computation (auto-detected if None)
            cache_dir: Directory for caching models
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "openclip")
        
        # Setup device and precision
        self.device = DeviceManager.get_optimal_device(device)
        self.precision = DeviceManager.get_optimal_precision(self.device)
        
        # Initialize components
        self.model, self.preprocess = self._load_model()
        self.image_loader = ImageLoader()
        
        logger.info(f"Initialized {model_name}/{pretrained} on {self.device}")
    
    def _load_model(self):
        """Load OpenCLIP model with error handling"""
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
                precision=self.precision,
                cache_dir=self.cache_dir
            )
            model.eval()
            
            # Apply mixed precision optimization
            if self.precision == 'fp16' and self.device.type == 'cuda':
                model = model.half()
            
            return model, preprocess
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}/{self.pretrained}: {e}")
            raise
    
    def _calculate_clip_score(self, raw_cosine: float) -> float:
        """Convert to CLIP standard score: max(100 * cosine, 0)"""
        return max(100 * raw_cosine, 0)
    
    def _create_failed_result(
        self, 
        image_input: Union[str, Image.Image, Path],
        text_prompt: str,
        error_message: str
    ) -> EvaluationResult:
        """Create result for failed evaluation"""
        return EvaluationResult(
            image_path=str(image_input),
            text_prompt=text_prompt,
            clip_score=0.0,
            processing_time_ms=0.0,
            error=error_message
        )
    
    @torch.no_grad()
    def evaluate_single(
        self,
        image_input: Union[str, Image.Image, Path],
        text_prompt: str
    ) -> EvaluationResult:
        """
        Evaluate single image-text pair
        
        Args:
            image_input: Image source (URL, file path, or PIL Image)
            text_prompt: Text description to compare
        
        Returns:
            EvaluationResult with CLIP score
        """
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = self.image_loader.load_image(image_input)
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text
            text_tokens = open_clip.tokenize([text_prompt]).to(self.device)
            
            # Extract and normalize features
            image_features = F.normalize(
                self.model.encode_image(image_tensor), p=2, dim=-1
            )
            text_features = F.normalize(
                self.model.encode_text(text_tokens), p=2, dim=-1
            )
            
            # Calculate similarity and convert to CLIP score
            raw_cosine = torch.cosine_similarity(
                image_features, text_features, dim=-1
            ).item()
            
            clip_score = self._calculate_clip_score(raw_cosine)
            
            return EvaluationResult(
                image_path=str(image_input),
                text_prompt=text_prompt,
                clip_score=clip_score,
                processing_time_ms=(time.time() - start_time) * 1000,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Evaluation failed for {image_input}: {e}")
            return self._create_failed_result(image_input, text_prompt, str(e))
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        image_inputs: List[Union[str, Image.Image, Path]],
        text_prompts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple image-text pairs using efficient batch processing
        
        Args:
            image_inputs: List of image sources
            text_prompts: List of text descriptions
            batch_size: Batch size for processing
            show_progress: Show progress bar
        
        Returns:
            List of EvaluationResult objects
        """
        if len(image_inputs) != len(text_prompts):
            raise ValueError(
                f"Mismatch: {len(image_inputs)} images vs {len(text_prompts)} prompts"
            )
        
        results = []
        total_batches = (len(image_inputs) + batch_size - 1) // batch_size
        
        iterator = range(0, len(image_inputs), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=total_batches, desc="Processing batches")
        
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(image_inputs))
            batch_images_raw = image_inputs[batch_start:batch_end]
            batch_prompts = text_prompts[batch_start:batch_end]
            
            start_time = time.time()
            
            # Load and preprocess all images in batch
            batch_images = []
            valid_indices = []
            failed_indices = []
            
            for idx, img_input in enumerate(batch_images_raw):
                try:
                    image = self.image_loader.load_image(img_input)
                    batch_images.append(image)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.error(f"Failed to load image {img_input}: {e}")
                    failed_indices.append(idx)
            
            # Process valid images in batch
            if batch_images:
                try:
                    # Stack preprocessed images into a single tensor
                    image_tensors = torch.stack([
                        self.preprocess(img) for img in batch_images
                    ]).to(self.device)
                    
                    # Tokenize all texts at once
                    valid_prompts = [batch_prompts[i] for i in valid_indices]
                    text_tokens = open_clip.tokenize(valid_prompts).to(self.device)
                    
                    # Extract features in batch
                    image_features = F.normalize(
                        self.model.encode_image(image_tensors), p=2, dim=-1
                    )
                    text_features = F.normalize(
                        self.model.encode_text(text_tokens), p=2, dim=-1
                    )
                    
                    # Calculate similarities for entire batch
                    cosine_similarities = torch.sum(image_features * text_features, dim=-1)
                    
                    batch_time = (time.time() - start_time) * 1000
                    avg_time_per_item = batch_time / len(batch_images)
                    
                    # Create results for valid images
                    for idx, (valid_idx, cosine_sim) in enumerate(zip(valid_indices, cosine_similarities)):
                        raw_cosine = cosine_sim.item()
                        clip_score = self._calculate_clip_score(raw_cosine)
                        
                        results.append(EvaluationResult(
                            image_path=str(batch_images_raw[valid_idx]),
                            text_prompt=batch_prompts[valid_idx],
                            clip_score=clip_score,
                            processing_time_ms=avg_time_per_item,
                            error=None
                        ))
                
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Fall back to individual processing for this batch
                    for valid_idx in valid_indices:
                        result = self.evaluate_single(
                            batch_images_raw[valid_idx], 
                            batch_prompts[valid_idx]
                        )
                        results.append(result)
            
            # Add failed results
            for failed_idx in failed_indices:
                results.append(self._create_failed_result(
                    batch_images_raw[failed_idx], 
                    batch_prompts[failed_idx],
                    "Image loading failed"
                ))
        
        return results
    
    @classmethod
    def create_fast_evaluator(cls, device: Optional[str] = None) -> 'MinimalOpenCLIPEvaluator':
        """Create evaluator optimized for speed"""
        return cls(
            model_name='ViT-B-32',
            pretrained='laion2b_s34b_b79k',
            device=device
        )
    
    @classmethod
    def create_accurate_evaluator(cls, device: Optional[str] = None) -> 'MinimalOpenCLIPEvaluator':
        """Create evaluator optimized for accuracy"""
        return cls(
            model_name='ViT-L-14',
            pretrained='laion2b_s32b_b82k',
            device=device
        )