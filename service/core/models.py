"""
Core data models for the vision-language similarity evaluator
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationResult:
    """Simple evaluation result with CLIP score only"""
    image_path: str
    text_prompt: str
    clip_score: float
    processing_time_ms: float
    error: Optional[str] = None