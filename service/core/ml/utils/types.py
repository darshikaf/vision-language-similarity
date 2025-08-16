from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Simple evaluation result with CLIP score only"""

    image_path: str
    text_prompt: str
    clip_score: float
    processing_time_ms: float
    error: str | None = None
    error_type: str | None = None
