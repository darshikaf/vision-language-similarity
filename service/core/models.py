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


@dataclass
class ModelConfig:
    """Configuration for CLIP model"""

    model_name: str
    pretrained: str

    @classmethod
    def get_fast_config(cls) -> "ModelConfig":
        """Get fast model configuration"""
        return cls(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")

    @classmethod
    def get_accurate_config(cls) -> "ModelConfig":
        """Get accurate model configuration"""
        return cls(model_name="ViT-L-14", pretrained="laion2b_s32b_b82k")
