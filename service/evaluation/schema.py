from pydantic import BaseModel, Field, field_validator


class EvaluationRequest(BaseModel):
    """Single evaluation request"""

    image_input: str = Field(..., description="Image URL or base64-encoded image")
    text_prompt: str = Field(..., description="Text description to compare with image")
    model_config_name: str | None = Field("fast", description="Model configuration: 'fast' or 'accurate'")

    @field_validator("image_input", "text_prompt")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate that required string fields are not empty"""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v


class BatchEvaluationRequest(BaseModel):
    """Batch evaluation request"""

    evaluations: list[EvaluationRequest] = Field(..., description="List of image-text pairs to evaluate")
    batch_size: int | None = Field(32, description="Batch size for processing", ge=1, le=128)
    show_progress: bool | None = Field(True, description="Show progress during batch processing")


class EvaluationResponse(BaseModel):
    """Single evaluation response"""

    image_input: str = Field(..., description="Original image input")
    text_prompt: str = Field(..., description="Original text prompt")
    clip_score: float = Field(..., description="CLIP similarity score (0-100)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    error: str | None = Field(None, description="Error message if evaluation failed")
    model_used: str = Field(..., description="Model configuration used")


class BatchEvaluationResponse(BaseModel):
    """Batch evaluation response"""

    results: list[EvaluationResponse] = Field(..., description="Individual evaluation results")
    total_processed: int = Field(..., description="Total number of evaluations processed")
    total_successful: int = Field(..., description="Number of successful evaluations")
    total_failed: int = Field(..., description="Number of failed evaluations")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether models are loaded and ready")
    available_configs: list[str] = Field(..., description="Available model configurations")
