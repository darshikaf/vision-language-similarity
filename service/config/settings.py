from pydantic import Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    """Pure application model configuration"""

    cache_dir: str = Field(default="./model-cache", env="MODEL_CACHE_DIR")
    max_workers: int = Field(default=4, env="MODEL_MAX_WORKERS")
    default_timeout: float = Field(default=30.0, env="MODEL_TIMEOUT")
    enable_mixed_precision: bool = Field(default=True, env="ENABLE_MIXED_PRECISION")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")


class APIConfig(BaseSettings):
    """API configuration"""

    max_batch_size: int = Field(default=100, env="MAX_BATCH_SIZE")
    max_image_size_mb: int = Field(default=10, env="MAX_IMAGE_SIZE_MB")
    rate_limit_per_minute: int = Field(default=1000, env="RATE_LIMIT_PER_MINUTE")


class ObservabilityConfig(BaseSettings):
    """Observability configuration"""

    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")
    metrics_port: int = Field(default=8080, env="METRICS_PORT")


class AppSettings(BaseSettings):
    """Clean application settings"""

    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Sub-configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = AppSettings()
