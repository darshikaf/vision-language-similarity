from collections.abc import Callable
import time
from typing import Any

from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import psutil
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
import torch

from service.constants import APP_NAME


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Prometheus middleware for FastAPI with ML-specific metrics
    """

    _instance: "PrometheusMiddleware | None" = None
    _metrics_initialized = False

    def __init__(self, app: Any, app_name: str = APP_NAME) -> None:
        super().__init__(app)
        self.app_name = app_name

        # Use singleton pattern to avoid duplicate metrics registration
        if not PrometheusMiddleware._metrics_initialized:
            self._initialize_metrics()
            PrometheusMiddleware._metrics_initialized = True
            PrometheusMiddleware._instance = self

    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics (called once)"""
        self.REQUEST_COUNT = Counter(
            "vision_similarity_requests_total",
            "Total requests processed",
            ["method", "path", "app_name"],
        )

        self.RESPONSE_COUNT = Counter(
            "vision_similarity_responses_total",
            "Total responses sent",
            ["method", "path", "status_code", "app_name"],
        )

        self.REQUEST_DURATION = Histogram(
            "vision_similarity_requests_duration_seconds",
            "Request processing time",
            ["method", "path", "app_name"],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0],
        )

        self.REQUESTS_IN_PROGRESS = Gauge(
            "vision_similarity_requests_in_progress",
            "Active requests being processed",
            ["method", "path", "app_name"],
        )

        # Exception tracking
        self.EXCEPTION_COUNT = Counter(
            "vision_similarity_exceptions_total",
            "Total exceptions raised during request processing",
            ["exception_type", "method", "path", "app_name"],
        )

        # ML-specific metrics
        self.MODEL_INFERENCE_DURATION = Histogram(
            "vision_similarity_model_inference_seconds",
            "Model inference processing time",
            ["model_config", "model_name", "device_type", "app_name"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        # Model loading time
        self.MODEL_LOAD_DURATION = Histogram(
            "vision_similarity_model_load_seconds",
            "Model loading and initialization time",
            ["model_config", "model_name", "app_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
        )

        # Model-specific error tracking
        self.MODEL_ERRORS = Counter(
            "vision_similarity_model_errors_total",
            "Model-specific error counts by type",
            ["model_config", "model_name", "error_type", "app_name"],
        )

        # Batch processing efficiency
        self.BATCH_EFFICIENCY = Histogram(
            "vision_similarity_batch_efficiency_ratio",
            "Batch processing efficiency (batch_time / (single_time * batch_size))",
            ["model_config", "model_name", "app_name"],
            buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0],
        )

        self.CLIP_SCORE_DISTRIBUTION = Histogram(
            "vision_similarity_clip_scores",
            "Distribution of CLIP similarity scores",
            ["model_config", "model_name", "app_name"],
            buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        )

        self.IMAGE_PROCESSING_DURATION = Histogram(
            "vision_similarity_image_processing_seconds",
            "Image loading and preprocessing time",
            ["source_type", "app_name"],  # url, file, base64
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        )

        self.BATCH_SIZE_DISTRIBUTION = Histogram(
            "vision_similarity_batch_sizes",
            "Distribution of batch sizes processed",
            ["app_name"],
            buckets=[1, 2, 4, 8, 16, 32, 64, 128],
        )

        self.EVALUATION_ERRORS = Counter(
            "vision_similarity_evaluation_errors_total",
            "Total evaluation errors by type",
            ["error_type", "model_config", "app_name"],
        )

        # Simple system metrics
        self.SYSTEM_CPU_USAGE = Gauge(
            "vision_similarity_cpu_usage_percent",
            "CPU usage percentage",
            ["app_name"],
        )

        self.SYSTEM_MEMORY_USAGE = Gauge(
            "vision_similarity_memory_usage_bytes",
            "Memory usage in bytes",
            ["memory_type", "app_name"],  # rss, vms
        )

        self.GPU_MEMORY_USAGE = Gauge(
            "vision_similarity_gpu_memory_bytes",
            "GPU memory usage in bytes",
            ["gpu_id", "memory_type", "app_name"],  # allocated, reserved
        )

        self.APP_INFO = Gauge(
            "vision_similarity_app_info",
            "Application information",
            ["app_name", "version"],
        )
        # Set app info (version to be added later)
        self.APP_INFO.labels(app_name=self.app_name, version="0.1.0").set(1)

        # Error pattern tracking
        self.ERROR_PATTERNS = Counter(
            "vision_similarity_error_patterns_total",
            "Error patterns by type, context and model",
            ["error_type", "error_context", "model_config", "app_name"],
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process HTTP request with metrics collection"""
        method = request.method
        path = self._resolve_path(request)

        # Start tracking request
        start_time = time.time()
        self.REQUEST_COUNT.labels(method=method, path=path, app_name=self.app_name).inc()
        self.REQUESTS_IN_PROGRESS.labels(method=method, path=path, app_name=self.app_name).inc()

        response = None
        try:
            response = await call_next(request)
            status_code = response.status_code

        except Exception as e:
            exception_type = type(e).__name__
            self.EXCEPTION_COUNT.labels(
                exception_type=exception_type,
                method=method,
                path=path,
                app_name=self.app_name,
            ).inc()
            raise

        finally:
            # Always decrement in-progress counter
            self.REQUESTS_IN_PROGRESS.labels(method=method, path=path, app_name=self.app_name).dec()

            # Record timing and response metrics
            if response is not None:
                duration = time.time() - start_time
                self.REQUEST_DURATION.labels(method=method, path=path, app_name=self.app_name).observe(duration)
                self.RESPONSE_COUNT.labels(
                    method=method,
                    path=path,
                    status_code=status_code,
                    app_name=self.app_name,
                ).inc()

        return response

    def _resolve_path(self, request: Request) -> str:
        """Resolve FastAPI route path, handling path parameters"""
        if hasattr(request, "scope") and "route" in request.scope:
            route = request.scope["route"]
            if hasattr(route, "path"):
                return route.path

        # Fallback to raw path
        return request.url.path

    def record_inference_time(
        self, duration: float, model_config: str, device_type: str, model_name: str = "unknown"
    ) -> None:
        """Record model inference timing"""
        self.MODEL_INFERENCE_DURATION.labels(
            model_config=model_config,
            model_name=model_name,
            device_type=device_type,
            app_name=self.app_name,
        ).observe(duration)

    def record_clip_score(self, score: float, model_config: str, model_name: str = "unknown") -> None:
        """Record CLIP similarity score"""
        self.CLIP_SCORE_DISTRIBUTION.labels(
            model_config=model_config, model_name=model_name, app_name=self.app_name
        ).observe(score)

    def record_image_processing_time(self, duration: float, source_type: str) -> None:
        """Record image processing timing"""
        self.IMAGE_PROCESSING_DURATION.labels(source_type=source_type, app_name=self.app_name).observe(duration)

    def record_batch_size(self, batch_size: int) -> None:
        """Record batch processing size"""
        self.BATCH_SIZE_DISTRIBUTION.labels(app_name=self.app_name).observe(batch_size)

    def record_evaluation_error(self, error_type: str, model_config: str) -> None:
        """Record evaluation error"""
        self.EVALUATION_ERRORS.labels(
            error_type=error_type,
            model_config=model_config,
            app_name=self.app_name,
        ).inc()

    def record_model_error(self, error_type: str, model_config: str, model_name: str) -> None:
        """Record model-specific error"""
        self.MODEL_ERRORS.labels(
            model_config=model_config,
            model_name=model_name,
            error_type=error_type,
            app_name=self.app_name,
        ).inc()

    def record_model_load_time(self, duration: float, model_config: str, model_name: str) -> None:
        """Record model loading time"""
        self.MODEL_LOAD_DURATION.labels(
            model_config=model_config,
            model_name=model_name,
            app_name=self.app_name,
        ).observe(duration)

    def record_batch_efficiency(self, efficiency_ratio: float, model_config: str, model_name: str) -> None:
        """Record batch processing efficiency"""
        self.BATCH_EFFICIENCY.labels(
            model_config=model_config,
            model_name=model_name,
            app_name=self.app_name,
        ).observe(efficiency_ratio)

    def record_error_pattern(self, error_type: str, error_context: str, model_config: str) -> None:
        """Record simple error pattern for monitoring"""
        self.ERROR_PATTERNS.labels(
            error_type=error_type,
            error_context=error_context,
            model_config=model_config,
            app_name=self.app_name,
        ).inc()

    def update_system_metrics(self) -> None:
        """Update system resource metrics - called periodically"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.SYSTEM_CPU_USAGE.labels(app_name=self.app_name).set(cpu_percent)

            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            self.SYSTEM_MEMORY_USAGE.labels(memory_type="rss", app_name=self.app_name).set(memory_info.rss)
            self.SYSTEM_MEMORY_USAGE.labels(memory_type="vms", app_name=self.app_name).set(memory_info.vms)

            # GPU memory if available
            if torch.cuda.is_available():
                for gpu_id in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(gpu_id)
                    reserved = torch.cuda.memory_reserved(gpu_id)

                    self.GPU_MEMORY_USAGE.labels(
                        gpu_id=str(gpu_id), memory_type="allocated", app_name=self.app_name
                    ).set(allocated)
                    self.GPU_MEMORY_USAGE.labels(
                        gpu_id=str(gpu_id), memory_type="reserved", app_name=self.app_name
                    ).set(reserved)

        except Exception:
            # Silently fail to avoid disrupting service
            pass


_metrics_middleware: PrometheusMiddleware | None = None


def get_metrics_middleware() -> PrometheusMiddleware:
    """Get the global metrics middleware instance"""
    if PrometheusMiddleware._instance is None:
        raise RuntimeError("Metrics middleware not initialized. Add middleware to FastAPI app first.")
    return PrometheusMiddleware._instance


def metrics_endpoint(request: Request) -> StarletteResponse:
    """Prometheus metrics endpoint"""
    return StarletteResponse(generate_latest(), media_type="text/plain")
