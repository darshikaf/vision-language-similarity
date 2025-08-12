import time
from functools import wraps
from typing import Any, Callable

from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from ..constants import APP_NAME


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Prometheus middleware for FastAPI with ML-specific metrics"""
    
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
            ["model_config", "device_type", "app_name"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        self.CLIP_SCORE_DISTRIBUTION = Histogram(
            "vision_similarity_clip_scores",
            "Distribution of CLIP similarity scores",
            ["model_config", "app_name"],
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

        self.APP_INFO = Gauge(
            "vision_similarity_app_info",
            "Application information",
            ["app_name", "version"],
        )
        # Set app info (version to be added later)
        self.APP_INFO.labels(app_name=self.app_name, version="0.1.0").set(1)

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

    def record_inference_time(self, duration: float, model_config: str, device_type: str) -> None:
        """Record model inference timing"""
        self.MODEL_INFERENCE_DURATION.labels(
            model_config=model_config,
            device_type=device_type,
            app_name=self.app_name,
        ).observe(duration)

    def record_clip_score(self, score: float, model_config: str) -> None:
        """Record CLIP similarity score"""
        self.CLIP_SCORE_DISTRIBUTION.labels(model_config=model_config, app_name=self.app_name).observe(score)

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


_metrics_middleware: PrometheusMiddleware | None = None


def get_metrics_middleware() -> PrometheusMiddleware:
    """Get the global metrics middleware instance"""
    if PrometheusMiddleware._instance is None:
        raise RuntimeError("Metrics middleware not initialized. Add middleware to FastAPI app first.")
    return PrometheusMiddleware._instance


def metrics_endpoint(request: Request) -> StarletteResponse:
    """Prometheus metrics endpoint"""
    return StarletteResponse(generate_latest(), media_type="text/plain")


def measure_db_query_time(func: Callable) -> Callable:
    """Decorator to measure database query time (future use)"""

    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            # TODO: Add database query metrics when needed
            # metrics.record_db_query_time(duration, func.__name__)

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            # TODO: Add database query metrics when needed
            # metrics.record_db_query_time(duration, func.__name__)

    # Return appropriate wrapper based on function type
    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper