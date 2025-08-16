import asyncio

from fastapi import FastAPI
from starlette.responses import JSONResponse

from service import evaluation, system
from service.constants import APP_NAME, APP_TITLE, PATH_PREFIX
from service.core.observability import (
    PrometheusMiddleware,
    get_metrics_middleware,
    metrics_endpoint,
)
from service.log import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title=APP_TITLE,
    docs_url=f"{PATH_PREFIX}/docs",
    redoc_url=f"{PATH_PREFIX}/redoc",
    openapi_url=f"{PATH_PREFIX}/openapi.json",
    description="Vision-Language similarity evaluation service using OpenCLIP models",
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware, app_name=APP_NAME)

# Add metrics endpoint
app.add_route(f"{PATH_PREFIX}/metrics", metrics_endpoint)

# Include routers
app.include_router(evaluation.router, prefix=PATH_PREFIX)
app.include_router(system.router, prefix=PATH_PREFIX)


@app.on_event("startup")
async def startup_event():
    """Initialize system metrics monitoring"""

    async def update_metrics_task():
        """Update system metrics every 10 seconds"""
        while True:
            try:
                metrics = get_metrics_middleware()
                metrics.update_system_metrics()
                await asyncio.sleep(10)
            except Exception:
                await asyncio.sleep(10)

    asyncio.create_task(update_metrics_task())


@app.get(f"{PATH_PREFIX}/health")
async def health():
    """Basic health check endpoint"""
    return JSONResponse(content={"status": "available", "service": APP_NAME})


@app.get("/")
async def root():
    """Root endpoint redirect"""
    return JSONResponse(
        content={"message": f"Welcome to {APP_TITLE}", "docs": f"{PATH_PREFIX}/docs", "health": f"{PATH_PREFIX}/health"}
    )
