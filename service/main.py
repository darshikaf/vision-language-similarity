from fastapi import FastAPI
from starlette.responses import JSONResponse

from service import evaluation
from service.constants import APP_NAME, APP_TITLE, PATH_PREFIX

app = FastAPI(
    title=APP_TITLE,
    docs_url=f"{PATH_PREFIX}/docs",
    redoc_url=f"{PATH_PREFIX}/redoc",
    openapi_url=f"{PATH_PREFIX}/openapi.json",
    description="Vision-Language similarity evaluation service using OpenCLIP models",
)

# Include routers
app.include_router(evaluation.router, prefix=PATH_PREFIX)


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
