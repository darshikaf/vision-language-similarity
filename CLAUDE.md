# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Initial setup (ARM64/M1 only)
chmod +x dev-setup.sh
./dev-setup.sh
source .venv/bin/activate
```

### Running the Service
```bash
# FastAPI service (local development)
uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload

# Ray Serve deployment (advanced scaling)
serve run service.ray_main:deployment

# Docker service with observability stack
make run-local-otel
```

### Testing
```bash
# Local unit tests
python -m pytest tests/

# Docker-based testing
make run-unit-tests           # Unit tests in Docker
make run-integration-tests    # Integration tests in Docker
make run-unit-tests-local     # Local unit tests with coverage
```

### Code Quality
```bash
# Format code (recommended before commits)
make run-style-inplace-local

# Check code style
make run-style
```

### Load Testing
```bash
# Interactive load test with web UI at http://localhost:8089
make load-test
```

### CLI Tool
```bash
# Batch evaluation of images with performance comparison
python cli/evaluation_cli.py cli/challenge_set.csv --compare
```

## Architecture

### Service Architecture
This is a vision-language similarity evaluation service with two deployment modes:

1. **Standard FastAPI Service** (`service/main.py`): Simple REST API for development and basic production
2. **Ray Serve Deployment** (`service/ray_main.py`): Advanced ML serving with auto-scaling, fault tolerance, and load balancing

### Core Components

- **ML Engine** (`service/core/ml/engines/`): 
  - `model_manager.py`: Singleton model cache and lifecycle management
  - `base_evaluator.py`: Abstract evaluator interface
  - `openclip_evaluator.py`: OpenCLIP implementation with batching support

- **ML Models** (`service/core/ml/models/`):
  - `factory.py`: Model factory for creating similarity models
  - `openclip_model.py`: OpenCLIP model wrapper with device management
  - Dynamic model configuration system with JSON/environment variable overrides

- **ML Utils** (`service/core/ml/utils/`):
  - `config.py`: Dynamic model registry with built-in defaults ("fast", "accurate")
  - `device_manager.py`: Automatic device detection (CUDA/MPS/CPU)
  - `image_processor.py`: Image preprocessing and tensor conversion
  - `metrics_recorder.py`: Performance and resource metrics collection

- **API Layers**:
  - `service/evaluation/`: CLIP score evaluation endpoints (single/batch)
  - `service/system/`: Health checks and system status
  - Prometheus metrics at `/evaluator/metrics`

### Key Design Patterns

- **Singleton Model Manager**: Caches models across requests to avoid repeated loading
- **Dynamic Configuration**: Models configurable via JSON files or environment variables
- **Batch Processing**: Optimized batch evaluation with configurable batch sizes
- **Device Abstraction**: Automatic device selection with fallback (CUDA → MPS → CPU)
- **Observability First**: Built-in metrics, logging, and OpenTelemetry integration

### Testing Structure

- **Unit Tests** (`tests/unit/`): Component-level testing with mocked dependencies
- **Integration Tests** (`tests/integration/`): End-to-end API testing with real models
- **Fixtures** (`tests/fixtures/`): Reusable test data and mock objects
- **Test Data** (`tests/data/samples/`): Sample images and challenge sets for testing

### Performance Characteristics

- **Single Evaluation**: ~4.2 images/second on Apple M1
- **Batch Processing**: ~21.1 images/second on Apple M1  
- **Memory Footprint**: ~980-1015MB per evaluator instance
- **Memory Persistence**: ~577MB remains after model destruction (cached weights)

### Environment Requirements

- Python 3.11+
- ARM64/M1 architecture for local development
- Docker and Docker Compose for containerized deployment
- Ray Serve for advanced deployment scenarios

### Model Configuration

Built-in model profiles:
- `fast`: ViT-B-32/laion2b_s34b_b79k (faster inference)
- `accurate`: ViT-L-14/laion2b_s32b_b82k (higher accuracy)

Models can be extended via `/app/config/models.json` or `MODEL_CONFIG_*` environment variables.

### Observability Stack

When using `make run-local-otel`:
- **Grafana Dashboard**: http://localhost:3000/d/vision-language-similarity/
- **Prometheus Metrics**: http://localhost:9090
- **Service Metrics**: http://localhost:8000/evaluator/metrics
- **API Docs**: http://localhost:8000/evaluator/docs

### CLIP Score Calculation

Standard CLIP score formula:
```python
CLIP Score = max(100 * cosine_similarity(image_embedding, text_embedding), 0)
```

Supports both single image evaluation and batch processing with progress tracking.
