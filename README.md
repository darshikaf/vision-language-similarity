# Vision-Language Similarity Service

A high-performance vision-language similarity evaluation service built with OpenCLIP models. This service provides CLIP score calculations for comparing images with text descriptions, supporting both single and batch processing with comprehensive observability, profiling, and advanced ML serving capabilities.

## Features

- **CLIP Score Evaluation**: Calculate semantic similarity between images and text using state-of-the-art OpenCLIP models
- **Batch Processing**: Efficient batch evaluation with configurable batch sizes and progress tracking
- **Multiple Model Configurations**: Support for different model profiles (fast vs accurate) with dynamic configuration
- **Dual Deployment Options**: Standard FastAPI service or advanced Ray Serve deployment with auto-scaling
- **Comprehensive Observability**: Prometheus metrics, Grafana dashboards, Jaeger tracing, and structured logging
- **Error Recovery**: Graceful handling of failed images in batch processing without stopping entire operations
- **Performance Profiling**: Built-in profiling tools for optimization and performance analysis
- **Load Testing**: Integrated load testing with Locust for performance validation

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Make (for build automation)

### Development Setup

1. **Clone and setup the environment:**
   ```bash
   git clone <repository-url>
   cd vision-language-similarity
   make dev-setup
   ```

2. **Run the service locally:**
   ```bash
   # Standard FastAPI service
   make run-local
   
   # Or run directly with Python
   python service/run_service.py
   ```

3. **Access the service:**
   - API Documentation: http://localhost:8000/evaluator/docs
   - Health Check: http://localhost:8000/evaluator/health
   - Metrics: http://localhost:8000/evaluator/metrics

### Ray Serve Deployment (Advanced ML Serving)

For production ML workloads requiring auto-scaling and advanced deployment features:

```bash
# Build Ray Serve images
make build-ray-base
make build-ray

# Run Ray Serve deployment
make run-local-ray
```

- Ray Dashboard: http://localhost:8265
- API Documentation: http://localhost:8000/evaluator/docs

## API Usage

### Single Image Evaluation

```bash
curl -X POST "http://localhost:8000/evaluator/v1/evaluation/single" \
     -H "Content-Type: application/json" \
     -d '{
       "image_input": "https://example.com/image.jpg",
       "text_prompt": "A beautiful sunset over mountains",
       "model_config_name": "fast"
     }'
```

**Response:**
```json
{
  "image_input": "https://example.com/image.jpg",
  "text_prompt": "A beautiful sunset over mountains",
  "clip_score": 82.5,
  "processing_time_ms": 45.2,
  "model_used": "fast",
  "error": null
}
```

### Batch Evaluation

```bash
curl -X POST "http://localhost:8000/evaluator/v1/evaluation/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "evaluations": [
         {
           "image_input": "https://example.com/image1.jpg",
           "text_prompt": "A cat sitting on a table",
           "model_config_name": "fast"
         },
         {
           "image_input": "https://example.com/image2.jpg", 
           "text_prompt": "A dog running in a park",
           "model_config_name": "accurate"
         }
       ],
       "batch_size": 32,
       "show_progress": true
     }'
```

### Health Check

```bash
curl "http://localhost:8000/evaluator/health"
```

**Response:**
```json
{
  "status": "available",
  "service": "vision_language_similarity_evaluator"
}
```

## Development

### Running Tests

```bash
# Run all tests with coverage locally
python -m pytest tests/

# Run unit tests in Docker
make run-unit-test-suite

# Run integration tests in Docker
make run-integration-test-suite

# Run specific test files
python -m pytest tests/test_evaluator.py -v
```

### Code Quality

```bash
# Format code with ruff
make run-style-inplace-local
```

### Observability Stack

Start the full observability stack for monitoring and debugging:

```bash
# Start observability services
make run-local-otel

# Access dashboards
# Grafana: http://localhost:3000 (admin/grafana)  
# Prometheus: http://localhost:9090
# Jaeger: http://localhost:16686

# Clean up
make clean-otel
```

### Load Testing

```bash
# Interactive load test with web UI
make load-test
# Web UI: http://localhost:8089

# Light load test for development
make load-test-light

# CI/CD performance validation
make load-test-ci
```

### Performance Profiling

```bash
# Run profiling analysis
cd evaluator_profiler
python run_profile.py

# Compare performance between runs
python profile_compare.py
```

## Docker Deployment

### Standard FastAPI Service

```bash
# Build images
make build-base
make build-app

# Run service
make run-local-otel
```

### Ray Serve Deployment

```bash
# Build Ray images
make build-ray-base  
make build-ray

# Deploy Ray Serve
make run-local-ray
```

## Architecture

The service is built with a modular architecture supporting two deployment modes:

### Standard FastAPI Service
- **Fast Development**: Quick iteration and debugging
- **Simple Deployment**: Standard container deployment
- **Resource Efficient**: Lower resource overhead

### Ray Serve Deployment  
- **Auto-scaling**: Automatic replica scaling based on traffic
- **Load Balancing**: Intelligent request distribution
- **A/B Testing**: Support for model variants and experiments
- **Production Features**: Blue-green deployments, health monitoring

### Core Components

- **Service Layer** (`service/`): FastAPI application with REST API endpoints
- **Core Logic** (`service/core/`): CLIP evaluation, device management, image loading
- **Model Management** (`service/model_management/`): Dynamic model configuration and loading
- **Observability** (`service/observability/`): Prometheus metrics and monitoring
- **Configuration** (`config/`): Model configurations and service settings

## API Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/evaluator/docs` | GET | Interactive API documentation |
| `/evaluator/health` | GET | Service health check |
| `/evaluator/v1/evaluation/single` | POST | Single image-text evaluation |
| `/evaluator/v1/evaluation/batch` | POST | Batch image-text evaluation |
| `/evaluator/metrics` | GET | Prometheus metrics |
| `/evaluator/admin/models` | GET | List available models |
| `/evaluator/admin/models/{model_name}` | PUT | Update model configuration |

## CLIP Score Calculation

The service uses the standard CLIP score formula:
```
CLIP Score = max(100 * cosine_similarity(image_embedding, text_embedding), 0)
```

Where embeddings are L2-normalized before similarity calculation.

## Contributing

1. **Setup Development Environment:**
   ```bash
   make dev-setup
   ```

2. **Run Tests Before Committing:**
   ```bash
   pytest tests/
   ```

3. **Test Your Changes:**
   ```bash
   # Test the service locally
   make run-local-tel
   
   # Run load tests
   make load-test
   ```

## Performance Characteristics

- **Single Evaluation**: 1 image/s depending on model
- **Batch Processing**: 5 image/s with significant speedup through vectorization
- **Memory Usage**: 2-8GB depending on model configuration
- **Throughput**: 100-500 requests/second (model dependent)
- **Error Recovery**: Failed images don't interrupt batch processing

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use smaller batch sizes or switch to 'fast' model
2. **Model Loading Errors**: Check model configuration in `config/models.json`
3. **Image Loading Failures**: Verify image URLs are accessible and formats supported
4. **Ray Serve Issues**: Check Ray Dashboard at http://localhost:8265

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG python service/run_service.py

# Check service logs
docker-compose logs -f vision-language-similarity-service
```

### Performance Optimization

- Use 'fast' model for development and real-time applications
- Increase batch size for throughput (watch memory usage)
- Enable mixed precision on CUDA devices (automatic)
- Use Ray Serve for auto-scaling production workloads
