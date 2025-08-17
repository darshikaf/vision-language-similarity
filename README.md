# Vision-Language Similarity Service

A vision-language similarity evaluation service built with OpenCLIP models. This service provides CLIP score calculations for comparing images with text descriptions, supporting both single and batch processing.

**Documentation**: See `challenge_brief/` directory for detailed design decisions, scaling strategies, and similarity metric performance analysis.

## Features

- **CLIP Score Evaluation**: Calculate semantic similarity between images and text using state-of-the-art OpenCLIP models
- **Batch Processing**: Efficient batch evaluation with configurable batch sizes and progress tracking
- **Multiple Model Configurations**: Support for different model profiles (fast vs accurate) with dynamic configuration
- **Dual Deployment Options**: Standard FastAPI service or advanced Ray Serve deployment with auto-scaling
- **Comprehensive Observability**: Prometheus metrics, Grafana dashboards, Jaeger tracing, and structured logging
- **Load Testing**: Integrated load testing with Locust for performance validation

## Performance Characteristics on Apple M1

- **Single Evaluation**: 4.2 images/s depending on model
- **Batch Processing**: 21.1 image/s

- **Single evaluator memory footprint**: ~980-1015MB
- **Memory persistence:** ~577MB remains after evaluator destruction (model weights)

- **Single Evaluator Memory Pattern**:
```
Baseline RSS:       405MB (before model loading)
After Model Load:   982MB (model weight loading)
Peak RSS:          1012MB (after real image evaluation) 
Memory Growth:       30MB (per evaluation session)
Persistent Memory:  577MB (model weights remain after destruction)
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Make (for build automation)

### Development Setup

__IMPORTANT__: Dev set up is only supported for `ARM64/M1`.

1. **Clone and setup the environment:**
```bash
git clone <repository-url>
cd vision-language-similarity

chmod +x dev-setup.sh
./dev-setup.sh

source .venv/bin/activate
```

2. **Run the service locally:**

#### Standard FastAPI Service on Apple M1

```bash
uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Ray Serve Deployment on Apple M1

```bash
serve run service.ray_main:deployment
```

#### fastAPI Service with Observability on Docker
```bash
# Build images
make build-base
make build-app

# Run service
make run-local-otel
```

3. **Access the service:**
- API Documentation: http://localhost:8000/evaluator/docs
- Health Check: http://localhost:8000/evaluator/health
- Metrics: http://localhost:8000/evaluator/metrics
- Grafana Dashboard: http://localhost:3000/d/vision-language-similarity/vision-language-similarity-service


## API Usage

### Single Image Evaluation

```bash
curl -X POST "http://localhost:8000/evaluator/v1/evaluation/single" \
     -H "Content-Type: application/json" \
     -d '{
       "image_input": "https://cdn.leonardo.ai/users/566cd98a-7e64-47b1-ab9d-abda10a741a7/generations/cddf3436-81d5-46af-8680-4be5077a5841/Leonardo_Diffusion_A_little_girl_wearing_a_red_dress_smiled_Pl_0.jpg",
       "text_prompt": "A little girl wearing a red dress smiled. Play with the lovely white rabbit. In a green forest. Butterflies and birds flying. On the side of the road, flowers are blooming brightly",
       "model_config_name": "fast"
     }' | jq
```

**Response:**
```json
{
  "image_input": "https://cdn.leonardo.ai/users/566cd98a-7e64-47b1-ab9d-abda10a741a7/generations/cddf3436-81d5-46af-8680-4be5077a5841/Leonardo_Diffusion_A_little_girl_wearing_a_red_dress_smiled_Pl_0.jpg",
  "text_prompt": "A little girl wearing a red dress smiled. Play with the lovely white rabbit. In a green forest. Butterflies and birds flying. On the side of the road, flowers are blooming brightly",
  "clip_score": 41.90204441547394,
  "processing_time_ms": 1108.2000732421875,
  "model_used": "fast"
}
```

### Batch Evaluation

```bash
curl -X POST "http://localhost:8000/evaluator/v1/evaluation/batch" \
     -H "Content-Type: application/json" \
     -d '{
      "evaluations": [
        {
          "image_input": "https://cdn.leonardo.ai/users/604cfbbb-25dc-4e07-998e-bcb92da145d3/generations/46babda8-86e6-480c-a43b-546e33d8fb20/Default_A_girl_studying_with_a_coffee_at_night_1.jpg",
          "text_prompt": "A girl studying with a coffee at night",
          "model_config_name": "fast"
        },
        {
          "image_input": "https://cdn.leonardo.ai/users/b81d5101-fb09-4410-9f47-b8b9ed1335df/generations/176b3af7-3cb7-497d-8d8f-3bd14a7e4c01/Default_A_pile_of_bricks_on_the_floor_2.jpg",
          "text_prompt": "A pile of bricks on the floor",
          "model_config_name": "fast"
        },
        {
          "image_input": "https://cdn.leonardo.ai/users/441e30e9-fe13-4f22-acaf-838135876ab7/generations/db00712b-deaf-463b-ba3a-759339dcf612/3D_Animation_Style_Cute_koala_bear_eating_bamboo_in_a_jungle_0.jpg",
          "text_prompt": "Cute koala bear eating bamboo in a jungle",
          "model_config_name": "fast"
        }
      ],
      "batch_size": 2
    }' | jq
```

## Command Line Interface

The repository includes a convenient CLI tool for batch evaluation of images with CLIP scores:

### Basic Usage

```bash
# Compare single vs batch performance
python cli/evaluation_cli.py cli/challenge_set.csv --compare
```

### Expected Output

The CLI tool will:
1. Process all images in your CSV file
2. Add a `clip_score` column with the evaluation results
3. Save results to `{filename}_with_scores.csv`
4. Display processing statistics and performance metrics

```bash
➜ python cli/evaluation_cli.py cli/challenge_set.csv --compare
Processing: cli/challenge_set.csv
Data directory: cli
Loaded 51 rows
Running comparison mode: both single and batch evaluation

=== Running Single Evaluation ===
Single evaluation completed: 51/51 successful
Time: 12.27s, Speed: 4.16 images/sec

=== Running Batch Evaluation ===
Batch evaluation completed: 51/51 successful
Time: 2.49s, Speed: 20.49 images/sec
Comparison statistics saved to: cli/challenge_set_comparison_stats.csv

=== Performance Comparison ===
Single: 12.27s (4.16 img/s)
Batch:  2.49s (20.49 img/s)
Speedup: 4.93x faster with batch
Results saved to: cli/challenge_set_with_scores.csv
Successfully processed 51/51 images
```

## Development

### Running Tests

```bash
# Run all tests with coverage locally
python -m pytest tests/
```

### Code Quality

```bash
# Format code with ruff
make run-style-inplace-local
```

### Load Testing

```bash
# Interactive load test with web UI
make load-test
# Web UI: http://localhost:8089
```

## Architecture

The service is built with a modular architecture supporting two deployment modes:

### Standard FastAPI Service
- **Fast Development**: Quick iteration and debugging
- **Simple Deployment**: Standard container deployment
- **Resource Efficient**: Lower resource overhead

### Ray Serve Deployment  
- **End-to-End Fault Tolerance**: Automatic replica recovery with distributed failure isolation
- **Intelligent Load Shedding**: Dynamic request throttling based on system capacity
- **Request Queuing & Backpressure**: Adaptive queuing with overflow handling
- **Auto-scaling**: ML-aware scaling based on request patterns and processing characteristics
- **Circuit Breaker Pattern**: Automatic failure detection with exponential backoff
- **Multi-Model Serving**: Traffic splitting for A/B testing and canary deployments
- **Resource Isolation**: Memory and compute isolation between replicas

### Core Components

- **Service Layer** (`service/`): FastAPI application with REST API endpoints
- **Core Logic** (`service/core/`): CLIP evaluation, device management, configuration
  - **ML Engine** (`service/core/ml/`): OpenCLIP model integration and evaluation engines
  - **ML Models** (`service/core/ml/models/`): Model abstractions and OpenCLIP implementations
  - **ML Utils** (`service/core/ml/utils/`): Image processing, metrics, and result building
- **Evaluation API** (`service/evaluation/`): Evaluation endpoints, handlers, and schemas
- **System API** (`service/system/`): Health checks and system management endpoints
- **Dynamic Configuration**: Built-in model configurations with environment variable override support

## API Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/evaluator/docs` | GET | Interactive API documentation |
| `/evaluator/health` | GET | Service health check |
| `/evaluator/v1/evaluation/single` | POST | Single image-text evaluation |
| `/evaluator/v1/evaluation/batch` | POST | Batch image-text evaluation |
| `/evaluator/metrics` | GET | Prometheus metrics |
| `/system/health` | GET | System health check |

## CLIP Score Calculation

The service uses the standard CLIP score formula:
```python
CLIP Score = max(100 * cosine_similarity(image_embedding, text_embedding), 0)
```

### Performance Optimization

- Use 'fast' model for development and real-time applications
- Increase batch size for throughput (watch memory usage)
- Enable mixed precision on CUDA devices (automatic)
- Ability to enable ensemble models for similarity calculation
- Use Ray Serve for auto-scaling production workloads

## Future Improvements for Resiliency in Production

### Performance Optimization
- **True Batch Processing**: Implement native PyTorch batching instead of concurrent single requests
- **Model Caching**: Add LRU cache with memory limits for model management
- **GPU Memory Management**: Monitor and optimize GPU memory usage
- **Feature Caching**: Cache embeddings for frequently processed images
- **Connection Pooling**: Optimize HTTP client connection reuse

### Scalability & Production
- **Auto-scaling**: Enhanced Ray Serve configuration for traffic-based scaling
- **Multi-GPU Support**: Distribute models across multiple GPUs

### Monitoring & Observability
- **Alerting**: Implement automated alerts for failures and performance degradation
- **Distributed Tracing**: Complete OpenTelemetry integration across all components
- **Performance Benchmarks**: Automated regression testing for performance
- **Security Logging**: Audit logs for security events and access patterns
