# Vision-Language Similarity Service

A high-performance vision-language similarity evaluation service built with OpenCLIP models. This service provides CLIP score calculations for comparing images with text descriptions, supporting both single and batch processing with comprehensive observability, profiling, and advanced ML serving capabilities.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Development Setup](#development-setup)
  - [Ray Serve Deployment (Advanced ML Serving)](#ray-serve-deployment-advanced-ml-serving)
- [API Usage](#api-usage)
  - [Single Image Evaluation](#single-image-evaluation)
  - [Batch Evaluation](#batch-evaluation)
  - [Health Check](#health-check)
- [Command Line Interface](#command-line-interface)
  - [Basic Usage](#basic-usage)
  - [CSV File Format](#csv-file-format)
  - [CLI Options](#cli-options)
  - [Output](#output)
  - [Example Run](#example-run)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Code Quality](#code-quality)
  - [Observability Stack](#observability-stack)
  - [Load Testing](#load-testing)
  - [Performance Profiling](#performance-profiling)
- [Docker Deployment](#docker-deployment)
  - [Standard FastAPI Service](#standard-fastapi-service)
  - [Ray Serve Deployment](#ray-serve-deployment)
- [Architecture](#architecture)
  - [Standard FastAPI Service](#standard-fastapi-service-1)
  - [Ray Serve Deployment](#ray-serve-deployment-1)
  - [Core Components](#core-components)
- [API Endpoints](#api-endpoints)
- [CLIP Score Calculation](#clip-score-calculation)
- [Contributing](#contributing)
- [Performance Characteristics](#performance-characteristics)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Debug Mode](#debug-mode)
  - [Performance Optimization](#performance-optimization)
- [Future Improvements](#future-improvements)
  - [Performance Optimization](#performance-optimization-1)
  - [Scalability & Production](#scalability--production)
  - [Monitoring & Observability](#monitoring--observability)
  - [Testing & Quality](#testing--quality)


## Features

- **CLIP Score Evaluation**: Calculate semantic similarity between images and text using state-of-the-art OpenCLIP models
- **Batch Processing**: Efficient batch evaluation with configurable batch sizes and progress tracking
- **Multiple Model Configurations**: Support for different model profiles (fast vs accurate) with dynamic configuration
- **Dual Deployment Options**: Standard FastAPI service or advanced Ray Serve deployment with auto-scaling
- **Comprehensive Observability**: Prometheus metrics, Grafana dashboards, Jaeger tracing, and structured logging
- **Load Testing**: Integrated load testing with Locust for performance validation

## Performance Characteristics on Apple M1

- **Single Evaluation**: 4.2 images/s depending on model
- **Batch Processing**: 21.1 image/s with significant speedup through vectorization
- **Memory Usage**: 2-8GB depending on model configuration
- **Throughput**: 100-500 requests/second (model dependent)
- **Error Recovery**: Failed images don't interrupt batch processing

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
   make dev-setup
   ```

2. **Run the service locally:**
   ```bash
   # Standard FastAPI service
   make run-local

   # Ray Serve on local
   make run-local-ray
   
   # Or run directly with Python
   uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Access the service:**
   - API Documentation: http://localhost:8000/evaluator/docs
   - Health Check: http://localhost:8000/evaluator/health
   - Metrics: http://localhost:8000/evaluator/metrics

### Ray Serve Deployment (Advanced ML Serving)

For production ML workloads requiring enterprise-grade reliability and scaling:

```bash
# Build Ray Serve images
make build-ray-base
make build-ray

# Run Ray Serve deployment
make run-local-ray
```

**Advanced Production Features:**

- **End-to-End Fault Tolerance**: Automatic replica recovery, distributed failure isolation, and graceful degradation
- **Intelligent Load Shedding**: Dynamic request throttling based on system capacity and queue depth
- **Request Queuing & Backpressure**: Adaptive queuing with automatic overflow handling and client backpressure
- **Auto-scaling**: ML-aware scaling based on request patterns, queue depth, and processing characteristics
- **Circuit Breaker Pattern**: Automatic failure detection with exponential backoff and recovery
- **Multi-Model Serving**: Traffic splitting for A/B testing and canary deployments
- **Resource Isolation**: Memory and compute isolation between model replicas

**Monitoring & Control:**
- Ray Dashboard: http://localhost:8265
- API Documentation: http://localhost:8000/evaluator/docs
- Replica health monitoring and automatic replacement
- Real-time metrics for request queue, processing latency, and failure rates

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

## Command Line Interface

The repository includes a convenient CLI tool for batch evaluation of images with CLIP scores:

### Basic Usage

```bash
# Evaluate images from a CSV file
cd cli
python evaluation_cli.py path/to/your_dataset.csv

# Use batch evaluation (faster for multiple images)
python evaluation_cli.py path/to/your_dataset.csv --batch --batch-size 32

# Compare single vs batch performance
python evaluation_cli.py path/to/your_dataset.csv --compare

# Use custom service URL
python evaluation_cli.py path/to/your_dataset.csv --service-url http://localhost:8000
```

### CSV File Format

Your CSV file should contain at least these columns:
- `url`: Image URL or file path
- `caption`: Text description to compare with the image

Example CSV:
```csv
url,caption
https://example.com/image1.jpg,A cat sitting on a table
https://example.com/image2.jpg,A dog running in a park
/path/to/local/image.jpg,A beautiful sunset over mountains
```

### CLI Options

- `--batch`: Use batch evaluation API (recommended for multiple images)
- `--batch-size N`: Set batch size (default: 32)
- `--compare`: Run both single and batch evaluation to compare performance
- `--service-url URL`: Custom service URL (default: http://localhost:8000)

### Output

The CLI tool will:
1. Process all images in your CSV file
2. Add a `clip_score` column with the evaluation results
3. Save results to `{filename}_with_scores.csv`
4. Display processing statistics and performance metrics

### Example Run

```bash
➜ python cli/evaluation_cli.py tests/data/samples/challenge_set.csv --compare
Processing: tests/data/samples/challenge_set.csv
Data directory: tests/data/samples
Loaded 51 rows
Running comparison mode: both single and batch evaluation

=== Running Single Evaluation ===
Single evaluation completed: 51/51 successful
Time: 46.73s, Speed: 1.09 images/sec

=== Running Batch Evaluation ===
Batch evaluation completed: 51/51 successful
Time: 8.81s, Speed: 5.79 images/sec
Comparison statistics saved to: tests/data/samples/challenge_set_comparison_stats.csv

=== Performance Comparison ===
Single: 46.73s (1.09 img/s)
Batch:  8.81s (5.79 img/s)
Speedup: 5.31x faster with batch
Results saved to: tests/data/samples/challenge_set_with_scores.csv
Successfully processed 51/51 images
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
# Grafana Dashboard: http://localhost:3000/d/vision-language-similarity/vision-language-similarity-service 
# Prometheus: http://localhost:9090

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
```
CLIP Score = max(100 * cosine_similarity(image_embedding, text_embedding), 0)
```

Where embeddings are L2-normalized before similarity calculation.

### Performance Optimization

- Use 'fast' model for development and real-time applications
- Increase batch size for throughput (watch memory usage)
- Enable mixed precision on CUDA devices (automatic)
- Enable ensemble models for similarity calculation
- Intelligent prompt classification
- Use Ray Serve for auto-scaling production workloads

## Future Improvements

### Performance Optimization
- **True Batch Processing**: Implement native PyTorch batching instead of concurrent single requests
- **Model Caching**: Add LRU cache with memory limits for model management
- **GPU Memory Management**: Monitor and optimize GPU memory usage
- **Feature Caching**: Cache embeddings for frequently processed images
- **Connection Pooling**: Optimize HTTP client connection reuse

### Scalability & Production
- **Graceful Shutdown**: Implement proper cleanup on service termination
- **Health Checks**: Add comprehensive health monitoring including model status
- **Auto-scaling**: Enhanced Ray Serve configuration for traffic-based scaling
- **Multi-GPU Support**: Distribute models across multiple GPUs
- **Distributed Caching**: Redis-based caching for multi-replica deployments

### Monitoring & Observability
- **Enhanced Metrics**: Add model-specific performance and accuracy metrics
- **Alerting**: Implement automated alerts for failures and performance degradation
- **Distributed Tracing**: Complete OpenTelemetry integration across all components
- **Performance Benchmarks**: Automated regression testing for performance
- **Security Logging**: Audit logs for security events and access patterns

### Testing & Quality
- **Test Coverage**: Expand coverage to include Ray Serve and error paths
- **Load Testing**: Automated performance validation in CI/CD
- **Security Testing**: Penetration testing and vulnerability scanning
- **Chaos Engineering**: Fault injection testing for resilience validation
