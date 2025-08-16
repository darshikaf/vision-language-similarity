# Load Testing for Vision-Language Similarity Service

This directory contains local load testing tools for the Vision-Language Similarity evaluation service.

## Features

- **Local Development Testing**: Designed for local performance validation and optimization
- **Intelligent Test Data Usage**: Automatically prioritizes local image files over remote URLs for faster, more reliable testing
- **Multiple Test Scenarios**: Health checks, single evaluations, and batch processing
- **Real Data Testing**: Uses actual Leonardo AI dataset for realistic load simulation

## Quick Start

### Prerequisites

```bash
# Install load testing dependencies
pip install -r load_tests/requirements.txt

# Ensure the service is running
make run-local
# or for Ray Serve
make run-local-ray
```

### Local Load Testing

```bash
# Interactive load test with web UI (recommended for development)
make load-test
# Opens web UI at http://localhost:8089

# Light load test for development
make load-test-light

# Direct locust commands
locust -f load_tests/locustfile.py --host=http://localhost:8000

# Headless load test with custom parameters
locust -f load_tests/locustfile.py --host=http://localhost:8000 \
       --users 10 --spawn-rate 2 --run-time 3m --headless

# Light load for development
locust -f load_tests/locustfile.py --host=http://localhost:8000 \
       --users 5 --spawn-rate 1 --run-time 2m --headless \
       --user-class LightLoadUser
```

### Performance Validation (Local Only)

```bash
# Run all performance tests locally (includes single and batch evaluation)
python -m pytest load_tests/test_load_performance.py -v

# Run specific performance tests
python -m pytest load_tests/test_load_performance.py::TestServicePerformance::test_single_evaluation_performance -v
python -m pytest load_tests/test_load_performance.py::TestServicePerformance::test_batch_evaluation_performance -v

# Note: CI/CD integration not implemented - tests designed for local use only
```

## Performance Thresholds

Default performance criteria for local validation:

```python
PERFORMANCE_THRESHOLDS = {
    "avg_response_time_ms": 500,      # Average response time under 5ms
    "95th_percentile_ms": 1000,        # 95th percentile under 1s
    "max_response_time_ms": 10000,     # Max response time under 10s
    "failure_rate_percent": 1.0,       # Failure rate under 1%
    "requests_per_second": 10,         # Minimum throughput
}
```

## Makefile Integration

The load testing is integrated with the project's Makefile:

```makefile
# Available load testing commands
make load-test        # Interactive load test with web UI
make load-test-light  # Light load test for development
```

**Note**: CI/CD integration is not currently implemented. These tools are designed for local development and testing only.

## Monitoring Integration

The load tests work with the existing observability stack:

- **Prometheus Metrics**: Load tests generate metrics visible in Grafana
- **Logging**: Structured logs for debugging performance issues

Start the observability stack before load testing:

```bash
make run-local-otel  # Start Grafana, Prometheus, Jaeger
make load-test       # Run load tests
# View metrics at http://localhost:3000 (Grafana)
```
