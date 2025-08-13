# Load Testing for Vision-Language Similarity Service

This directory contains comprehensive load testing tools for the Vision-Language Similarity evaluation service.

## Features

- **Intelligent Test Data Usage**: Automatically prioritizes local image files over remote URLs for faster, more reliable testing
- **Multiple Test Scenarios**: Health checks, single evaluations, and batch processing
- **CI/CD Integration**: Automated performance validation with configurable thresholds
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

### Interactive Load Testing

```bash
# Basic interactive load test (default: 10 users, 2/sec spawn rate)
locust -f load_tests/locustfile.py --host=http://localhost:8000

# Headless load test with custom parameters
locust -f load_tests/locustfile.py --host=http://localhost:8000 \
       --users 50 --spawn-rate 5 --run-time 5m --headless

# Light load for development testing
locust -f load_tests/locustfile.py --host=http://localhost:8000 \
       --users 5 --spawn-rate 1 --run-time 2m --headless \
       --user-class LightLoadUser

# Stress testing
locust -f load_tests/locustfile.py --host=http://localhost:8000 \
       --users 100 --spawn-rate 10 --run-time 10m --headless \
       --user-class StressTestUser
```

### CI/CD Performance Tests

```bash
# Run automated performance tests
python -m pytest load_tests/test_load_performance.py -v

# With custom service URL
LOAD_TEST_HOST=http://localhost:8000 python -m pytest load_tests/test_load_performance.py -v

# Generate HTML report
python -m pytest load_tests/test_load_performance.py --html=load_test_report.html --self-contained-html
```

## Test Data Strategy

The load tests intelligently use available test data:

1. **Local Files Priority**: Uses local PNG files from `tests/data/samples/` when available
2. **UUID Mapping**: Maps Leonardo AI URLs to local files using UUID extraction
3. **Fallback to URLs**: Falls back to remote URLs when local files aren't available
4. **Mixed Testing**: Supports base64, local files, and URL inputs

### Data Coverage

The test suite reports data availability:
- Total image-text pairs from CSV
- Available local image files
- Matched pairs (local files with captions)
- Coverage percentage

## Load Test Scenarios

### 1. Health Check Tasks (10% of traffic)
- Service health endpoint validation
- Available models endpoint testing

### 2. Single Evaluation Tasks (60% of traffic)
- Local file evaluations (highest priority)
- Base64 image evaluations
- URL-based evaluations (fallback)
- Mixed model configurations (fast/accurate)

### 3. Batch Evaluation Tasks (30% of traffic)
- Small batches (3-4 items) with local files
- Mixed batches (local files, base64, URLs)
- Configurable batch sizes and processing options

## User Classes

### VisionLanguageUser (Default)
- Realistic usage patterns
- 1-3 second wait times
- Mixed workload distribution

### LightLoadUser (CI/CD)
- Lightweight testing for pipelines
- 2-5 second wait times
- Fast model configurations only
- Prioritizes local files

### StressTestUser (Performance Testing)
- High-intensity load generation
- 0.1-0.5 second wait times
- Focuses on evaluation endpoints

## Performance Thresholds

Default performance criteria for CI/CD validation:

```python
PERFORMANCE_THRESHOLDS = {
    "avg_response_time_ms": 2000,      # Average response time under 2s
    "95th_percentile_ms": 5000,        # 95th percentile under 5s
    "max_response_time_ms": 10000,     # Max response time under 10s
    "failure_rate_percent": 1.0,       # Failure rate under 1%
    "requests_per_second": 10,         # Minimum throughput
}
```

## Integration with Existing Infrastructure

### Makefile Integration

Add to your `Makefile`:

```makefile
# Load testing targets
.PHONY: load-test load-test-light load-test-ci

load-test:
	locust -f load_tests/locustfile.py --host=http://localhost:8000

load-test-light:
	locust -f load_tests/locustfile.py --host=http://localhost:8000 \
		--users 5 --spawn-rate 1 --run-time 2m --headless \
		--user-class LightLoadUser

load-test-ci:
	python -m pytest load_tests/test_load_performance.py -v
```

### GitHub Actions Integration

```yaml
name: Performance Tests
on: [push, pull_request]

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r load_tests/requirements.txt
      
      - name: Start service
        run: |
          make run-local &
          sleep 30  # Wait for service to start
      
      - name: Run load tests
        run: python -m pytest load_tests/test_load_performance.py -v
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: load-test-results
          path: load_test_report.html
```

### Docker Integration

```dockerfile
# In your service Dockerfile, add load testing stage
FROM base-image as load-test
COPY load_tests/ /app/load_tests/
RUN pip install -r load_tests/requirements.txt
ENTRYPOINT ["python", "-m", "pytest", "load_tests/test_load_performance.py"]
```

## Monitoring Integration

The load tests work with the existing observability stack:

- **Prometheus Metrics**: Load tests generate metrics visible in Grafana
- **Tracing**: OpenTelemetry traces capture request flows
- **Logging**: Structured logs for debugging performance issues

Start the observability stack before load testing:

```bash
make run-local-otel  # Start Grafana, Prometheus, Jaeger
make load-test       # Run load tests
# View metrics at http://localhost:3000 (Grafana)
```

## Troubleshooting

### Common Issues

1. **Service Not Ready**
   - Ensure service is running and healthy
   - Check `http://localhost:8000/evaluator/health`

2. **Low Local File Coverage**
   - Verify PNG files exist in `tests/data/samples/`
   - Check UUID extraction from Leonardo AI URLs

3. **Performance Threshold Failures**
   - Review service resource allocation
   - Check for competing processes
   - Consider adjusting thresholds for your environment

### Debug Mode

```bash
# Run with verbose logging
python -m pytest load_tests/test_load_performance.py -v -s

# Run specific test
python -m pytest load_tests/test_load_performance.py::TestServicePerformance::test_light_load_performance -v
```

## Customization

### Custom Thresholds

Override performance thresholds in your CI environment:

```python
# In test_load_performance.py
CUSTOM_THRESHOLDS = {
    "avg_response_time_ms": 1500,  # Stricter requirement
    "failure_rate_percent": 0.5,   # Lower tolerance
}
```

### Additional Scenarios

Extend the load tests by adding new TaskSet classes:

```python
class CustomEvaluationTasks(TaskSet):
    @task
    def custom_scenario(self):
        # Your custom load testing scenario
        pass
```

## Results Interpretation

### Locust Web UI Metrics
- **RPS**: Requests per second (throughput)
- **Response Times**: P50, P95, P99 percentiles
- **Failure Rate**: Percentage of failed requests
- **User Distribution**: Active users across scenarios

### CI/CD Test Results
- **PASS**: All thresholds met, service performing adequately  
- **FAIL**: One or more thresholds exceeded, investigate performance issues

### Performance Baselines

Typical expected performance for the service:
- **Health endpoint**: <100ms response time
- **Single evaluation**: 1-3s (depending on model and image source)
- **Batch evaluation**: 5-15s for small batches (3-5 items)
- **Throughput**: 10-50 RPS depending on hardware and model configuration