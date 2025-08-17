# Load Testing for Vision-Language Similarity Service

This directory contains local load testing tools for the Vision-Language Similarity evaluation service.

## Quick Start

### Prerequisites

```bash
# dev environment should support load testing.
# If not, install load testing dependencies
pip install -r load_tests/requirements.txt

# Ensure the service is running with FastAPI only
make run-local-otel

# or for Ray Serve
make run-local-ray
```

### Local Load Testing

```bash
# Interactive load test with web UI (recommended for development)
make load-test
# Opens web UI at http://localhost:8089
```

**Note**: CI/CD integration is not currently implemented. These tools are designed for local development and testing only.

## Monitoring Integration

The load tests work with the existing observability stack:

- **Prometheus Metrics**: Load tests generate metrics visible in Grafana
