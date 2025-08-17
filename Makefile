export PROJECT_NAME := vision_language_similarity_service
export PROJECT_ALIAS := evaluator

.PHONY: help dev-setup build-base build-app build-test run-style run-style-inplace run-style-inplace-local
.PHONY: run-unit-test-suite run-integration-test-suite run-unit-test-suite-local run-local
.PHONY: clean-docker-images clean-docker-compose docker-push
.PHONY: run-local-otel stop-otel clean-otel test-otel
.PHONY: build-ray-base build-ray run-local-ray stop-ray clean-ray test-ray
.PHONY: load-test load-test-light load-test-ci
.PHONY: unit-test integration-test unit-test-local

.DEFAULT_GOAL := help

# Development Setup
dev-setup: ## Set up development environment
	scripts/build.sh dev-setup

# Build Targets
build-base: ## Build base Docker image
	scripts/build.sh build-base

build-app: build-base ## Build application image
	scripts/build.sh build-app

build-test: build-base ## Build test image
	scripts/build.sh build-test

# Code Quality
run-style: build-app ## Run style checks in Docker
	scripts/build.sh run-style

run-style-inplace: ## Run style checks and fix in-place
	scripts/build.sh run-style-inplace

run-style-inplace-local: ## Run style checks locally with ruff
	scripts/build.sh run-style-inplace-local

# Testing
run-unit-test-suite: build-test clean-docker-compose ## Run unit tests in Docker
	scripts/build.sh run-unit-test-suite

run-integration-test-suite: build-test clean-docker-compose ## Run integration tests in Docker
	scripts/build.sh run-integration-test-suite

run-unit-test-suite-local: ## Run unit tests locally
	scripts/build.sh run-unit-test-suite-local

unit-test: ## Alias for run-unit-test-suite
	scripts/build.sh run-unit-test-suite

integration-test: ## Alias for run-integration-test-suite
	scripts/build.sh run-integration-test-suite

unit-test-local: ## Alias for run-unit-test-suite-local
	scripts/build.sh run-unit-test-suite-local

# Service Deployment
run-local: clean-docker-compose ## Run FastAPI service locally
	scripts/build.sh run-local

# Docker Management
clean-docker-images: ## Clean up Docker images
	scripts/build.sh clean-docker-images

clean-docker-compose: ## Clean up Docker Compose resources
	scripts/build.sh clean-docker-compose

# Observability Stack
run-local-otel: build-base build-app ## Start observability stack (Grafana, Prometheus, Jaeger)
	scripts/build.sh run-local-otel

stop-otel: ## Stop observability stack
	scripts/build.sh stop-otel

clean-otel: ## Clean observability stack and volumes
	scripts/build.sh clean-otel

# Ray Serve Deployment
build-ray-base: ## Build Ray base image
	scripts/build.sh build-ray-base

build-ray: build-ray-base ## Build Ray service image
	scripts/build.sh build-ray

run-local-ray: build-ray ## Run Ray Serve deployment
	scripts/build.sh run-local-ray

stop-ray: ## Stop Ray deployment
	scripts/build.sh stop-ray

clean-ray: ## Clean Ray deployment resources
	scripts/build.sh clean-ray

# Load Testing
load-test: ## Interactive load test (opens web UI)
	scripts/build.sh load-test

load-test-light: ## Light load test for development
	scripts/build.sh load-test-light

load-test-ci: ## CI/CD performance validation tests
	scripts/build.sh load-test-ci

# Help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
