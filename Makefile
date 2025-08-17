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
dev-setup:
	scripts/build.sh dev-setup

# Build Targets
build-base:
	scripts/build.sh build-base

build-app: build-base
	scripts/build.sh build-app

build-test: build-base
	scripts/build.sh build-test

# Code Quality
run-style: build-app
	scripts/build.sh run-style

run-style-inplace:
	scripts/build.sh run-style-inplace

run-style-inplace-local:
	scripts/build.sh run-style-inplace-local

# Testing
run-unit-test-suite: build-test clean-docker-compose
	scripts/build.sh run-unit-test-suite

run-integration-test-suite: build-test clean-docker-compose
	scripts/build.sh run-integration-test-suite

run-unit-test-suite-local:
	scripts/build.sh run-unit-test-suite-local

unit-test:
	scripts/build.sh run-unit-test-suite

integration-test:
	scripts/build.sh run-integration-test-suite

unit-test-local:
	scripts/build.sh run-unit-test-suite-local

# Service Deployment
run-local: clean-docker-compose
	scripts/build.sh run-local

# Docker Management
clean-docker-images:
	scripts/build.sh clean-docker-images

clean-docker-compose:
	scripts/build.sh clean-docker-compose

# Observability Stack
run-local-otel: build-base build-app
	scripts/build.sh run-local-otel

stop-otel:
	scripts/build.sh stop-otel

clean-otel:
	scripts/build.sh clean-otel

# Ray Serve Deployment
build-ray-base:
	scripts/build.sh build-ray-base

build-ray: build-ray-base
	scripts/build.sh build-ray

run-local-ray: build-ray
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
