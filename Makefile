export PROJECT_NAME := vision_language_similarity_service
export PROJECT_ALIAS := evaluator

.PHONY: help dev-setup build-base build-app build-test run-style run-style-inplace-local
.PHONY: run-unit-tests run-integration-tests run-unit-tests-local run-local
.PHONY: clean
.PHONY: run-local-otel
.PHONY: load-test

.DEFAULT_GOAL := help

# Development Setup
dev-setup: ## Set up local development environment
	scripts/build.sh dev-setup

# Build Targets
build-base: ## Build base Docker image
	scripts/build.sh build-base

build-app: build-base ## Build application Docker image
	scripts/build.sh build-app

build-test: build-base ## Build test Docker image
	scripts/build.sh build-test

# Code Quality
run-style: build-app ## Run code style checks
	scripts/build.sh run-style


run-style-inplace-local: ## Run code style fixes in-place (local)
	scripts/build.sh run-style-inplace-local

# Testing
run-unit-tests: build-test ## Run unit tests in Docker
	scripts/build.sh run-unit-tests

run-integration-tests: build-test ## Run integration tests in Docker
	scripts/build.sh run-integration-tests

run-unit-tests-local: ## Run unit tests locally
	scripts/build.sh run-unit-tests-local


# Service Deployment
run-local: build-base build-app ## Run service locally with Docker
	scripts/build.sh run-local

# Service Deployment with Observability Stack
run-local-otel: build-base build-app ## Run observability stack (Grafana, Prometheus)
	scripts/build.sh run-local-otel

clean: ## Clean all Docker resources (images, containers, volumes)
	scripts/build.sh clean

# Load Testing
load-test: ## Interactive load test (opens web UI)
	scripts/build.sh load-test


# Help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
