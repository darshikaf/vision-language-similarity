export PROJECT_NAME := vision_language_similarity_service
export PROJECT_ALIAS := evaluator

include scripts/Makefile

.PHONY: run-local-otel stop-otel clean-otel test-otel build-ray-base build-ray run-local-ray stop-ray clean-ray test-ray load-test load-test-light load-test-ci

run-local-otel:
	scripts/build.sh run-local-otel

stop-otel:
	scripts/build.sh stop-otel

clean-otel:
	scripts/build.sh clean-otel

test-otel:
	scripts/build.sh test-otel

build-ray-base:
	scripts/build.sh build-ray-base

build-ray: build-ray-base
	scripts/build.sh build-ray

run-local-ray: build-ray
	scripts/build.sh run-local-ray

stop-ray:
	scripts/build.sh stop-ray

clean-ray:
	scripts/build.sh clean-ray

test-ray:
	scripts/build.sh test-ray

# Load testing targets
load-test:
	scripts/build.sh load-test

load-test-light:
	scripts/build.sh load-test-light

load-test-ci:
	scripts/build.sh load-test-ci