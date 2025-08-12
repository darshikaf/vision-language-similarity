.PHONY: build clean run-local test-unit test-integration style-check style-inplace dev-setup
.DEFAULT_GOAL := help

dev-setup:
	scripts/build.sh dev-setup

build-base:
	scripts/build.sh build-base

build-app: build-base
	scripts/build.sh build-app

build-test: build-base
	scripts/build.sh build-test

run-style: build-app
	scripts/build.sh run-style

run-style-inplace:
	scripts/build.sh run-style-inplace

run-style-inplace-local:
	scripts/build.sh run-style-inplace-local

run-unit-test-suite: build-test clean-docker-compose
	scripts/build.sh run-unit-test-suite

run-integration-test-suite: build-test clean-docker-compose
	scripts/build.sh run-integration-test-suite

run-unit-test-suite-local:
	scripts/build.sh run-unit-test-suite-local

run-local: clean-docker-compose
	scripts/build.sh run-local

clean-docker-images:
	scripts/build.sh clean-docker-images

clean-docker-compose:
	scripts/build.sh clean-docker-compose

docker-push:
	scripts/build.sh docker-push

help:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'