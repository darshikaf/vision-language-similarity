#!/usr/bin/env bash

usage () { printf >&2 "Usage\n    $(basename $0) dev-setup|build-base|build-app|build-test|build-ray-base|build-ray|run-local|run-ray-local|run-style|run-unit-test-suite|run-integration-test-suite|clean-docker-images|clean-docker-compose|load-test|load-test-light|load-test-ci|run-local-otel|stop-otel|clean-otel|test-otel|run-local-ray|stop-ray|clean-ray|test-ray\n"; }
usage_error () { printf >&2 "$(basename $0): $1\n"; usage; exit 2; }

if [[ $# -ne 1 ]]; then
	usage_error "One argument expected"
fi

# Check if PROJECT_NAME is set
if [[ -z "$PROJECT_NAME" ]]; then
    echo "ERROR: PROJECT_NAME is not set. Please set it and try again."
    exit 1
fi

# Check if PROJECT_ALIAS is set
if [[ -z "$PROJECT_ALIAS" ]]; then
    echo "ERROR: PROJECT_ALIAS is not set. Please set it and try again."
    exit 1
fi

case "$1" in
	dev-setup) OPTION=dev-setup;;
	build-base) OPTION=build-base;;
	build-app) OPTION=build-app;;
	build-test) OPTION=build-test;;
	build-ray-base) OPTION=build-ray-base;;
	build-ray) OPTION=build-ray;;
	run-local) OPTION=run-local;;
	run-ray-local) OPTION=run-ray-local;;
	run-style) OPTION=run-style;;
	run-style-inplace) OPTION=run-style-inplace;;
	run-style-inplace-local) OPTION=run-style-inplace-local;;
	run-integration-test-suite) OPTION=run-integration-test-suite;;
	run-unit-test-suite) OPTION=run-unit-test-suite;;
  	run-unit-test-suite-local) OPTION=run-unit-test-suite-local;;
	clean-docker-images) OPTION=clean-docker-images;;
	clean-docker-compose) OPTION=clean-docker-compose;;
  	docker-push) OPTION=docker-push;;
	load-test) OPTION=load-test;;
	load-test-light) OPTION=load-test-light;;
	load-test-ci) OPTION=load-test-ci;;
	run-local-otel) OPTION=run-local-otel;;
	stop-otel) OPTION=stop-otel;;
	clean-otel) OPTION=clean-otel;;
	test-otel) OPTION=test-otel;;
	run-local-ray) OPTION=run-local-ray;;
	stop-ray) OPTION=stop-ray;;
	clean-ray) OPTION=clean-ray;;
	test-ray) OPTION=test-ray;;
	*) usage_error "Unknown option"
esac

# For local development, we don't need AWS ECR
# Use local registry or Docker Hub for personal projects
if [ -n "$CI" ]
then
  # CI environment - could be GitHub Actions, GitLab CI, etc.
  REGISTRY_PREFIX="your-registry/"
else
  # Local development
  REGISTRY_PREFIX="local/"
fi

REPOSITORY=${REGISTRY_PREFIX}${PROJECT_NAME}

function str_to_int() {
    local s="$1"  # input string
    local base=62
    local alphabet="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Initialize the result
    local num=0

    # Go through each character in the string
    for (( i=0; i<${#s}; i++ )); do
        local char=${s:$i:1}
        # Find the position of the character in the alphabet
        local pos=$(echo "$alphabet" | awk -v char="$char" '{print index($0, char)}')
        pos=$(( pos - 1 ))

        # num = num * base + pos
        num=$(( num * base + pos ))
    done

    echo ${num#-}
}

# Generate version from git
if [ -z "$CI_COMMIT_TAG" ]
then
  if git describe --tags >/dev/null 2>&1; then
    GIT_VER=$(git describe --tags) # this will look like 2.7.0-10-gbee1c05
    GIT_VER=${GIT_VER/-/.dev}  # replace the first dash with a dot and add dev
    PREFIX=${GIT_VER%-*} # get the first part, for example 2.7.0.dev10
    SUFFIX=${GIT_VER##*-} # get the last part, for example gbee1c05
    SUFFIX_INT=$(str_to_int $SUFFIX) # convert the last part to an integer
    VERSION=${PREFIX}${SUFFIX_INT} # combine the two parts: Version: 2.7.0.dev1056983669303565
  else
    # No tags yet, use commit hash
    VERSION="0.1.0.dev$(git rev-parse --short HEAD)"
  fi
  echo Building for git commit... Version: ${VERSION}
else
  echo Building for git release... Version: ${CI_COMMIT_TAG}
  VERSION=${CI_COMMIT_TAG}
fi

# On macOS, we don't have a sha1sum function, so we create one
if [[ "$OSTYPE" == "darwin"* ]]; then
  function sha1sum() { shasum -a 1 "$@" ; } && export -f sha1sum
fi

# Docker Compose compatibility function - supports both v1 and v2
function docker_compose() {
  if command -v docker-compose &> /dev/null; then
    # Docker Compose v1 (standalone)
    docker-compose "$@"
  elif docker compose version &> /dev/null; then
    # Docker Compose v2 (plugin)
    docker compose "$@"
  else
    echo "ERROR: Neither 'docker-compose' (v1) nor 'docker compose' (v2) is available."
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
  fi
}

# rebuild base image on any change of relevant files
BASE_IMAGE_VERSION=$(cat docker/base.Dockerfile pyproject.toml | sha1sum | awk '{ print $1 }')

BASE_IMAGE_NAME=${REPOSITORY}:base.${BASE_IMAGE_VERSION}
IMAGE_NAME=${REPOSITORY}:${VERSION}
# this one is for local development
IMAGE_NAME_LATEST=${REPOSITORY}:latest
TEST_IMAGE_NAME=${REPOSITORY}:test.${VERSION}

# Ray Serve image names
RAY_BASE_IMAGE_VERSION=$(cat docker/ray/ray-base.Dockerfile pyproject.toml | sha1sum | awk '{ print $1 }')
RAY_BASE_IMAGE_NAME=${REPOSITORY}:ray-base.${RAY_BASE_IMAGE_VERSION}
RAY_IMAGE_NAME=${REPOSITORY}:ray.${VERSION}
RAY_IMAGE_NAME_LATEST=${REPOSITORY}:ray.latest

# Export for docker-compose
export BASE_IMAGE_NAME
export IMAGE_NAME
export IMAGE_NAME_LATEST
export TEST_IMAGE_NAME
export RAY_BASE_IMAGE_NAME
export RAY_IMAGE_NAME
export RAY_IMAGE_NAME_LATEST

case "$OPTION" in
	dev-setup)
		echo "Setting up development environment..."
		if ! command -v uv &> /dev/null; then
		    echo "Installing uv..."
		    curl -LsSf https://astral.sh/uv/install.sh | sh
		    export PATH="$HOME/.cargo/bin:$PATH"
		fi
		echo "Creating virtual environment and installing dependencies..."
		uv venv
		source .venv/bin/activate
		uv pip install -e ".[dev]"
		echo "Development environment setup complete!"
		echo "To activate: source .venv/bin/activate"
		;;
	build-base)
		echo Using base image tag: ${BASE_IMAGE_VERSION}
		# Check if image exists locally first
		if docker image inspect ${BASE_IMAGE_NAME} >/dev/null 2>&1; then
			echo "Base Docker image exists locally -- ${BASE_IMAGE_NAME}"
		else
			echo "Building base image -- ${BASE_IMAGE_NAME}"
			docker build \
				--progress plain \
				-t ${BASE_IMAGE_NAME} \
				-f docker/base.Dockerfile .
		fi
		;;
	build-app)
		docker build \
			--build-arg BASE_IMAGE_NAME=${BASE_IMAGE_NAME} \
			--build-arg VERSION=${VERSION} \
			--progress plain \
			-t ${IMAGE_NAME} \
			-t ${IMAGE_NAME_LATEST} \
			-f docker/service.Dockerfile .
		;;
	build-test)
		docker build \
			--build-arg BASE_IMAGE_NAME=${BASE_IMAGE_NAME} \
			--build-arg VERSION=${VERSION} \
			--progress plain \
			--no-cache \
			-t ${TEST_IMAGE_NAME} \
			-f docker/fastapi-testing/test.Dockerfile . \
			|| exit $?
		;;
	build-ray-base)
		echo Using Ray base image tag: ${RAY_BASE_IMAGE_VERSION}
		# Check if image exists locally first
		if docker image inspect ${RAY_BASE_IMAGE_NAME} >/dev/null 2>&1; then
			echo "Ray base Docker image exists locally -- ${RAY_BASE_IMAGE_NAME}"
		else
			echo "Building Ray base image -- ${RAY_BASE_IMAGE_NAME}"
			docker build \
				--progress plain \
				-t ${RAY_BASE_IMAGE_NAME} \
				-f docker/ray/ray-base.Dockerfile .
		fi
		;;
	build-ray)
		docker build \
			--build-arg BASE_IMAGE_NAME=${RAY_BASE_IMAGE_NAME} \
			--build-arg VERSION=${VERSION} \
			--progress plain \
			-t ${RAY_IMAGE_NAME} \
			-t ${RAY_IMAGE_NAME_LATEST} \
			-f docker/ray/ray-service.Dockerfile .
		;;
	run-local)
		docker_compose -f docker/fastapi-testing/docker-compose.service.yml -f docker/docker-compose.local.yml up \
			--exit-code-from $PROJECT_ALIAS \
			--force-recreate \
			--always-recreate-deps
		;;
	run-ray-local)
		# TODO: Create Ray docker-compose files
		echo "Ray docker-compose files not yet created. Please create docker/ray/docker-compose.ray-service.yml and docker/ray/docker-compose.ray-local.yml"
		# docker_compose -f docker/ray/docker-compose.ray-service.yml -f docker/ray/docker-compose.ray-local.yml up \
			--exit-code-from $PROJECT_ALIAS \
			--force-recreate \
			--always-recreate-deps
		;;
	run-style)
		docker run --rm -v `pwd`:/app ${IMAGE_NAME_LATEST} scripts/run_style.sh
		;;
	run-style-inplace)
		docker run --rm -v `pwd`:/app ${TEST_IMAGE_NAME} scripts/run_style.sh --in-place-fixup yes
		;;
	run-style-inplace-local)
		if [ -f .venv/bin/activate ]; then
			source .venv/bin/activate
			./scripts/run_style.sh -i yes
		else
			echo "Virtual environment not found. Run 'make dev-setup' first."
			exit 1
		fi
		;;
	run-integration-test-suite)
		mkdir -p test-reports
		TEST_IMAGE_NAME=${TEST_IMAGE_NAME} \
		docker_compose -f docker/fastapi-testing/docker-compose.service.yml -f docker/fastapi-testing/docker-compose.integration-test.yml up \
			--exit-code-from $PROJECT_ALIAS \
			--force-recreate \
			--always-recreate-deps
		;;
	run-unit-test-suite)
		mkdir -p test-reports
		TEST_IMAGE_NAME=${TEST_IMAGE_NAME} \
		docker_compose -f docker/fastapi-testing/docker-compose.service.yml -f docker/fastapi-testing/docker-compose.unit-test.yml up \
			--exit-code-from $PROJECT_ALIAS \
			--force-recreate \
			--always-recreate-deps
		;;
	run-unit-test-suite-local)
		if [ -f .venv/bin/activate ]; then
			source .venv/bin/activate
			./scripts/run_test.sh --test-suite unit
		else
			echo "Virtual environment not found. Run 'make dev-setup' first."
			exit 1
		fi
		;;
	clean-docker-images)
		docker rmi ${BASE_IMAGE_NAME} 2>/dev/null || true
		docker rmi ${IMAGE_NAME_LATEST} 2>/dev/null || true
		docker rmi ${TEST_IMAGE_NAME} 2>/dev/null || true
		docker rmi ${IMAGE_NAME} 2>/dev/null || true
		;;
	clean-docker-compose)
		docker_compose \
			-f docker/fastapi-testing/docker-compose.service.yml \
			-f docker/docker-compose.local.yml \
			-f docker/fastapi-testing/docker-compose.unit-test.yml \
			-f docker/fastapi-testing/docker-compose.integration-test.yml \
			down --remove-orphans --volumes 2>/dev/null || true
		;;
	docker-push)
		if [ -n "$CI" ]; then
			docker push ${IMAGE_NAME}
			docker push ${IMAGE_NAME_LATEST}
		else
			echo "Docker push is only supported in CI environment"
		fi
		;;
	load-test)
		echo "Installing load test dependencies..."
		pip install -q -r load_tests/requirements.txt
		echo "Starting interactive load test..."
		echo "Open http://localhost:8089 in your browser"
		echo "Service endpoint: http://localhost:8000"
		locust -f load_tests/locustfile.py --host=http://localhost:8000
		;;
	load-test-light)
		echo "Installing load test dependencies..."
		pip install -q -r load_tests/requirements.txt
		echo "Running light load test (5 users, 2 minutes)..."
		locust -f load_tests/locustfile.py --host=http://localhost:8000 \
			--users 5 --spawn-rate 1 --run-time 2m --headless \
			--user-class LightLoadUser
		;;
	load-test-ci)
		echo "Installing load test dependencies..."
		pip install -q -r load_tests/requirements.txt
		echo "Running CI/CD performance tests..."
		python -m pytest load_tests/test_load_performance.py -v
		;;
	run-local-otel)
		echo "Starting observability stack..."
		echo "Grafana: http://localhost:3000 (admin/grafana)"
		echo "Prometheus: http://localhost:9090"
		echo "Jaeger: http://localhost:16686"
		docker_compose -f docker/observability/docker-compose.otel.yml up --build --force-recreate
		;;
	stop-otel)
		docker_compose -f docker/observability/docker-compose.otel.yml down
		;;
	clean-otel)
		docker_compose -f docker/observability/docker-compose.otel.yml down -v
		docker system prune -f
		;;
	test-otel)
		curl -s http://localhost:8000/evaluator/health | jq .
		curl -s -X POST http://localhost:8000/evaluator/v1/evaluation/single \
		  -H "Content-Type: application/json" \
		  -d '{"image_input": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", "text_prompt": "test", "model_config_name": "fast"}' | jq .
		;;
	run-local-ray)
		echo "Starting Ray Serve stack..."
		echo "Service: http://localhost:8000/evaluator/docs"
		echo "Ray Dashboard: http://localhost:8265"
		echo "Health Check: http://localhost:8000/evaluator/health"
		echo ""
		echo "Starting Ray Serve container..."
		docker run --rm -p 8000:8000 -p 8265:8265 --shm-size=4gb \
			-e RAY_SERVE_ENABLE_SCALING=1 \
			-e RAY_DISABLE_DOCKER_CPU_WARNING=1 \
			-e RAY_DEDUP_LOGS=0 \
			-e RAY_OBJECT_STORE_ALLOW_SLOW=1 \
			-e RAY_memory_monitor_refresh_ms=0 \
			-e RAY_task_queue_timeout_ms=100 \
			${RAY_IMAGE_NAME_LATEST} \
			python service/ray_main.py
		;;
	stop-ray)
		echo "Stopping Ray Serve containers..."
		docker ps -q --filter "ancestor=${RAY_IMAGE_NAME_LATEST}" | xargs -r docker stop
		echo "Ray Serve containers stopped."
		;;
	clean-ray)
		echo "Cleaning Ray Serve containers and images..."
		docker ps -q --filter "ancestor=${RAY_IMAGE_NAME_LATEST}" | xargs -r docker stop
		docker system prune -f
		echo "Ray Serve cleanup complete."
		;;
	test-ray)
		curl -s http://localhost:8000/evaluator/health | jq .
		curl -s -X POST http://localhost:8000/evaluator/v1/evaluation/single \
		  -H "Content-Type: application/json" \
		  -d '{"image_input": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", "text_prompt": "test", "model_config_name": "fast"}' | jq .
		;;
esac