#!/usr/bin/env bash

usage () { printf >&2 "Usage\n    $(basename $0) dev-setup|build-base|build-app|build-test|run-local|run-style|run-style-inplace-local|run-unit-tests|run-unit-tests-local|run-integration-tests|clean|load-test|run-local-otel\n"; }
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
	run-local) OPTION=run-local;;
	run-style) OPTION=run-style;;
	run-style-inplace-local) OPTION=run-style-inplace-local;;
	run-integration-tests) OPTION=run-integration-tests;;
	run-unit-tests) OPTION=run-unit-tests;;
  	run-unit-tests-local) OPTION=run-unit-tests-local;;
	clean) OPTION=clean;;
	load-test) OPTION=load-test;;
	run-local-otel) OPTION=run-local-otel;;
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


# Export for docker-compose
export BASE_IMAGE_NAME
export IMAGE_NAME
export IMAGE_NAME_LATEST
export TEST_IMAGE_NAME

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
	run-local)
		# Build prerequisites first
		echo "Building base image..."
		$0 build-base
		echo "Building app image..."
		$0 build-app
		
		echo "Starting local service..."
		docker_compose -f docker/docker-compose.local.yml up \
			--force-recreate
		;;
	run-style)
		docker run --rm -v `pwd`:/app ${IMAGE_NAME_LATEST} scripts/run_style.sh
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
	run-integration-tests)
		mkdir -p test-reports
		TEST_IMAGE_NAME=${TEST_IMAGE_NAME} \
		docker_compose -f docker/fastapi-testing/docker-compose.service.yml -f docker/fastapi-testing/docker-compose.integration-test.yml up \
			--exit-code-from $PROJECT_ALIAS \
			--force-recreate \
			--always-recreate-deps
		;;
	run-unit-tests)
		mkdir -p test-reports
		TEST_IMAGE_NAME=${TEST_IMAGE_NAME} \
		docker_compose -f docker/fastapi-testing/docker-compose.service.yml -f docker/fastapi-testing/docker-compose.unit-test.yml up \
			--exit-code-from $PROJECT_ALIAS \
			--force-recreate \
			--always-recreate-deps
		;;
	run-unit-tests-local)
		if [ -f .venv/bin/activate ]; then
			source .venv/bin/activate
			./scripts/run_test.sh --test-suite unit
		else
			echo "Virtual environment not found. Run 'make dev-setup' first."
			exit 1
		fi
		;;
	clean)
		echo "Cleaning up all Docker resources..."
		
		# Stop and remove all docker-compose services
		echo "Stopping Docker Compose services..."
		docker_compose \
			-f docker/fastapi-testing/docker-compose.service.yml \
			-f docker/docker-compose.local.yml \
			-f docker/fastapi-testing/docker-compose.unit-test.yml \
			-f docker/fastapi-testing/docker-compose.integration-test.yml \
			down --remove-orphans --volumes 2>/dev/null || true
		
		# Stop observability stack
		echo "Stopping observability stack..."
		docker_compose -f docker/observability/docker-compose.otel.yml down -v 2>/dev/null || true
		
		# Remove project-specific Docker images
		echo "Removing Docker images..."
		docker rmi ${BASE_IMAGE_NAME} 2>/dev/null || true
		docker rmi ${IMAGE_NAME_LATEST} 2>/dev/null || true
		docker rmi ${TEST_IMAGE_NAME} 2>/dev/null || true
		docker rmi ${IMAGE_NAME} 2>/dev/null || true
		
		# Clean up Docker system
		echo "Running Docker system cleanup..."
		docker system prune -f
		
		echo "Cleanup complete!"
		;;
	load-test)
		echo "Installing load test dependencies..."
		pip install -q -r load_tests/requirements.txt
		echo "Starting interactive load test..."
		echo "Open http://localhost:8089 in your browser"
		echo "Service endpoint: http://localhost:8000"
		locust -f load_tests/locustfile.py --host=http://localhost:8000
		;;
	run-local-otel)
		echo "Starting observability stack..."
		echo "Grafana: http://localhost:3000 (admin/grafana)"
		echo "Prometheus: http://localhost:9090"
		docker_compose -f docker/observability/docker-compose.otel.yml up --build --force-recreate
		;;
esac