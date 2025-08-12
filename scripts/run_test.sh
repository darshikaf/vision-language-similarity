#!/bin/bash
set -e

function help_text() {
    cat <<EOF
    Usage: $0 [ -t|--test-suite TEST_SUITE ]
        --test-suite	Test suite (unit/integration)
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    arg=$1
    case $arg in
    -h | --help)
        help_text
        ;;
    -t | --test-suite)
        export TEST_SUITE="$2"
        shift
        shift
        ;;
    *)
        echo "ERROR: Unrecognised option: ${arg}"
        help_text
        exit 1
        ;;
    esac
done

echo "**********************"
if [ -z "${SRC_DIR}" ]; then
    SRC_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"
fi

if [ -z "${TEST_SUITE}" ]; then
    TEST_SUITE="unit"
fi

# Determine Python executable
if command -v python3 &> /dev/null; then
    PY_EXE=python3
elif command -v python &> /dev/null; then
    PY_EXE=python
else
    echo "ERROR: Python not found"
    exit 1
fi

# Try to get version from service module
if ${PY_EXE} -c "import service" 2>/dev/null; then
    SOURCE_DIR=$(${PY_EXE} -c "import service; print(list(service.__path__)[0])")
    VERSION=$(${PY_EXE} -c "
try:
    from service import __version__
    print(__version__)
except:
    import os
    if os.path.exists('service/_version'):
        with open('service/_version') as f:
            print(f.read().strip())
    else:
        print('dev')
" 2>/dev/null || echo "dev")
else
    SOURCE_DIR="service"
    VERSION="dev"
fi

echo "SRC_DIR: $SRC_DIR"
echo "Python: $PY_EXE"
echo "Python Path: $PYTHONPATH" 
echo "Path: $PATH"
echo "Version: $VERSION"
echo "Test Suite: $TEST_SUITE"

echo "********************** pytest"
mkdir -p /tmp/test-reports

# Run pytest with coverage
${PY_EXE} -m pytest tests/$TEST_SUITE --cov=service --cov-report=html --cov-report=json --cov-report=term-missing --junitxml=/tmp/test-reports/pytest-junit.xml