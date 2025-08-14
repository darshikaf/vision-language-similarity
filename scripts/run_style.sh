#!/bin/bash
set -e

function help_text() {
    cat <<EOF
    Usage: $0 [ -i|--in-place-fixup FIX_FLAG ]
        --in-place-fixup	Fix imports (yes/no)
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    arg=$1
    case $arg in
    -h | --help)
        help_text
        ;;
    -i | --in-place-fixup)
        export FIX_FLAG="$2"
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
echo "SRC_DIR: $SRC_DIR"
ls -lhFa $SRC_DIR

echo "**********************"
if [ -z "${FIX_FLAG}" ]; then
    FIX_FLAG="no"
fi

if [[ "$FIX_FLAG" == "yes" ]]; then
    RUFF_CHECK_ARGS="--fix --show-fixes"
    RUFF_FORMAT_ARGS=""
else
    RUFF_CHECK_ARGS=""
    RUFF_FORMAT_ARGS="--check"
fi
echo "FIX_FLAG = $FIX_FLAG"

echo "**********************"
ruff --version

echo "**********************"
echo "Running ruff format"
ruff format ${RUFF_FORMAT_ARGS} service/

echo "**********************"
echo "Running ruff check"
ruff check ${RUFF_CHECK_ARGS} service/