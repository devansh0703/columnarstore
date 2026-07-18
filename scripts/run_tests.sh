#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found. Run ./scripts/build.sh first."
    exit 1
fi

echo "Running tests..."
cd "$BUILD_DIR"

# Run unit tests with output on failure
ctest --output-on-failure --parallel $(nproc) "$@"

echo "All tests passed!"
