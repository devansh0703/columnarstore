#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BUILD_TYPE="${1:-Release}"
BUILD_DIR="$PROJECT_ROOT/build"

echo "Building ColumnarStore ($BUILD_TYPE)..."
echo "Project root: $PROJECT_ROOT"
echo "Build directory: $BUILD_DIR"

cmake -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=ON \
    -DENABLE_AVX512=ON \
    "$PROJECT_ROOT"

cmake --build "$BUILD_DIR" --parallel $(nproc)

echo "Build complete!"
echo "Binaries in: $BUILD_DIR"
