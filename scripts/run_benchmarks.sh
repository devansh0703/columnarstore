#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found. Run ./scripts/build.sh first."
    exit 1
fi

echo "Running benchmarks..."
cd "$BUILD_DIR"

BENCHMARKS=(
    "bm_encoding"
    "bm_scan"
    "bm_filter"
    "bm_aggregate"
)

for bench in "${BENCHMARKS[@]}"; do
    if [ -f "./bench/$bench" ]; then
        echo "=== Running $bench ==="
        "./bench/$bench" --benchmark_format=console --benchmark_time_unit=ms "$@"
    else
        echo "Warning: $bench not found, skipping"
    fi
done

echo "All benchmarks complete!"
