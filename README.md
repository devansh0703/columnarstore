# ColumnarStore

SIMD-accelerated columnar storage library for analytical workloads.

## Features

- **Columnar Storage**: Efficient column-oriented data layout with segment-based architecture
- **SIMD Acceleration**: AVX2/AVX-512 vectorized operations for encoding, filtering, and aggregation
- **Multiple Encodings**: Plain, Dictionary, RLE, Bit-Packing, Frame-of-Reference (FOR), Delta, ZSTD, LZ4
- **Predicate Pushdown**: Zone maps and Bloom filters for early row elimination
- **Delta Store**: In-memory B+tree with MVCC for real-time inserts/deletes
- **Background Merger**: Multi-threaded segment compaction with parallel column encoding
- **Vectorized Scanner**: Branchless SIMD filtering with late materialization
- **Bytecode Predicate Engine**: Compiled expression trees for SIMD evaluation

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Delta      │────▶│  Merger     │────▶│  Segments   │
│  Store      │     │  (background)    │  (immutable)│
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Zone Maps  │     │  Zone Maps  │
                    │  Bloom Filters     Bloom Filters│
                    └─────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌─────────────────────────────────┐
                    │       Vectorized Scanner        │
                    │  Zone Map Prune → Bloom → SIMD  │
                    └─────────────────────────────────┘
```

## Requirements

- C++20 compiler (GCC 10+, Clang 12+)
- CMake 3.20+
- ZSTD, LZ4, XXHash libraries
- Google Test, Google Benchmark (for tests/benchmarks)

## Building

```bash
# Quick build
./scripts/build.sh

# Debug build
./scripts/build.sh --debug

# Clean build
./scripts/build.sh --clean
```

## Running Tests

```bash
./scripts/run_tests.sh
```

## Running Benchmarks

```bash
./scripts/run_benchmarks.sh
```

## Docker

```bash
# Build image
docker-compose build

# Run tests
docker-compose run test

# Run benchmarks
docker-compose run benchmark

# Development shell
docker-compose run columnarstore
```

## Usage

```cpp
#include <columnar/segment.h>
#include <columnar/types.h>
#include <vector>

using namespace columnar;

// Create a segment
std::vector<int32_t> data(10000);
std::vector<bool> nulls(10000, false);
for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<int32_t>(i);

auto writer = SegmentWriter("/path/to/segment.col");
writer.AddColumn<DataType::Int32>(data, nulls, EncodingType::RLE);
auto segment = writer.Finish();

// Read back
auto reader = Segment::Open("/path/to/segment.col");
auto column = std::static_pointer_cast<TypedColumn<DataType::Int32>>(reader->GetColumn(0));

// Scan with predicate
Predicate pred;
pred.Equal<int32_t>(0, 5000)->And()->LessThan<int32_t>(0, 6000);

ScanContext ctx;
ctx.projected_columns = {0};
ctx.enable_zone_map_pruning = true;
ctx.enable_bloom_filter = true;

SegmentScanner scanner(reader, ctx);
scanner.Scan(pred, [](const void** cols, size_t count) {
    auto col = static_cast<const int32_t*>(cols[0]);
    for (size_t i = 0; i < count; ++i) {
        // Process filtered row
    }
});
```

## Encodings

| Encoding | Best For | Compression | Speed |
|----------|----------|-------------|-------|
| Plain | Random data | None | Fastest |
| Dictionary | Low cardinality | High | Fast |
| RLE | Long runs | High | Fast |
| BitPack | Small integer ranges | Medium | Fast |
| FOR | Sequential integers | High | Fast |
| Delta | Monotonic sequences | High | Fast |
| ZSTD | General purpose | Highest | Medium |
| LZ4 | General purpose | High | Fast |

## SIMD Operations

The library automatically detects CPU capabilities at runtime:

- **AVX-512** (AVX512F + AVX512BW + AVX512VL): 512-bit vectors, 16x int32, 8x int64
- **AVX2**: 256-bit vectors, 8x int32, 4x int64
- **SSE4.2**: 128-bit vectors, 4x int32
- **Scalar**: Fallback

Supported operations: arithmetic, comparison, bitwise, shuffle, blend, mask operations, prefix sum, compress/expand, conflict detection.

## Performance

Typical throughput on modern CPUs (AVX-512):

- **Encoding**: 1-3 GB/s per column
- **Scanning (filtered)**: 500M-2B rows/sec
- **Aggregation**: 2-5B rows/sec
- **Compression ratio**: 3-10x typical

## License

MIT License
