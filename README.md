# ColumnarStore

SIMD-accelerated columnar storage library for analytical workloads.

[![CI](https://github.com/devansh0703/columnarstore/actions/workflows/ci.yml/badge.svg)](https://github.com/devansh0703/columnarstore/actions/workflows/ci.yml)

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
- ZSTD, LZ4 libraries (auto-fetched via FetchContent if not found)
- Google Test, Google Benchmark (for tests/benchmarks; auto-fetched if not found)

## Building

```bash
# Quick build
./scripts/build.sh

# Debug build (ASan + UBSan enabled)
./scripts/build.sh --debug

# Clean build
./scripts/build.sh --clean
```

## Installing

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DCOLUMNARSTORE_BUILD_EXAMPLES=OFF
cmake --build build
sudo cmake --install build     # installs to /usr/local
```

Consumers can then use the library from CMake:

```cmake
find_package(columnarstore 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE columnarstore::columnarstore)
```

A pkg-config file (`columnarstore.pc`) is also installed for Makefile-based
projects, and `make package` produces binary/source tarballs via CPack.

## Running Tests

```bash
./scripts/run_tests.sh
```

## Running Benchmarks

```bash
./scripts/run_benchmarks.sh
```

## Example

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCOLUMNARSTORE_BUILD_EXAMPLES=ON
cmake --build build --target basic_scan
./build/examples/basic_scan
```

## Usage

```cpp
#include <columnar/segment.h>
#include <columnar/scanner.h>
#include <columnar/predicate.h>
#include <vector>

using namespace columnar;
using namespace columnar::scanner;
using namespace columnar::predicate;

// 1. Write a segment (immutable, column-oriented)
std::vector<int32_t> ids(100000);
std::vector<int32_t> values(100000);
for (size_t i = 0; i < ids.size(); ++i) {
    ids[i] = static_cast<int32_t>(i);
    values[i] = static_cast<int32_t>(i % 64);
}

auto writer = SegmentWriter("/tmp/data.col");
writer.AddColumn<DataType::Int32>(ids, std::vector<bool>(100000, false), EncodingType::Delta);
writer.AddColumn<DataType::Int32>(values, std::vector<bool>(100000, false), EncodingType::RLE);
auto segment = writer.Finish();

// 2. Reopen from disk
auto opened = Segment::Open("/tmp/data.col");

// 3. Build a compiled predicate and scan.
//    Zone maps and bloom filters prune blocks before row evaluation.
auto pred = PredicateBuilder()
                .Lt<DataType::Int32>(0, 50000)   // id < 50000
                .And()
                .Eq<DataType::Int32>(1, 7)       // value == 7
                .Build();

ScanContext ctx;
ctx.enable_zone_map_pruning = true;
ctx.enable_bloom_filter = true;

SegmentScanner scanner(opened, ctx);
scanner.Scan(*pred, [](const void** cols, size_t count) {
    auto* col = static_cast<const int32_t*>(cols[0]);
    for (size_t i = 0; i < count; ++i) {
        // Process matching row
    }
});

const ScanStats& stats = scanner.Stats();
// stats.blocks_pruned_zone_map, stats.rows_returned, ...
```

See `examples/basic_scan.cpp` for a complete, runnable program.

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

Measured with `bench/bm_compare.cpp` — see [BENCHMARKS.md](BENCHMARKS.md) for
tables, methodology, and honest limits. Headlines (1M int32 rows, AVX2 CPU,
GCC 15, Release):

- **~1.09 G rows/s** end-to-end scan at ~1% selectivity (4.2x a scalar scan)
- **13.6x speedup** from zone-map pruning on clustered keys
- 2.5 G rows/s raw SIMD filter kernel; 5 G rows/s memcpy bandwidth ceiling
- High-selectivity scans with random matches are currently delivery-bound
  (~86 M rows/s) — see the honest breakdown in BENCHMARKS.md

## Versioning

Releases are tagged `vX.Y.Z`; the version single source of truth is the
`VERSION` file. See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

MIT License — see [LICENSE](LICENSE).
