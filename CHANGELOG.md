# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-19

### Added
- Columnar segment format with immutable, mmap-friendly segment files.
- Encodings: Plain, Dictionary, RLE, BitPack, FOR, Delta, ZSTD, LZ4.
- SIMD-accelerated scanning, filtering, and aggregation (AVX-512 / AVX2 / SSE4.2 / scalar dispatch at runtime).
- Predicate pushdown with zone maps and bloom filters for early block elimination.
- Bytecode predicate engine with compiled expression trees.
- Delta store: in-memory B+tree with MVCC for real-time inserts and deletes.
- Background merger with multi-threaded segment compaction and parallel column encoding.
- Vectorized scanner with branchless SIMD filtering and late materialization.
- CMake package with `find_package(columnarstore)` support and pkg-config file.
- Example program (`examples/basic_scan`) covering the write / open / scan flow.
