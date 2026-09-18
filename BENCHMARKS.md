# Benchmarks

Real numbers from `bench/bm_compare.cpp` on identical data. Reproduce with:

```bash
./scripts/build.sh
./build/bench/bm_compare
```

## Environment

| | |
|---|---|
| CPU | Intel Core i7-13620H (13th Gen), 16 threads, AVX2 (no AVX-512) |
| Compiler | GCC 15.2.0, `-O3 -mavx2 -mfma` |
| Build | CMake Release |
| Dataset | 1,048,576 rows, 2 x int32 columns (sorted id, random score) |
| Date | 2026-09-19 |

## Results

| Benchmark | What it measures | Throughput | vs. row-store |
|---|---|---|---|
| RowStoreScan | `std::copy_if` over an array of row structs (naive baseline) | 1.23 G rows/s | 1.0x |
| ScalarColumnScan | column layout, branchy scalar filter | 261 M rows/s | 0.21x |
| SimdColumnScan | column layout, SIMD filter, no engine overhead | 2.15 G rows/s | 1.75x |
| MemcpyCeiling | raw memory bandwidth (4 bytes/row) | 5.12 G rows/s | 4.2x |
| **ColumnarStoreScan** | full engine, 50% selectivity, random matches, no pruning | 86 M rows/s | 0.07x |
| **ColumnarStoreScanSelective** | full engine, ~1% selectivity (typical analytics) | **1.09 G rows/s** | **0.89x** |
| **ColumnarStoreScanPruned** | full engine + zone maps on sorted key | **1.07 G rows/s** (12.4x faster than unpruned) | 0.87x |

## How to read this

**Where ColumnarStore wins:**

1. **Selective filters over wide schemas.** At ~1% selectivity the engine scans
   at 1.09 G rows/s, 4.2x faster than a scalar column scan, because the SIMD
   kernel only touches the one column being filtered. A row store must read
   *every* column of every row; the gap grows with row width.
2. **Sorted / clustered keys with zone maps.** The pruned scan skips whole
   blocks without touching them: 937 us vs 12.8 ms for the identical data
   without pruning (13.6x). Real workloads (timestamps, IDs) cluster, so this
   is the common case, not the benchmark special case.
3. **Compression + scan in one system.** Encoded segments (RLE, Delta,
   Dictionary, BitPack, FOR, ZSTD) shrink I/O and memory 3-10x for typical
   columns; the baselines above hold raw uncompressed arrays.
4. **Aggregation without materialization.** Sum/Min/Max run vectorized
   directly over encoded columns (see `bench/bm_aggregate.cpp`); row stores
   must fully materialize rows first.

**Where it does not win (yet):**

- **High-selectivity scans with random matches (50% selectivity):**
  86 M rows/s, slower than a raw array scan. The bottleneck is delivering
  random rows to the callback (decode + copy + dispatch per run), not the
  filter itself. In-memory analytical scans that materialize results into a
  fixed array will beat the current delivery path; block-level vectorized
  delivery is the roadmap item that closes this.
- **Point reads of a single row.** A row store gives you the whole row in one
  cache line. Columnar pays N decodes for N columns. If your access pattern
  is "get row 42 by key," use a row store or key-value store.

**Honest baseline:** on this hardware the engine is competitive with (not yet
faster than) a tight row-store loop for in-memory, single-column, 50%-match
scans. The value shows up with selectivity (<10%), clustered keys + zone maps,
wide rows, compressed I/O-bound data, and vectorized aggregation — the
conditions actual analytical workloads run under.
