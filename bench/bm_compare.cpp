// bm_compare.cpp — ColumnarStore vs. baselines on identical data.
//
// Answers: "when is this faster than the alternatives?"
//
// Baselines:
//   1. RowStoreScan      — struct-of-row array scanned with std::copy_if
//                          (the naive OLAP baseline most apps start with)
//   2. ScalarColumnScan  — column layout, scalar branchy filter
//   3. SimdColumnScan    — column layout, vectorized filter, no storage engine
//                          (the upper bound for a hand-rolled in-memory scan)
//   4. MemcpyCeiling     — raw memory bandwidth ceiling for moving 4 bytes/row
//
// ColumnarStore adds zone-map pruning + compressed/encoded storage on top of 3.

#include <benchmark/benchmark.h>
#include <columnar/scanner.h>
#include <columnar/predicate.h>
#include <columnar/segment.h>
#include <columnar/types.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace columnar;
using namespace columnar::scanner;
using namespace columnar::predicate;

namespace {

struct Row {
    int32_t id;
    int32_t value;
    int32_t score;
    int32_t pad;
};

constexpr size_t kRows = 1 << 20;  // 1,048,576 rows

std::vector<int32_t> MakeIds() {
    std::vector<int32_t> ids(kRows);
    for (size_t i = 0; i < kRows; ++i) ids[i] = static_cast<int32_t>(i);
    return ids;
}

std::vector<int32_t> MakeScores() {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 999999);
    std::vector<int32_t> scores(kRows);
    for (auto& v : scores) v = dist(gen);
    return scores;
}

std::shared_ptr<Segment> BuildSegment(const std::string& path,
                                      const std::vector<int32_t>& ids,
                                      const std::vector<int32_t>& scores) {
    auto writer = SegmentWriter(path);
    writer.AddColumn<DataType::Int32>(ids, std::vector<bool>(kRows, false), EncodingType::Plain);
    writer.AddColumn<DataType::Int32>(scores, std::vector<bool>(kRows, false), EncodingType::Plain);
    auto seg = writer.Finish();
    std::remove(path.c_str());
    return seg;
}

std::vector<Row> BuildRowStore(const std::vector<int32_t>& ids,
                               const std::vector<int32_t>& scores) {
    std::vector<Row> rows(kRows);
    for (size_t i = 0; i < kRows; ++i) {
        rows[i] = {ids[i], static_cast<int32_t>(i % 64), scores[i], 0};
    }
    return rows;
}

}  // namespace

// --- Baseline 1: naive row-store scan (std::copy_if over structs) ------------
static void RowStoreScan(benchmark::State& state) {
    auto rows = BuildRowStore(MakeIds(), MakeScores());
    constexpr int32_t kThreshold = 500000;
    size_t matched = 0;

    for (auto _ : state) {
        auto end = std::copy_if(rows.begin(), rows.end(), rows.begin(),
                                [](const Row& r) { return r.score < kThreshold; });
        benchmark::DoNotOptimize(end);
    }
    state.SetItemsProcessed(state.iterations() * kRows);
}

// --- Baseline 2: column layout, scalar branchy filter ------------------------
static void ScalarColumnScan(benchmark::State& state) {
    auto scores = MakeScores();
    constexpr int32_t kThreshold = 500000;
    std::vector<int32_t> out(kRows);

    for (auto _ : state) {
        auto end = std::copy_if(scores.begin(), scores.end(), out.begin(),
                                [](int32_t v) { return v < kThreshold; });
        benchmark::DoNotOptimize(end);
    }
    state.SetItemsProcessed(state.iterations() * kRows);
}

// --- Baseline 3: column layout, SIMD filter (no engine around it) ------------
static void SimdColumnScan(benchmark::State& state) {
    auto scores = MakeScores();
    constexpr int32_t kThreshold = 500000;
    std::unique_ptr<bool[]> mask(new bool[kRows]);
    size_t matched = 0;

    for (auto _ : state) {
        VectorFilter<DataType::Int32>::LessThan(scores.data(), &kThreshold,
                                                mask.get(), kRows);
        matched = 0;
        for (size_t i = 0; i < kRows; ++i) matched += mask[i];
        benchmark::DoNotOptimize(matched);
    }
    state.SetItemsProcessed(state.iterations() * kRows);
}

// --- Baseline 4: memory bandwidth ceiling ------------------------------------
static void MemcpyCeiling(benchmark::State& state) {
    auto scores = MakeScores();
    std::vector<int32_t> out(kRows);

    for (auto _ : state) {
        std::memcpy(out.data(), scores.data(), kRows * sizeof(int32_t));
        benchmark::DoNotOptimize(out[0]);
    }
    state.SetItemsProcessed(state.iterations() * kRows);
}

// --- ColumnarStore: full engine, single-column predicate (fast path) ---------
static void ColumnarStoreScan(benchmark::State& state) {
    auto seg = BuildSegment("/tmp/bm_compare.col", MakeIds(), MakeScores());
    auto pred = PredicateBuilder().Lt<DataType::Int32>(1, 500000).Build();

    ScanContext ctx;
    ctx.enable_zone_map_pruning = false;  // random data: pruning can't help
    ctx.enable_bloom_filter = false;

    size_t matched = 0;
    for (auto _ : state) {
        SegmentScanner scanner(seg, ctx);
        scanner.Scan(*pred, [&](const void** /*cols*/, size_t count) { matched += count; });
    }
    state.SetItemsProcessed(state.iterations() * kRows);
}

// --- ColumnarStore: low-selectivity scan (the typical analytics case) --------
// Only ~1% of rows match, so delivery cost is negligible and the pipeline is
// dominated by the SIMD filter over the column.
static void ColumnarStoreScanSelective(benchmark::State& state) {
    auto seg = BuildSegment("/tmp/bm_compare_sel.col", MakeIds(), MakeScores());
    auto pred = PredicateBuilder().Lt<DataType::Int32>(1, 10000).Build();

    ScanContext ctx;
    ctx.enable_zone_map_pruning = false;
    ctx.enable_bloom_filter = false;

    size_t matched = 0;
    for (auto _ : state) {
        SegmentScanner scanner(seg, ctx);
        scanner.Scan(*pred, [&](const void** /*cols*/, size_t count) { matched += count; });
    }
    state.SetItemsProcessed(state.iterations() * kRows);
}

// --- ColumnarStore: same scan WITH zone-map pruning --------------------------
// Uses a clustered id column (sorted) so pruning has something to prune:
// half the blocks are entirely below the threshold and get skipped.
static void ColumnarStoreScanPruned(benchmark::State& state) {
    auto ids = MakeIds();

    // Write the segment to disk, then attach the zone map through a writer
    // pass so it is persisted and picked up by Segment::Open.
    const std::string path = "/tmp/bm_compare_pruned.col";
    {
        auto writer = SegmentWriter(path);
        writer.AddColumn<DataType::Int32>(ids, std::vector<bool>(kRows, false), EncodingType::Plain);
        writer.AddColumn<DataType::Int32>(ids, std::vector<bool>(kRows, false), EncodingType::Plain);
        zone_map::ZoneMap zm(kScanBlockSize);
        zm.BuildFromColumn<DataType::Int32>(ids.data(), nullptr, kRows);
        writer.AddZoneMap(0, zm);
        writer.Finish();
    }
    auto seg = Segment::Open(path);
    std::remove(path.c_str());

    auto pred = PredicateBuilder().Lt<DataType::Int32>(0, kRows / 2).Build();

    ScanContext ctx;
    ctx.enable_zone_map_pruning = true;
    ctx.enable_bloom_filter = false;

    size_t matched = 0;
    for (auto _ : state) {
        SegmentScanner scanner(seg, ctx);
        scanner.Scan(*pred, [&](const void** /*cols*/, size_t count) { matched += count; });
    }
    state.SetItemsProcessed(state.iterations() * kRows);
}

BENCHMARK(RowStoreScan);
BENCHMARK(ScalarColumnScan);
BENCHMARK(SimdColumnScan);
BENCHMARK(MemcpyCeiling);
BENCHMARK(ColumnarStoreScan);
BENCHMARK(ColumnarStoreScanSelective);
BENCHMARK(ColumnarStoreScanPruned);

BENCHMARK_MAIN();
