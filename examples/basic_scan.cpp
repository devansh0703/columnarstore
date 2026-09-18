// basic_scan.cpp — end-to-end tour of ColumnarStore.
//
// 1. Write a segment with three differently-encoded columns
// 2. Reopen it from disk and read a column back
// 3. Build a compiled predicate and scan with zone-map pruning + bloom filters
//
// Build (from repo root):
//   cmake -B build -DCMAKE_BUILD_TYPE=Release -DCOLUMNARSTORE_BUILD_EXAMPLES=ON
//   cmake --build build --target basic_scan
//   ./build/examples/basic_scan

#include <columnar/segment.h>
#include <columnar/scanner.h>
#include <columnar/predicate.h>
#include <columnar/types.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <vector>

using namespace columnar;
using namespace columnar::scanner;
using namespace columnar::predicate;

namespace {

// Build a segment with 100k rows:
//   col 0: id    (monotonic, Delta encoded)
//   col 1: value (repeating runs, RLE encoded)
//   col 2: score (random-ish, Plain encoded)
std::shared_ptr<Segment> WriteSampleSegment(const char* path) {
    constexpr size_t kRows = 100000;

    std::vector<int32_t> ids(kRows);
    std::vector<int32_t> values(kRows);
    std::vector<int32_t> scores(kRows);
    for (size_t i = 0; i < kRows; ++i) {
        ids[i] = static_cast<int32_t>(i);
        values[i] = static_cast<int32_t>(i % 64);
        scores[i] = static_cast<int32_t>((i * 2654435761u) % 1000003);
    }

    auto writer = SegmentWriter(path);
    writer.AddColumn<DataType::Int32>(ids, std::vector<bool>(kRows, false), EncodingType::Delta);
    writer.AddColumn<DataType::Int32>(values, std::vector<bool>(kRows, false), EncodingType::RLE);
    writer.AddColumn<DataType::Int32>(scores, std::vector<bool>(kRows, false), EncodingType::Plain);

    // Zone maps + bloom filters enable block pruning at scan time.
    zone_map::ZoneMap zm(kScanBlockSize);
    zm.BuildFromColumn<DataType::Int32>(ids.data(), nullptr, kRows);
    writer.AddZoneMap(0, zm);

    bloom_filter::BloomFilter bf(1 << 16, 7);
    for (auto v : ids) bf.Add(v);
    writer.AddBloomFilter(0, bf);

    return writer.Finish();
}

}  // namespace

int main() {
    const char* path = "/tmp/columnarstore_example.col";

    auto t0 = std::chrono::steady_clock::now();
    auto segment = WriteSampleSegment(path);
    auto t1 = std::chrono::steady_clock::now();
    std::printf("wrote %u rows in %.1f ms\n",
                static_cast<unsigned>(segment->NumRows()),
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Reopen from disk.
    auto opened = Segment::Open(path);
    if (!opened) {
        std::fprintf(stderr, "failed to open %s\n", path);
        return 1;
    }
    std::printf("opened segment: %u rows, %u columns\n",
                static_cast<unsigned>(opened->NumRows()),
                static_cast<unsigned>(opened->NumColumns()));

    // Predicate: id < 50000 AND value == 7
    auto pred = PredicateBuilder()
                    .Lt<DataType::Int32>(0, 50000)
                    .And()
                    .Eq<DataType::Int32>(1, 7)
                    .Build();

    ScanContext ctx;
    ctx.projected_columns = {0};
    ctx.enable_zone_map_pruning = true;
    ctx.enable_bloom_filter = true;

    SegmentScanner scanner(opened, ctx);

    size_t matched = 0;
    auto t2 = std::chrono::steady_clock::now();
    scanner.Scan(*pred, [&](const void** /*cols*/, size_t /*count*/) { ++matched; });
    auto t3 = std::chrono::steady_clock::now();

    const auto& stats = scanner.Stats();
    std::printf("scan matched %zu rows in %.2f ms "
                "(blocks pruned: zone_map=%zu bloom=%zu)\n",
                matched, std::chrono::duration<double, std::milli>(t3 - t2).count(),
                stats.blocks_pruned_zone_map, stats.blocks_pruned_bloom);

    std::filesystem::remove(path);
    return 0;
}
