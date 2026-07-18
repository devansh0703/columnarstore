#include <benchmark/benchmark.h>
#include <columnar/bloom_filter.h>
#include <columnar/zone_map.h>
#include <vector>
#include <random>

using namespace columnar;
using namespace columnar::bloom_filter;
using namespace columnar::zone_map;

static void BM_BloomFilterAdd(benchmark::State& state) {
    BloomFilter bf(1 << 20, 7);
    std::vector<int64_t> data(10000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int64_t> dist(0, 1000000);
    for (auto& v : data) v = dist(gen);
    
    size_t idx = 0;
    for (auto _ : state) {
        bf.Add(data[idx]);
        idx = (idx + 1) % data.size();
        benchmark::DoNotOptimize(bf.BitsSet());
    }
}

static void BM_BloomFilterMightContain(benchmark::State& state) {
    BloomFilter bf(1 << 20, 7);
    std::vector<int64_t> data(10000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int64_t> dist(0, 1000000);
    for (auto& v : data) v = dist(gen);
    for (auto v : data) bf.Add(v);
    
    size_t idx = 0;
    for (auto _ : state) {
        bool result = bf.MightContain(data[idx]);
        idx = (idx + 1) % data.size();
        benchmark::DoNotOptimize(result);
    }
}

static void BM_BloomFilterBatch(benchmark::State& state) {
    BloomFilter bf(1 << 20, 7);
    std::vector<int64_t> data(10000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int64_t> dist(0, 1000000);
    for (auto& v : data) v = dist(gen);
    for (auto v : data) bf.Add(v);
    
    bool results[1000];
    
    for (auto _ : state) {
        bf.MightContainBatch(data.data(), 1000, results);
        benchmark::DoNotOptimize(results[0]);
    }
    
    state.SetItemsProcessed(state.iterations() * 1000);
}

static void BM_BloomFilterAvx512Batch(benchmark::State& state) {
    BloomFilter bf(1 << 20, 7);
    std::vector<int64_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int64_t> dist(0, 1000000);
    for (auto& v : data) v = dist(gen);
    for (auto v : data) bf.Add(v);
    
    bool results[8192];
    
    for (auto _ : state) {
        bf.MightContainBatchAvx512(data.data(), 8192, results);
        benchmark::DoNotOptimize(results[0]);
    }
    
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_ZoneMapBuild(benchmark::State& state) {
    std::vector<int32_t> data(65536);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000000);
    for (auto& v : data) v = dist(gen);
    
    ZoneMap zm(65536);
    
    for (auto _ : state) {
        zm.BuildFromColumn<DataType::Int32>(data.data(), nullptr, 65536);
        benchmark::DoNotOptimize(zm.NumBlocks());
    }
}

static void BM_ZoneMapPruning(benchmark::State& state) {
    std::vector<int32_t> data(65536);
    for (size_t i = 0; i < 65536; ++i) data[i] = static_cast<int32_t>(i);
    
    ZoneMap zm(65536);
    zm.BuildFromColumn<DataType::Int32>(data.data(), nullptr, 65536);
    
    int32_t value = 50000;
    
    for (auto _ : state) {
        bool can_scan = zm.ShouldScanBlock<PredicateType::Equal, DataType::Int32>(0, value);
        benchmark::DoNotOptimize(can_scan);
    }
}

static void BM_ZoneMapRangePruning(benchmark::State& state) {
    std::vector<int32_t> data(65536);
    for (size_t i = 0; i < 65536; ++i) data[i] = static_cast<int32_t>(i);
    
    ZoneMap zm(65536);
    zm.BuildFromColumn<DataType::Int32>(data.data(), nullptr, 65536);
    
    int32_t value = 32768;
    
    for (auto _ : state) {
        bool can_scan = zm.ShouldScanBlock<PredicateType::LessThan, DataType::Int32>(0, value);
        benchmark::DoNotOptimize(can_scan);
    }
}

static void BM_BlockedBloomFilter(benchmark::State& state) {
    BlockedBloomFilter bbf(1 << 20, 8192, 7);
    std::vector<int64_t> data(10000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int64_t> dist(0, 1000000);
    for (auto& v : data) v = dist(gen);
    
    size_t block = 0;
    for (auto v : data) bbf.Add(v, block++ % bbf.NumBlocks());
    
    bool results[1000];
    
    for (auto _ : state) {
        bbf.MightContainBatch(data.data(), 1000, block % bbf.NumBlocks(), results);
        block++;
        benchmark::DoNotOptimize(results[0]);
    }
    
    state.SetItemsProcessed(state.iterations() * 1000);
}

BENCHMARK(BM_BloomFilterAdd);
BENCHMARK(BM_BloomFilterMightContain);
BENCHMARK(BM_BloomFilterBatch);
BENCHMARK(BM_BloomFilterAvx512Batch);
BENCHMARK(BM_ZoneMapBuild);
BENCHMARK(BM_ZoneMapPruning);
BENCHMARK(BM_ZoneMapRangePruning);
BENCHMARK(BM_BlockedBloomFilter);

BENCHMARK_MAIN();
