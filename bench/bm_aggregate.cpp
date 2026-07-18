#include <benchmark/benchmark.h>
#include <columnar/column/column.h>
#include <simd/avx512.h>
#include <simd/avx2.h>
#include <simd/dispatch.h>
#include <vector>
#include <random>
#include <immintrin.h>

using namespace columnar;

static void BM_VectorSumInt32(benchmark::State& state) {
    std::vector<int32_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);
    for (auto& v : data) v = dist(gen);
    
    int32_t result = 0;
    for (auto _ : state) {
        result = column::VectorOps<DataType::Int32>::Sum(data.data(), data.size());
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_VectorSumInt64(benchmark::State& state) {
    std::vector<int64_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int64_t> dist(-1000000, 1000000);
    for (auto& v : data) v = dist(gen);
    
    int64_t result = 0;
    for (auto _ : state) {
        result = column::VectorOps<DataType::Int64>::Sum(data.data(), data.size());
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_VectorSumFloat(benchmark::State& state) {
    std::vector<float> data(8192);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (auto& v : data) v = dist(gen);
    
    float result = 0;
    for (auto _ : state) {
        result = column::VectorOps<DataType::Float>::Sum(data.data(), data.size());
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_VectorSumDouble(benchmark::State& state) {
    std::vector<double> data(8192);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (auto& v : data) v = dist(gen);
    
    double result = 0;
    for (auto _ : state) {
        result = column::VectorOps<DataType::Double>::Sum(data.data(), data.size());
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_VectorMinMaxInt32(benchmark::State& state) {
    std::vector<int32_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);
    for (auto& v : data) v = dist(gen);
    
    int32_t min_val = 0, max_val = 0;
    for (auto _ : state) {
        min_val = column::VectorOps<DataType::Int32>::Min(data.data(), data.size());
        max_val = column::VectorOps<DataType::Int32>::Max(data.data(), data.size());
        benchmark::DoNotOptimize(min_val);
        benchmark::DoNotOptimize(max_val);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_VectorMinMaxFloat(benchmark::State& state) {
    std::vector<float> data(8192);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (auto& v : data) v = dist(gen);
    
    float min_val = 0, max_val = 0;
    for (auto _ : state) {
        min_val = column::VectorOps<DataType::Float>::Min(data.data(), data.size());
        max_val = column::VectorOps<DataType::Float>::Max(data.data(), data.size());
        benchmark::DoNotOptimize(min_val);
        benchmark::DoNotOptimize(max_val);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_VectorAddInt32(benchmark::State& state) {
    std::vector<int32_t> a(8192), b(8192), out(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(-1000, 1000);
    for (auto& v : a) v = dist(gen);
    for (auto& v : b) v = dist(gen);
    
    for (auto _ : state) {
        column::VectorOps<DataType::Int32>::Add(a.data(), b.data(), out.data(), 8192);
        benchmark::DoNotOptimize(out[0]);
    }
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_VectorMulFloat(benchmark::State& state) {
    std::vector<float> a(8192), b(8192), out(8192);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& v : a) v = dist(gen);
    for (auto& v : b) v = dist(gen);
    
    for (auto _ : state) {
        column::VectorOps<DataType::Float>::Mul(a.data(), b.data(), out.data(), 8192);
        benchmark::DoNotOptimize(out[0]);
    }
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_Avx512PrefixSum(benchmark::State& state) {
    std::vector<int32_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(1, 10);
    for (auto& v : data) v = dist(gen);
    
    std::vector<int32_t> out(8192);
    
    for (auto _ : state) {
        for (size_t i = 0; i < 8192; i += 16) {
            __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data.data() + i));
            v = simd::avx512::PrefixSum(v);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(out.data() + i), v);
        }
        benchmark::DoNotOptimize(out[0]);
    }
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_Avx512Compress(benchmark::State& state) {
    std::vector<int32_t> data(8192);
    std::vector<int32_t> out(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 100);
    for (auto& v : data) v = dist(gen);
    
    __mmask16 mask = 0x5555;
    
    for (auto _ : state) {
        for (size_t i = 0; i < 8192; i += 16) {
            __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data.data() + i));
            __m512i result = simd::avx512::Compress(v, mask);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(out.data() + i), result);
        }
        benchmark::DoNotOptimize(out[0]);
    }
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_Avx512ConflictDetect(benchmark::State& state) {
    std::vector<int32_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000);
    for (auto& v : data) v = dist(gen);
    
    std::vector<int32_t> out(8192);
    
    for (auto _ : state) {
        for (size_t i = 0; i < 8192; i += 16) {
            __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data.data() + i));
            __m512i result = simd::avx512::ConflictDetect(v);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(out.data() + i), result);
        }
        benchmark::DoNotOptimize(out[0]);
    }
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_Avx2PrefixSum(benchmark::State& state) {
    std::vector<int32_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(1, 10);
    for (auto& v : data) v = dist(gen);
    
    std::vector<int32_t> out(8192);
    
    for (auto _ : state) {
        for (size_t i = 0; i < 8192; i += 8) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data.data() + i));
            v = simd::avx2::PrefixSum(v);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out.data() + i), v);
        }
        benchmark::DoNotOptimize(out[0]);
    }
    state.SetItemsProcessed(state.iterations() * 8192);
}

BENCHMARK(BM_VectorSumInt32);
BENCHMARK(BM_VectorSumInt64);
BENCHMARK(BM_VectorSumFloat);
BENCHMARK(BM_VectorSumDouble);
BENCHMARK(BM_VectorMinMaxInt32);
BENCHMARK(BM_VectorMinMaxFloat);
BENCHMARK(BM_VectorAddInt32);
BENCHMARK(BM_VectorMulFloat);
BENCHMARK(BM_Avx512PrefixSum);
BENCHMARK(BM_Avx512Compress);
BENCHMARK(BM_Avx512ConflictDetect);
BENCHMARK(BM_Avx2PrefixSum);

BENCHMARK_MAIN();
