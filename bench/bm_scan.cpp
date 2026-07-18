#include <benchmark/benchmark.h>
#include <columnar/scanner.h>
#include <columnar/predicate.h>
#include <columnar/segment.h>
#include <columnar/types.h>
#include <vector>
#include <random>

using namespace columnar;
using namespace columnar::scanner;
using namespace columnar::predicate;

static void BM_VectorFilterEqualInt32(benchmark::State& state) {
    std::vector<int32_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000);
    for (auto& v : data) v = dist(gen);
    
    bool mask[8192];
    int32_t value = 500;
    
    for (auto _ : state) {
        VectorFilter<DataType::Int32>::Equal(data.data(), &value, mask, 8192);
        benchmark::DoNotOptimize(mask[0]);
    }
    
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_VectorFilterLessThanInt32(benchmark::State& state) {
    std::vector<int32_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000);
    for (auto& v : data) v = dist(gen);
    
    bool mask[8192];
    int32_t value = 500;
    
    for (auto _ : state) {
        VectorFilter<DataType::Int32>::LessThan(data.data(), &value, mask, 8192);
        benchmark::DoNotOptimize(mask[0]);
    }
    
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_VectorFilterEqualInt64(benchmark::State& state) {
    std::vector<int64_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int64_t> dist(0, 1000000);
    for (auto& v : data) v = dist(gen);
    
    bool mask[8192];
    int64_t value = 500000;
    
    for (auto _ : state) {
        VectorFilter<DataType::Int64>::Equal(data.data(), &value, mask, 8192);
        benchmark::DoNotOptimize(mask[0]);
    }
    
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_SimdPredicateEqual(benchmark::State& state) {
    std::vector<int32_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000);
    for (auto& v : data) v = dist(gen);
    
    bool mask[8192];
    int32_t value = 500;
    
    for (auto _ : state) {
        SimdPredicateEvaluator::EvaluateVector<PredicateType::Equal, DataType::Int32>(
            data.data(), &value, mask, 8192);
        benchmark::DoNotOptimize(mask[0]);
    }
    
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_SimdPredicateLessThan(benchmark::State& state) {
    std::vector<int32_t> data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 1000);
    for (auto& v : data) v = dist(gen);
    
    bool mask[8192];
    int32_t value = 500;
    
    for (auto _ : state) {
        SimdPredicateEvaluator::EvaluateVector<PredicateType::LessThan, DataType::Int32>(
            data.data(), &value, mask, 8192);
        benchmark::DoNotOptimize(mask[0]);
    }
    
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_BlockFilterAnd(benchmark::State& state) {
    BlockFilter filter1, filter2;
    filter1.SetAll(true);
    filter2.SetAll(true);
    filter1.Data()[0] = false;
    filter2.Data()[4096] = false;
    
    BlockFilter result;
    
    for (auto _ : state) {
        result = filter1;
        result.And(filter2);
        benchmark::DoNotOptimize(result.CountTrue());
    }
}

static void BM_BlockFilterOr(benchmark::State& state) {
    BlockFilter filter1, filter2;
    filter1.SetAll(false);
    filter2.SetAll(false);
    filter1.Data()[0] = true;
    filter2.Data()[4096] = true;
    
    BlockFilter result;
    
    for (auto _ : state) {
        result = filter1;
        result.Or(filter2);
        benchmark::DoNotOptimize(result.CountTrue());
    }
}

static void BM_BytecodeInterpreter(benchmark::State& state) {
    BytecodeProgram program;
    size_t col0 = program.AddColumn(0);
    size_t const500 = program.AddConstant(int64_t(500));
    size_t const1000 = program.AddConstant(int64_t(1000));
    
    program.Emit(OpCode::LoadColumn, col0);
    program.Emit(OpCode::LoadConst, const500);
    program.Emit(OpCode::CompareGt);
    program.Emit(OpCode::LoadColumn, col0);
    program.Emit(OpCode::LoadConst, const1000);
    program.Emit(OpCode::CompareLt);
    program.Emit(OpCode::And);
    program.Emit(OpCode::Return);
    
    std::vector<int64_t> col0_data(8192);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int64_t> dist(0, 2000);
    for (auto& v : col0_data) v = dist(gen);
    
    std::vector<const void*> cols = {col0_data.data()};
    std::vector<const bool*> nulls = {nullptr};
    bool output[8192];
    
    BytecodeInterpreter interpreter(program);
    
    for (auto _ : state) {
        interpreter.Execute(cols, nulls, 8192, output);
        benchmark::DoNotOptimize(output[0]);
    }
    
    state.SetItemsProcessed(state.iterations() * 8192);
}

static void BM_PredicateBuilder(benchmark::State& state) {
    for (auto _ : state) {
        auto pred = columnar::predicate::PredicateBuilder()
            .Eq<DataType::Int32>(0, 100)
            .And()
            .Lt<DataType::Int32>(1, 500)
            .Or()
            .Null(2)
            .Build();
        benchmark::DoNotOptimize(pred.get());
    }
}

BENCHMARK(BM_VectorFilterEqualInt32);
BENCHMARK(BM_VectorFilterLessThanInt32);
BENCHMARK(BM_VectorFilterEqualInt64);
BENCHMARK(BM_SimdPredicateEqual);
BENCHMARK(BM_SimdPredicateLessThan);
BENCHMARK(BM_BlockFilterAnd);
BENCHMARK(BM_BlockFilterOr);
BENCHMARK(BM_BytecodeInterpreter);
BENCHMARK(BM_PredicateBuilder);

BENCHMARK_MAIN();
