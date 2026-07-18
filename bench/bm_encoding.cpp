#include <benchmark/benchmark.h>
#include <columnar/encoding/encoding.h>
#include <columnar/types.h>
#include <vector>
#include <random>

using namespace columnar;
using namespace columnar::encoding;

static void BM_PlainEncodeInt32(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(-1000000, 1000000);
    for (auto& v : data) v = dist(gen);
    
    Buffer buffer;
    auto encoder = CreateEncoder<EncodingType::Plain, DataType::Int32>();
    
    for (auto _ : state) {
        buffer.Clear();
        encoder->Encode(data.data(), data.size(), buffer);
        benchmark::DoNotOptimize(buffer.Size());
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_PlainDecodeInt32(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(-1000000, 1000000);
    for (auto& v : data) v = dist(gen);
    
    Buffer buffer;
    auto encoder = CreateEncoder<EncodingType::Plain, DataType::Int32>();
    encoder->Encode(data.data(), data.size(), buffer);
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateDecoder<EncodingType::Plain, DataType::Int32>();
    
    for (auto _ : state) {
        decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
        benchmark::DoNotOptimize(decoded[0]);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_RLEEncodeInt32(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 10);
    for (auto& v : data) v = dist(gen);
    
    Buffer buffer;
    auto encoder = CreateEncoder<EncodingType::RLE, DataType::Int32>();
    
    for (auto _ : state) {
        buffer.Clear();
        encoder->Encode(data.data(), data.size(), buffer);
        benchmark::DoNotOptimize(buffer.Size());
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_RLEDecodeInt32(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 10);
    for (auto& v : data) v = dist(gen);
    
    Buffer buffer;
    auto encoder = CreateEncoder<EncodingType::RLE, DataType::Int32>();
    encoder->Encode(data.data(), data.size(), buffer);
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateDecoder<EncodingType::RLE, DataType::Int32>();
    
    for (auto _ : state) {
        decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
        benchmark::DoNotOptimize(decoded[0]);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_FOREncodeInt32(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(1000000, 1000100);
    for (auto& v : data) v = dist(gen);
    
    Buffer buffer;
    auto encoder = CreateEncoder<EncodingType::FOR, DataType::Int32>();
    
    for (auto _ : state) {
        buffer.Clear();
        encoder->Encode(data.data(), data.size(), buffer);
        benchmark::DoNotOptimize(buffer.Size());
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_FORDecodeInt32(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(1000000, 1000100);
    for (auto& v : data) v = dist(gen);
    
    Buffer buffer;
    auto encoder = CreateEncoder<EncodingType::FOR, DataType::Int32>();
    encoder->Encode(data.data(), data.size(), buffer);
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateDecoder<EncodingType::FOR, DataType::Int32>();
    
    for (auto _ : state) {
        decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
        benchmark::DoNotOptimize(decoded[0]);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_DictionaryEncodeInt32(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    std::vector<int32_t> values = {10, 20, 30, 40, 50};
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(0, 4);
    for (auto& v : data) v = values[dist(gen)];
    
    Buffer buffer;
    auto encoder = CreateEncoder<EncodingType::Dictionary, DataType::Int32>();
    
    for (auto _ : state) {
        buffer.Clear();
        encoder->Encode(data.data(), data.size(), buffer);
        benchmark::DoNotOptimize(buffer.Size());
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_DictionaryDecodeInt32(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    std::vector<int32_t> values = {10, 20, 30, 40, 50};
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(0, 4);
    for (auto& v : data) v = values[dist(gen)];
    
    Buffer buffer;
    auto encoder = CreateEncoder<EncodingType::Dictionary, DataType::Int32>();
    encoder->Encode(data.data(), data.size(), buffer);
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateDecoder<EncodingType::Dictionary, DataType::Int32>();
    
    for (auto _ : state) {
        decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
        benchmark::DoNotOptimize(decoded[0]);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_BitPackEncodeInt32(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    for (size_t i = 0; i < 10000; ++i) data[i] = static_cast<int32_t>(i % 16);
    
    Buffer buffer;
    auto encoder = CreateEncoder<EncodingType::BitPack, DataType::Int32>();
    
    for (auto _ : state) {
        buffer.Clear();
        encoder->Encode(data.data(), data.size(), buffer);
        benchmark::DoNotOptimize(buffer.Size());
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_BitPackDecodeInt32(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    for (size_t i = 0; i < 10000; ++i) data[i] = static_cast<int32_t>(i % 16);
    
    Buffer buffer;
    auto encoder = CreateEncoder<EncodingType::BitPack, DataType::Int32>();
    encoder->Encode(data.data(), data.size(), buffer);
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateDecoder<EncodingType::BitPack, DataType::Int32>();
    
    for (auto _ : state) {
        decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
        benchmark::DoNotOptimize(decoded[0]);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_ZSTDEncode(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    for (size_t i = 0; i < 10000; ++i) data[i] = static_cast<int32_t>(i % 100);
    
    Buffer buffer;
    auto encoder = CreateZstdEncoder(3);
    
    for (auto _ : state) {
        buffer.Clear();
        encoder->Encode(data.data(), data.size() * sizeof(int32_t), buffer);
        benchmark::DoNotOptimize(buffer.Size());
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_ZSTDDecode(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    for (size_t i = 0; i < 10000; ++i) data[i] = static_cast<int32_t>(i % 100);
    
    Buffer buffer;
    auto encoder = CreateZstdEncoder(3);
    encoder->Encode(data.data(), data.size() * sizeof(int32_t), buffer);
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateZstdDecoder();
    
    for (auto _ : state) {
        decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size() * sizeof(int32_t));
        benchmark::DoNotOptimize(decoded[0]);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_LZ4Encode(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    for (size_t i = 0; i < 10000; ++i) data[i] = static_cast<int32_t>(i % 100);
    
    Buffer buffer;
    auto encoder = CreateLz4Encoder(9);
    
    for (auto _ : state) {
        buffer.Clear();
        encoder->Encode(data.data(), data.size() * sizeof(int32_t), buffer);
        benchmark::DoNotOptimize(buffer.Size());
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

static void BM_LZ4Decode(benchmark::State& state) {
    std::vector<int32_t> data(10000);
    for (size_t i = 0; i < 10000; ++i) data[i] = static_cast<int32_t>(i % 100);
    
    Buffer buffer;
    auto encoder = CreateLz4Encoder(9);
    encoder->Encode(data.data(), data.size() * sizeof(int32_t), buffer);
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateLz4Decoder();
    
    for (auto _ : state) {
        decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size() * sizeof(int32_t));
        benchmark::DoNotOptimize(decoded[0]);
    }
    state.SetItemsProcessed(state.iterations() * data.size());
}

BENCHMARK(BM_PlainEncodeInt32);
BENCHMARK(BM_PlainDecodeInt32);
BENCHMARK(BM_RLEEncodeInt32);
BENCHMARK(BM_RLEDecodeInt32);
BENCHMARK(BM_FOREncodeInt32);
BENCHMARK(BM_FORDecodeInt32);
BENCHMARK(BM_DictionaryEncodeInt32);
BENCHMARK(BM_DictionaryDecodeInt32);
BENCHMARK(BM_BitPackEncodeInt32);
BENCHMARK(BM_BitPackDecodeInt32);
BENCHMARK(BM_ZSTDEncode);
BENCHMARK(BM_ZSTDDecode);
BENCHMARK(BM_LZ4Encode);
BENCHMARK(BM_LZ4Decode);

BENCHMARK_MAIN();
