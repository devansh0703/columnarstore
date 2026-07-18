#include <gtest/gtest.h>
#include <columnar/encoding/encoding.h>
#include <columnar/types.h>
#include <vector>
#include <random>

using namespace columnar;
using namespace columnar::encoding;

TEST(EncodingTest, RoundTripInt32) {
    std::vector<int32_t> data(1000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(-10000, 10000);
    for (auto& v : data) v = dist(gen);
    
    Buffer buffer;
    auto encoder = CreateEncoder(EncodingType::Plain, DataType::Int32);
    encoder->Encode(data.data(), data.size(), buffer);
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateDecoder(EncodingType::Plain, DataType::Int32);
    decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
    
    EXPECT_EQ(data, decoded);
}

TEST(EncodingTest, RoundTripInt64) {
    std::vector<int64_t> data(1000);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int64_t> dist(-10000000000LL, 10000000000LL);
    for (auto& v : data) v = dist(gen);
    
    Buffer buffer;
    auto encoder = CreateEncoder(EncodingType::Plain, DataType::Int64);
    encoder->Encode(data.data(), data.size(), buffer);
    
    std::vector<int64_t> decoded(data.size());
    auto decoder = CreateDecoder(EncodingType::Plain, DataType::Int64);
    decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
    
    EXPECT_EQ(data, decoded);
}

TEST(EncodingTest, RoundTripFloat) {
    std::vector<float> data(1000);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    for (auto& v : data) v = dist(gen);
    
    Buffer buffer;
    auto encoder = CreateEncoder(EncodingType::Plain, DataType::Float);
    encoder->Encode(data.data(), data.size(), buffer);
    
    std::vector<float> decoded(data.size());
    auto decoder = CreateDecoder(EncodingType::Plain, DataType::Float);
    decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
    
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], decoded[i]);
    }
}

TEST(EncodingTest, RoundTripDouble) {
    std::vector<double> data(1000);
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    for (auto& v : data) v = dist(gen);
    
    Buffer buffer;
    auto encoder = CreateEncoder(EncodingType::Plain, DataType::Double);
    encoder->Encode(data.data(), data.size(), buffer);
    
    std::vector<double> decoded(data.size());
    auto decoder = CreateDecoder(EncodingType::Plain, DataType::Double);
    decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
    
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_DOUBLE_EQ(data[i], decoded[i]);
    }
}

TEST(EncodingTest, RLECompression) {
    std::vector<int32_t> data(1000, 42);
    
    Buffer buffer;
    auto encoder = CreateEncoder(EncodingType::RLE, DataType::Int32);
    encoder->Encode(data.data(), data.size(), buffer);
    
    EXPECT_LT(buffer.Size(), data.size() * sizeof(int32_t));
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateDecoder(EncodingType::RLE, DataType::Int32);
    decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
    
    EXPECT_EQ(data, decoded);
}

TEST(EncodingTest, DeltaEncoding) {
    std::vector<int32_t> data(1000);
    for (size_t i = 0; i < 1000; ++i) data[i] = static_cast<int32_t>(i * 10);
    
    Buffer buffer;
    auto encoder = CreateEncoder(EncodingType::Delta, DataType::Int32);
    encoder->Encode(data.data(), data.size(), buffer);
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateDecoder(EncodingType::Delta, DataType::Int32);
    decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
    
    EXPECT_EQ(data, decoded);
}

TEST(EncodingTest, FOREncoding) {
    std::vector<int32_t> data(1000);
    for (size_t i = 0; i < 1000; ++i) data[i] = 10000 + static_cast<int32_t>(i);
    
    Buffer buffer;
    auto encoder = CreateEncoder(EncodingType::FOR, DataType::Int32);
    encoder->Encode(data.data(), data.size(), buffer);
    
    EXPECT_LT(buffer.Size(), data.size() * sizeof(int32_t));
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateDecoder(EncodingType::FOR, DataType::Int32);
    decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
    
    EXPECT_EQ(data, decoded);
}

TEST(EncodingTest, DictionaryEncoding) {
    std::vector<int32_t> data(1000);
    std::vector<int32_t> values = {10, 20, 30, 40, 50};
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(0, 4);
    for (auto& v : data) v = values[dist(gen)];
    
    Buffer buffer;
    auto encoder = CreateEncoder(EncodingType::Dictionary, DataType::Int32);
    encoder->Encode(data.data(), data.size(), buffer);
    
    EXPECT_LT(buffer.Size(), data.size() * sizeof(int32_t));
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateDecoder(EncodingType::Dictionary, DataType::Int32);
    decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
    
    EXPECT_EQ(data, decoded);
}

TEST(EncodingTest, BitPackEncoding) {
    std::vector<int32_t> data(1000);
    for (size_t i = 0; i < 1000; ++i) data[i] = static_cast<int32_t>(i % 16);
    
    Buffer buffer;
    auto encoder = CreateEncoder(EncodingType::BitPack, DataType::Int32);
    encoder->Encode(data.data(), data.size(), buffer);
    
    EXPECT_LT(buffer.Size(), data.size() * sizeof(int32_t));
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateDecoder(EncodingType::BitPack, DataType::Int32);
    decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size());
    
    EXPECT_EQ(data, decoded);
}

TEST(EncodingTest, ZSTDCompression) {
    std::vector<int32_t> data(10000);
    for (size_t i = 0; i < 10000; ++i) data[i] = static_cast<int32_t>(i % 100);
    
    Buffer buffer;
    auto encoder = CreateZstdEncoder(3);
    encoder->Encode(data.data(), data.size() * sizeof(int32_t), buffer);
    
    EXPECT_LT(buffer.Size(), data.size() * sizeof(int32_t));
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateZstdDecoder();
    decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size() * sizeof(int32_t));
    
    EXPECT_EQ(data, decoded);
}

TEST(EncodingTest, LZ4Compression) {
    std::vector<int32_t> data(10000);
    for (size_t i = 0; i < 10000; ++i) data[i] = static_cast<int32_t>(i % 100);
    
    Buffer buffer;
    auto encoder = CreateLz4Encoder(9);
    encoder->Encode(data.data(), data.size() * sizeof(int32_t), buffer);
    
    EXPECT_LT(buffer.Size(), data.size() * sizeof(int32_t));
    
    std::vector<int32_t> decoded(data.size());
    auto decoder = CreateLz4Decoder();
    decoder->Decode(buffer.Data(), buffer.Size(), decoded.data(), data.size() * sizeof(int32_t));
    
    EXPECT_EQ(data, decoded);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
