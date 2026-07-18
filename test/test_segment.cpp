#include <gtest/gtest.h>
#include <columnar/segment.h>
#include <columnar/types.h>
#include <vector>
#include <filesystem>
#include <random>

using namespace columnar;

TEST(SegmentTest, CreateAndOpen) {
    std::string path = "/tmp/test_segment.col";
    
    ColumnStats int32_stat{DataType::Int32, 0, 100, 100, true};
    ColumnStats int64_stat{DataType::Int64, 0, 100, 100, true};
    
    auto writer = SegmentWriter(path);
    writer.AddColumn<DataType::Int32>(
        std::vector<int32_t>(100, 0), std::vector<bool>(100, false), EncodingType::Plain);
    writer.AddColumn<DataType::Int64>(
        std::vector<int64_t>(100, 0), std::vector<bool>(100, false), EncodingType::Plain);
    
    auto segment = writer.Finish();
    EXPECT_NE(segment, nullptr);
    EXPECT_EQ(segment->NumRows(), 100);
    EXPECT_EQ(segment->NumColumns(), 2);
    
    auto opened = Segment::Open(path);
    EXPECT_NE(opened, nullptr);
    EXPECT_EQ(opened->NumRows(), 100);
    EXPECT_EQ(opened->NumColumns(), 2);
    
    std::filesystem::remove(path);
}

TEST(SegmentTest, MultipleEncodings) {
    std::string path = "/tmp/test_encoding.col";
    
    std::vector<int32_t> plain_data(1000);
    std::vector<int32_t> rle_data(1000, 42);
    std::vector<int32_t> delta_data(1000);
    for (size_t i = 0; i < 1000; ++i) delta_data[i] = static_cast<int32_t>(i * 10);
    
    auto writer = SegmentWriter(path);
    writer.AddColumn<DataType::Int32>(plain_data, std::vector<bool>(1000, false), EncodingType::Plain);
    writer.AddColumn<DataType::Int32>(rle_data, std::vector<bool>(1000, false), EncodingType::RLE);
    writer.AddColumn<DataType::Int32>(delta_data, std::vector<bool>(1000, false), EncodingType::Delta);
    
    auto segment = writer.Finish();
    EXPECT_EQ(segment->NumColumns(), 3);
    EXPECT_EQ(segment->ColumnEncoding(0), EncodingType::Plain);
    EXPECT_EQ(segment->ColumnEncoding(1), EncodingType::RLE);
    EXPECT_EQ(segment->ColumnEncoding(2), EncodingType::Delta);
    
    std::filesystem::remove(path);
}

TEST(SegmentTest, ZoneMaps) {
    std::string path = "/tmp/test_zm.col";
    
    std::vector<int32_t> data(10000);
    for (size_t i = 0; i < 10000; ++i) data[i] = static_cast<int32_t>(i);
    
    auto writer = SegmentWriter(path);
    writer.AddColumn<DataType::Int32>(data, std::vector<bool>(10000, false), EncodingType::Plain);
    
    zone_map::ZoneMap zm(1000);
    zm.BuildFromColumn<DataType::Int32>(data.data(), nullptr, 10000);
    writer.AddZoneMap(0, zm);
    
    auto segment = writer.Finish();
    auto loaded_zm = segment->GetZoneMap(0);
    EXPECT_EQ(loaded_zm.NumBlocks(), 10);
    
    std::filesystem::remove(path);
}

TEST(SegmentTest, BloomFilters) {
    std::string path = "/tmp/test_bf.col";
    
    std::vector<int32_t> data(1000);
    for (size_t i = 0; i < 1000; ++i) data[i] = static_cast<int32_t>(i * 7);
    
    auto writer = SegmentWriter(path);
    writer.AddColumn<DataType::Int32>(data, std::vector<bool>(1000, false), EncodingType::Plain);
    
    bloom_filter::BloomFilter bf(1 << 16, 7);
    for (auto v : data) bf.Add(v);
    writer.AddBloomFilter(0, bf);
    
    auto segment = writer.Finish();
    auto loaded_bf = segment->GetBloomFilter(0);
    
    for (auto v : data) {
        EXPECT_TRUE(loaded_bf.MightContain(v));
    }
    
    int false_positives = 0;
    for (int i = 0; i < 10000; ++i) {
        if (i % 7 != 0 && loaded_bf.MightContain(i)) ++false_positives;
    }
    EXPECT_LT(false_positives, 100);
    
    std::filesystem::remove(path);
}

TEST(SegmentTest, ColumnCache) {
    std::string path = "/tmp/test_cache.col";
    
    std::vector<int32_t> data(10000);
    for (size_t i = 0; i < 10000; ++i) data[i] = static_cast<int32_t>(i);
    
    auto writer = SegmentWriter(path);
    writer.AddColumn<DataType::Int32>(data, std::vector<bool>(10000, false), EncodingType::Plain);
    auto segment = writer.Finish();
    
    auto col1 = segment->GetColumn(0);
    auto col2 = segment->GetColumn(0);
    EXPECT_EQ(col1.get(), col2.get());
    
    std::filesystem::remove(path);
}

TEST(SegmentTest, PrefetchColumns) {
    std::string path = "/tmp/test_prefetch.col";
    
    std::vector<int32_t> data(10000);
    for (size_t i = 0; i < 10000; ++i) data[i] = static_cast<int32_t>(i);
    
    auto writer = SegmentWriter(path);
    writer.AddColumn<DataType::Int32>(data, std::vector<bool>(10000, false), EncodingType::Plain);
    writer.AddColumn<DataType::Int32>(data, std::vector<bool>(10000, false), EncodingType::Plain);
    auto segment = writer.Finish();
    
    segment->PrefetchColumns({0, 1});
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    std::filesystem::remove(path);
}

TEST(SegmentTest, Statistics) {
    std::string path = "/tmp/test_stats.col";
    
    std::vector<int32_t> data(100);
    for (size_t i = 0; i < 100; ++i) data[i] = static_cast<int32_t>(i);
    
    auto writer = SegmentWriter(path);
    writer.AddColumn<DataType::Int32>(data, std::vector<bool>(100, false), EncodingType::Plain);
    auto segment = writer.Finish();
    
    auto stats = segment->GetColumnStats(0);
    EXPECT_EQ(stats.total_count, 100);
    EXPECT_EQ(stats.null_count, 0);
    EXPECT_EQ(stats.distinct_count, 100);
    EXPECT_EQ(stats.min_int, 0);
    EXPECT_EQ(stats.max_int, 99);
    EXPECT_TRUE(stats.has_min_max);
    
    std::filesystem::remove(path);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
