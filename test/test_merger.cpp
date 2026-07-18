#include <gtest/gtest.h>
#include <columnar/merger.h>
#include <columnar/segment.h>
#include <columnar/delta_store.h>
#include <vector>
#include <filesystem>

using namespace columnar;
using namespace columnar::merger;
using namespace columnar::delta;

TEST(MergerTest, ColumnMergerInt32) {
    std::vector<int32_t> data1(100);
    std::vector<int32_t> data2(100);
    for (size_t i = 0; i < 100; ++i) {
        data1[i] = static_cast<int32_t>(i);
        data2[i] = static_cast<int32_t>(i + 100);
    }
    
    ColumnStats stats1{DataType::Int32, 0, 100, 100, true};
    ColumnStats stats2{DataType::Int32, 0, 100, 100, true};
    
    auto seg1 = Segment::Create("/tmp/merger_seg1.col", {stats1}, 100);
    auto seg2 = Segment::Create("/tmp/merger_seg2.col", {stats2}, 200);
    
    auto writer1 = SegmentWriter("/tmp/merger_seg1.col");
    writer1.AddColumn<DataType::Int32>(data1, std::vector<bool>(100, false), EncodingType::Plain);
    auto finished1 = writer1.Finish();
    
    auto writer2 = SegmentWriter("/tmp/merger_seg2.col");
    writer2.AddColumn<DataType::Int32>(data2, std::vector<bool>(100, false), EncodingType::Plain);
    auto finished2 = writer2.Finish();
    
    ColumnMerger merger(DataType::Int32, EncodingType::Plain);
    merger.SetColumnIndex(0);
    
    std::shared_ptr<DeltaStore> delta = nullptr;
    std::vector<uint8_t> output;
    ColumnStats out_stats;
    
    merger.MergeColumn<DataType::Int32>({finished1, finished2}, delta, output, out_stats);
    
    EXPECT_EQ(out_stats.total_count, 200);
    EXPECT_EQ(out_stats.null_count, 0);
    EXPECT_EQ(out_stats.min_int, 0);
    EXPECT_EQ(out_stats.max_int, 199);
    
    std::filesystem::remove("/tmp/merger_seg1.col");
    std::filesystem::remove("/tmp/merger_seg2.col");
}

TEST(MergerTest, SegmentMergerBasic) {
    std::vector<DataType> types = {DataType::Int32};
    std::vector<EncodingType> encodings = {EncodingType::Plain};
    
    SegmentMerger merger(1, types, encodings, 2);
    
    MergeTask task;
    task.output_path = "/tmp/merged_segment.col";
    task.min_timestamp = 0;
    task.max_timestamp = 1000;
    
    auto future = merger.SubmitMerge(std::move(task));
    auto result = future.get();
    
    EXPECT_NE(result, nullptr);
    std::filesystem::remove("/tmp/merged_segment.col");
}

TEST(MergerTest, BackgroundMerger) {
    std::vector<DataType> types = {DataType::Int32};
    std::vector<EncodingType> encodings = {EncodingType::Plain};
    
    BackgroundMerger bg_merger(1, types, encodings, 1);
    bg_merger.Start();
    
    MergeTask task;
    task.output_path = "/tmp/bg_merged.col";
    task.min_timestamp = 0;
    task.max_timestamp = 100;
    
    auto future = bg_merger.ScheduleMerge(std::move(task));
    auto result = future.get();
    
    bg_merger.Stop();
    
    EXPECT_NE(result, nullptr);
    std::filesystem::remove("/tmp/bg_merged.col");
}

TEST(MergerTest, MergeWithDelta) {
    std::vector<int32_t> seg_data(100);
    for (size_t i = 0; i < 100; ++i) seg_data[i] = static_cast<int32_t>(i);
    
    ColumnStats stats{DataType::Int32, 0, 100, 100, true};
    auto seg = Segment::Create("/tmp/merge_delta_seg.col", {stats}, 100);
    
    auto writer = SegmentWriter("/tmp/merge_delta_seg.col");
    writer.AddColumn<DataType::Int32>(seg_data, std::vector<bool>(100, false), EncodingType::Plain);
    auto finished_seg = writer.Finish();
    
    auto delta = std::make_shared<DeltaStore>(1, 1000);
    std::vector<std::vector<uint8_t>> delta_data(1);
    delta_data[0] = {0x64, 0x00, 0x00, 0x00};
    delta->Insert(100, delta_data, 200);
    delta_data[0] = {0x65, 0x00, 0x00, 0x00};
    delta->Insert(101, delta_data, 201);
    
    ColumnMerger merger(DataType::Int32, EncodingType::Plain);
    merger.SetColumnIndex(0);
    
    std::vector<uint8_t> output;
    ColumnStats out_stats;
    
    merger.MergeColumn<DataType::Int32>({finished_seg}, delta, output, out_stats);
    
    EXPECT_EQ(out_stats.total_count, 102);
    EXPECT_EQ(out_stats.min_int, 0);
    EXPECT_EQ(out_stats.max_int, 101);
    
    std::filesystem::remove("/tmp/merge_delta_seg.col");
}

TEST(MergerTest, MergeSortedOutput) {
    std::vector<int32_t> data1 = {5, 1, 3};
    std::vector<int32_t> data2 = {4, 2, 6};
    
    ColumnStats stats1{DataType::Int32, 0, 3, 3, true};
    ColumnStats stats2{DataType::Int32, 0, 3, 3, true};
    
    auto seg1 = Segment::Create("/tmp/merge_sort1.col", {stats1}, 100);
    auto seg2 = Segment::Create("/tmp/merge_sort2.col", {stats2}, 200);
    
    auto w1 = SegmentWriter("/tmp/merge_sort1.col");
    w1.AddColumn<DataType::Int32>(data1, std::vector<bool>(3, false), EncodingType::Plain);
    auto f1 = w1.Finish();
    
    auto w2 = SegmentWriter("/tmp/merge_sort2.col");
    w2.AddColumn<DataType::Int32>(data2, std::vector<bool>(3, false), EncodingType::Plain);
    auto f2 = w2.Finish();
    
    ColumnMerger merger(DataType::Int32, EncodingType::Plain);
    merger.SetColumnIndex(0);
    
    std::vector<uint8_t> output;
    ColumnStats out_stats;
    merger.MergeColumn<DataType::Int32>({f1, f2}, nullptr, output, out_stats);
    
    EXPECT_EQ(out_stats.total_count, 6);
    
    std::filesystem::remove("/tmp/merge_sort1.col");
    std::filesystem::remove("/tmp/merge_sort2.col");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
