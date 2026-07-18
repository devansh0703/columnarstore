#include <gtest/gtest.h>
#include <columnar/scanner.h>
#include <columnar/predicate.h>
#include <columnar/segment.h>
#include <vector>
#include <memory>

using namespace columnar;
using namespace columnar::scanner;
using namespace columnar::predicate;

TEST(ScannerTest, VectorFilterInt32) {
    std::vector<int32_t> data(1000);
    for (size_t i = 0; i < 1000; ++i) data[i] = static_cast<int32_t>(i);
    
    bool mask[1000];
    int32_t value = 500;
    
    VectorFilter<DataType::Int32>::Equal(data.data(), &value, mask, 1000);
    
    int count = 0;
    for (size_t i = 0; i < 1000; ++i) if (mask[i]) ++count;
    EXPECT_EQ(count, 1);
    
    VectorFilter<DataType::Int32>::LessThan(data.data(), &value, mask, 1000);
    count = 0;
    for (size_t i = 0; i < 1000; ++i) if (mask[i]) ++count;
    EXPECT_EQ(count, 500);
    
    VectorFilter<DataType::Int32>::LessEqual(data.data(), &value, mask, 1000);
    count = 0;
    for (size_t i = 0; i < 1000; ++i) if (mask[i]) ++count;
    EXPECT_EQ(count, 501);
}

TEST(ScannerTest, VectorFilterInt64) {
    std::vector<int64_t> data(1000);
    for (size_t i = 0; i < 1000; ++i) data[i] = static_cast<int64_t>(i) * 1000;
    
    bool mask[1000];
    int64_t value = 500000;
    
    VectorFilter<DataType::Int64>::Equal(data.data(), &value, mask, 1000);
    
    int count = 0;
    for (size_t i = 0; i < 1000; ++i) if (mask[i]) ++count;
    EXPECT_EQ(count, 1);
}

TEST(ScannerTest, BlockFilter) {
    BlockFilter filter;
    EXPECT_EQ(filter.CountTrue(), kScanBlockSize);
    
    filter.SetAll(false);
    EXPECT_EQ(filter.CountTrue(), 0);
    
    filter.SetAll(true);
    filter.Data()[10] = false;
    filter.Data()[20] = false;
    filter.Data()[30] = false;
    
    EXPECT_EQ(filter.CountTrue(), kScanBlockSize - 3);
    
    BlockFilter other;
    other.SetAll(false);
    other.Data()[10] = true;
    other.Data()[20] = true;
    
    filter.And(other);
    EXPECT_EQ(filter.CountTrue(), 0);
    
    filter.SetAll(true);
    filter.Or(other);
    EXPECT_EQ(filter.CountTrue(), kScanBlockSize);
}

TEST(ScannerTest, PredicateBuilder) {
    auto pred = columnar::predicate::PredicateBuilder()
        .Eq<DataType::Int32>(0, 10)
        .And()
        .Lt<DataType::Int32>(1, 100)
        .Build();
    
    EXPECT_NE(pred, nullptr);
    EXPECT_EQ(pred->Program().NumInstructions(), 7);
}

TEST(ScannerTest, BytecodeInterpreter) {
    BytecodeProgram program;
    size_t col0 = program.AddColumn(0);
    size_t const10 = program.AddConstant(static_cast<int64_t>(10));
    
    program.Emit(OpCode::LoadColumn, col0);
    program.Emit(OpCode::LoadConst, const10);
    program.Emit(OpCode::CompareEq);
    program.Emit(OpCode::Return);
    
    std::vector<int64_t> col0_data = {1, 10, 5, 10, 20};
    std::vector<const void*> cols = {col0_data.data()};
    std::vector<const bool*> nulls = {nullptr};
    
    bool output[5] = {false};
    
    BytecodeInterpreter interpreter(program);
    interpreter.Execute(cols, nulls, 5, output);
    
    EXPECT_FALSE(output[0]);
    EXPECT_TRUE(output[1]);
    EXPECT_FALSE(output[2]);
    EXPECT_TRUE(output[3]);
    EXPECT_FALSE(output[4]);
}

TEST(ScannerTest, ScanContext) {
    ScanContext ctx;
    ctx.block_size = 4096;
    ctx.num_threads = 4;
    ctx.projected_columns = {0, 1, 2};
    ctx.enable_zone_map_pruning = true;
    ctx.enable_bloom_filter = true;
    ctx.late_materialization = true;
    
    EXPECT_EQ(ctx.block_size, 4096);
    EXPECT_EQ(ctx.num_threads, 4);
    EXPECT_EQ(ctx.projected_columns.size(), 3);
    EXPECT_TRUE(ctx.enable_zone_map_pruning);
    EXPECT_TRUE(ctx.enable_bloom_filter);
    EXPECT_TRUE(ctx.late_materialization);
}

TEST(ScannerTest, ScanStats) {
    ScanStats stats;
    stats.blocks_scanned = 10;
    stats.blocks_pruned_zone_map = 3;
    stats.blocks_pruned_bloom = 2;
    stats.rows_scanned = 10000;
    stats.rows_filtered = 2000;
    stats.rows_returned = 8000;
    stats.scan_time_ms = 5.5;
    
    EXPECT_EQ(stats.blocks_scanned, 10);
    EXPECT_EQ(stats.blocks_pruned_zone_map, 3);
    EXPECT_EQ(stats.blocks_pruned_bloom, 2);
    EXPECT_EQ(stats.rows_scanned, 10000);
    EXPECT_EQ(stats.rows_returned, 8000);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
