#include <gtest/gtest.h>
#include <columnar/scanner.h>
#include <columnar/predicate.h>
#include <columnar/segment.h>
#include <vector>
#include <memory>
#include <filesystem>

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

TEST(ScannerTest, BytecodeInterpreterEagerAndOr) {
    // PredicateBuilder emits operators in eager (infix) position: the And is
    // emitted BEFORE the right-hand operand's instructions. The interpreter
    // must defer the AND until both operands are on the stack.
    BytecodeProgram program;
    size_t col0 = program.AddColumn(0);
    size_t c10 = program.AddConstant(static_cast<int64_t>(10));
    size_t c20 = program.AddConstant(static_cast<int64_t>(20));

    // (col0 < 10) AND (col0 > 20) — eager And placement
    program.Emit(OpCode::LoadColumn, col0);
    program.Emit(OpCode::LoadConst, c10);
    program.Emit(OpCode::CompareLt);
    program.Emit(OpCode::And);
    program.Emit(OpCode::LoadColumn, col0);
    program.Emit(OpCode::LoadConst, c20);
    program.Emit(OpCode::CompareGt);
    program.Emit(OpCode::Return);

    std::vector<int64_t> col0_data = {5, 30, 15};
    std::vector<const void*> cols = {col0_data.data()};
    std::vector<const bool*> nulls = {nullptr};
    bool output[3] = {false, false, false};

    BytecodeInterpreter interpreter(program);
    interpreter.Execute(cols, nulls, 3, output);

    // (5<10 && 5>20)=false, (30<10 && 30>20)=false, (15<10 && 15>20)=false
    EXPECT_FALSE(output[0]);
    EXPECT_FALSE(output[1]);
    EXPECT_FALSE(output[2]);

    // True branch: (col0 < 100) AND (col0 > 20)
    BytecodeProgram program2;
    size_t p2col = program2.AddColumn(0);
    size_t p2c100 = program2.AddConstant(static_cast<int64_t>(100));
    size_t p2c20 = program2.AddConstant(static_cast<int64_t>(20));
    program2.Emit(OpCode::LoadColumn, p2col);
    program2.Emit(OpCode::LoadConst, p2c100);
    program2.Emit(OpCode::CompareLt);
    program2.Emit(OpCode::And);
    program2.Emit(OpCode::LoadColumn, p2col);
    program2.Emit(OpCode::LoadConst, p2c20);
    program2.Emit(OpCode::CompareGt);
    program2.Emit(OpCode::Return);

    std::vector<const void*> cols2 = {col0_data.data()};
    bool output2[3] = {false, false, false};
    BytecodeInterpreter interpreter2(program2);
    interpreter2.Execute(cols2, nulls, 3, output2);

    EXPECT_FALSE(output2[0]);  // 5 < 100 but 5 not > 20
    EXPECT_TRUE(output2[1]);   // 30 < 100 && 30 > 20 → true
    EXPECT_FALSE(output2[2]);  // 15 not > 20
}

TEST(ScannerTest, BytecodeInterpreterPostfixAnd) {
    // Hand-written postfix form (both operands before the And), as used in
    // bench/bm_scan.cpp and previously documented in the README.
    BytecodeProgram program;
    size_t col0 = program.AddColumn(0);
    size_t c500 = program.AddConstant(static_cast<int64_t>(500));
    size_t c1000 = program.AddConstant(static_cast<int64_t>(1000));

    program.Emit(OpCode::LoadColumn, col0);
    program.Emit(OpCode::LoadConst, c500);
    program.Emit(OpCode::CompareGt);
    program.Emit(OpCode::LoadColumn, col0);
    program.Emit(OpCode::LoadConst, c1000);
    program.Emit(OpCode::CompareLt);
    program.Emit(OpCode::And);
    program.Emit(OpCode::Return);

    std::vector<int64_t> col0_data = {400, 600, 1500};
    std::vector<const void*> cols = {col0_data.data()};
    std::vector<const bool*> nulls = {nullptr};
    bool output[3] = {false, false, false};

    BytecodeInterpreter interpreter(program);
    interpreter.Execute(cols, nulls, 3, output);

    EXPECT_FALSE(output[0]);  // 400 not > 500
    EXPECT_TRUE(output[1]);   // 500 < 600 < 1000
    EXPECT_FALSE(output[2]);  // 1500 not < 1000
}

TEST(ScannerTest, SegmentScannerReturnsMatchingRows) {
    // End-to-end: build a segment, scan with a compiled two-column predicate,
    // and verify the output callback receives the matching rows.
    std::string path = "/tmp/test_scanner_rows.col";

    constexpr size_t kRows = 4096;
    std::vector<int32_t> ids(kRows);
    std::vector<int32_t> vals(kRows);
    for (size_t i = 0; i < kRows; ++i) {
        ids[i] = static_cast<int32_t>(i);
        vals[i] = static_cast<int32_t>(i % 64);
    }

    auto writer = SegmentWriter(path);
    writer.AddColumn<DataType::Int32>(ids, std::vector<bool>(kRows, false), EncodingType::Plain);
    writer.AddColumn<DataType::Int32>(vals, std::vector<bool>(kRows, false), EncodingType::RLE);
    auto segment = writer.Finish();
    auto opened = Segment::Open(path);
    ASSERT_NE(opened, nullptr);

    // id < 256 AND value == 7 → matches i in {7, 71, 135, 199} → 4 rows
    auto pred = PredicateBuilder()
                    .Lt<DataType::Int32>(0, 256)
                    .And()
                    .Eq<DataType::Int32>(1, 7)
                    .Build();

    ScanContext ctx;
    ctx.enable_zone_map_pruning = false;
    ctx.enable_bloom_filter = false;

    SegmentScanner scanner(opened, ctx);
    std::vector<int32_t> matched_ids;
    scanner.Scan(*pred, [&](const void** cols, size_t count) {
        for (size_t r = 0; r < count; ++r) {
            matched_ids.push_back(static_cast<const int32_t*>(cols[0])[r]);
        }
    });

    ASSERT_EQ(matched_ids.size(), 4u);
    EXPECT_EQ(matched_ids[0], 7);
    EXPECT_EQ(matched_ids[1], 71);
    EXPECT_EQ(matched_ids[2], 135);
    EXPECT_EQ(matched_ids[3], 199);

    std::filesystem::remove(path);
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
